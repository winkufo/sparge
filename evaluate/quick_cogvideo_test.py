"""
CogVideoX快速准确性测试

快速验证稀疏化效果，只生成1个视频进行对比
运行时间: ~5分钟

用法:
    python evaluate/quick_cogvideo_test.py
    python evaluate/quick_cogvideo_test.py --use_tuned --tuned_path your_tuned.pt
"""

import torch
import os
import gc
import argparse
import numpy as np
from diffusers import CogVideoXPipeline
from diffusers.models import CogVideoXTransformer3DModel
from diffusers.utils import export_to_video

from modify_model.modify_cogvideo import set_spas_sage_attn_cogvideox
from spas_sage_attn.autotune import load_sparse_attention_state_dict
from spas_sage_attn.utils import precision_metric


def parse_args():
    parser = argparse.ArgumentParser(description="CogVideoX Quick Test")
    parser.add_argument("--use_tuned", action="store_true", help="使用调优参数")
    parser.add_argument(
        "--tuned_path",
        type=str,
        default="evaluate/models_dict/CogVideoX-2b_0.06_0.07.pt",
        help="调优参数路径"
    )
    parser.add_argument("--l1", type=float, default=0.06, help="L1误差上限")
    parser.add_argument("--pv_l1", type=float, default=0.065, help="PV L1误差上限")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A panda eating bamboo in a lush forest.",
        help="测试prompt"
    )
    parser.add_argument("--save_videos", action="store_true", help="保存视频")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("CogVideoX快速准确性测试")
    print("="*60)
    print(f"Prompt: {args.prompt}")
    print(f"使用调优参数: {args.use_tuned}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    num_frames = 49
    seed = 42
    
    # ===== 1. 密集版（baseline）=====
    print("\n[1/4] 创建密集版Pipeline...")
    
    transformer_dense = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-2b",
        subfolder="transformer",
        torch_dtype=dtype,
    )
    
    pipe_dense = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        transformer=transformer_dense,
        torch_dtype=dtype,
    ).to(device)
    
    pipe_dense.enable_model_cpu_offload()
    
    print("生成密集版视频...")
    with torch.no_grad():
        video_dense = pipe_dense(
            args.prompt,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=num_frames,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).frames[0]
    
    print(f"✓ 密集版生成完成 ({len(video_dense)}帧)")
    
    # 清理
    del pipe_dense, transformer_dense
    gc.collect()
    torch.cuda.empty_cache()
    
    # ===== 2. 稀疏版 =====
    print("\n[2/4] 创建稀疏版Pipeline...")
    
    transformer_sparse = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-2b",
        subfolder="transformer",
        torch_dtype=dtype,
    )
    
    # 应用稀疏attention
    set_spas_sage_attn_cogvideox(
        transformer_sparse,
        verbose=False,
        l1=args.l1,
        pv_l1=args.pv_l1
    )
    
    # 加载调优参数
    if args.use_tuned and os.path.exists(args.tuned_path):
        print(f"✓ 加载调优参数: {args.tuned_path}")
        tuned_params = torch.load(args.tuned_path)
        load_sparse_attention_state_dict(transformer_sparse, tuned_params)
    
    pipe_sparse = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        transformer=transformer_sparse,
        torch_dtype=dtype,
    ).to(device)
    
    pipe_sparse.enable_model_cpu_offload()
    
    print("生成稀疏版视频...")
    with torch.no_grad():
        video_sparse = pipe_sparse(
            args.prompt,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=num_frames,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).frames[0]
    
    print(f"✓ 稀疏版生成完成 ({len(video_sparse)}帧)")
    
    # ===== 3. 对比差异 =====
    print("\n[3/4] 对比视频差异...")
    
    # 转换为tensor
    frames_dense = torch.from_numpy(np.array(video_dense)).float() / 255.0
    frames_sparse = torch.from_numpy(np.array(video_sparse)).float() / 255.0
    
    print(f"视频形状: {frames_dense.shape}")  # [T, H, W, C]
    
    # 计算整体相似度
    metrics = precision_metric(frames_sparse, frames_dense, verbose=False)
    
    # 计算PSNR
    mse = ((frames_dense - frames_sparse) ** 2).mean().item()
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    
    # 计算每帧差异
    per_frame_diffs = []
    for t in range(len(video_dense)):
        diff = (frames_dense[t] - frames_sparse[t]).abs().mean().item()
        per_frame_diffs.append(diff)
    
    # ===== 4. 结果 =====
    print("\n[4/4] 测试结果")
    print("="*60)
    print(f"帧相似度(Cosine): {metrics['Cossim']:.6f}")
    print(f"帧L1误差: {metrics['L1']:.6f}")
    print(f"帧RMSE: {metrics['RMSE']:.6f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"平均帧差异: {np.mean(per_frame_diffs):.6f}")
    print(f"最大帧差异: {np.max(per_frame_diffs):.6f}")
    
    # 评估
    print("\n" + "="*60)
    print("质量评估")
    print("="*60)
    
    if metrics['Cossim'] > 0.98 and psnr > 40:
        print("🎉 优秀！视频质量几乎无损")
        quality = "excellent"
    elif metrics['Cossim'] > 0.95 and psnr > 35:
        print("✓ 良好！视频质量保持得很好")
        quality = "good"
    elif metrics['Cossim'] > 0.90 and psnr > 30:
        print("⚠ 可接受，有轻微质量下降")
        quality = "acceptable"
    else:
        print("✗ 需要改进，建议调整参数")
        quality = "needs_improvement"
    
    # 保存视频
    if args.save_videos:
        print("\n保存视频...")
        os.makedirs("evaluate/results/quick_test", exist_ok=True)
        
        dense_path = "evaluate/results/quick_test/dense.mp4"
        sparse_path = "evaluate/results/quick_test/sparse.mp4"
        
        export_to_video(video_dense, dense_path, fps=8)
        export_to_video(video_sparse, sparse_path, fps=8)
        
        print(f"✓ 密集版: {dense_path}")
        print(f"✓ 稀疏版: {sparse_path}")
        print("\n请人工对比两个视频的质量")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    
    return quality


if __name__ == "__main__":
    main()

