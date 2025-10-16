"""
CogVideoX模型准确性测试脚本

对比使用SpargeAttn vs 不使用SpargeAttn的输出差异

用法:
    # 测试不调优的稀疏attention
    python evaluate/cogvideo_accuracy_test.py
    
    # 测试调优后的稀疏attention
    python evaluate/cogvideo_accuracy_test.py \
        --use_tuned \
        --tuned_path evaluate/models_dict/CogVideoX-2b_0.06_0.07.pt
    
    # 指定测试prompts
    python evaluate/cogvideo_accuracy_test.py \
        --prompt_file evaluate/datasets/video/test_prompts.txt
"""

import torch
import os
import gc
import argparse
import json
import numpy as np
from tqdm import tqdm
from diffusers import CogVideoXPipeline
from diffusers.models import CogVideoXTransformer3DModel
from diffusers.utils import export_to_video

from modify_model.modify_cogvideo import set_spas_sage_attn_cogvideox
from spas_sage_attn.autotune import (
    extract_sparse_attention_state_dict,
    load_sparse_attention_state_dict,
)
from spas_sage_attn.utils import precision_metric


def parse_args():
    parser = argparse.ArgumentParser(description="CogVideoX Accuracy Test")
    
    # 模型配置
    parser.add_argument(
        "--model_name",
        type=str,
        default="THUDM/CogVideoX-2b",
        help="模型名称"
    )
    
    # 稀疏配置
    parser.add_argument(
        "--use_tuned",
        action="store_true",
        help="使用调优后的参数"
    )
    parser.add_argument(
        "--tuned_path",
        type=str,
        default="evaluate/models_dict/CogVideoX-2b_0.06_0.07.pt",
        help="调优参数路径"
    )
    parser.add_argument("--l1", type=float, default=0.06, help="L1误差上限")
    parser.add_argument("--pv_l1", type=float, default=0.065, help="PV L1误差上限")
    parser.add_argument(
        "--simthreshd1",
        type=float,
        default=0.6,
        help="相似度阈值（不调优时使用）"
    )
    parser.add_argument(
        "--cdfthreshd",
        type=float,
        default=0.98,
        help="CDF阈值（不调优时使用）"
    )
    
    # 测试配置
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="evaluate/datasets/video/prompts.txt",
        help="测试prompts文件"
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=5,
        help="测试prompt数量"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="推理步数"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=49,
        help="生成帧数"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
        help="Guidance scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    # 输出配置
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluate/results/cogvideo_accuracy",
        help="结果输出目录"
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="保存生成的视频"
    )
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    # 测试选项
    parser.add_argument(
        "--test_intermediate",
        action="store_true",
        help="测试中间层输出（更详细但更慢）"
    )
    
    args = parser.parse_args()
    return args


def load_prompts(prompt_file, num_prompts=5):
    """
    加载测试prompts
    """
    print(f"\n加载测试prompts: {prompt_file}")
    
    if not os.path.exists(prompt_file):
        # 如果文件不存在，使用默认prompts
        print("⚠️  Prompt文件不存在，使用默认prompts")
        prompts = [
            "A panda eating bamboo in a lush forest.",
            "A car driving on a mountain road during sunset.",
            "A cat playing with a ball of yarn.",
            "Ocean waves crashing on a beach.",
            "A city street with people walking."
        ]
    else:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    prompts = prompts[:num_prompts]
    print(f"✓ 加载了 {len(prompts)} 个测试prompts")
    for i, p in enumerate(prompts):
        print(f"  {i+1}. {p}")
    
    return prompts


def create_pipelines(args):
    """
    创建两个pipeline：密集版和稀疏版
    """
    print("\n" + "="*60)
    print("创建测试Pipeline")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    # 1. 创建密集版本（baseline）
    print(f"\n[1/2] 创建密集版Pipeline (baseline)")
    
    transformer_dense = CogVideoXTransformer3DModel.from_pretrained(
        args.model_name,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    
    pipe_dense = CogVideoXPipeline.from_pretrained(
        args.model_name,
        transformer=transformer_dense,
        torch_dtype=dtype,
    ).to(device)
    
    pipe_dense.enable_model_cpu_offload()
    print("✓ 密集版Pipeline创建完成")
    
    # 2. 创建稀疏版本
    print(f"\n[2/2] 创建稀疏版Pipeline")
    
    transformer_sparse = CogVideoXTransformer3DModel.from_pretrained(
        args.model_name,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    
    # 应用稀疏attention
    set_spas_sage_attn_cogvideox(
        transformer_sparse,
        verbose=args.verbose,
        l1=args.l1,
        pv_l1=args.pv_l1
    )
    
    # 加载调优参数（如果指定）
    if args.use_tuned:
        if os.path.exists(args.tuned_path):
            print(f"✓ 加载调优参数: {args.tuned_path}")
            tuned_params = torch.load(args.tuned_path)
            load_sparse_attention_state_dict(
                transformer_sparse,
                tuned_params,
                verbose=args.verbose
            )
        else:
            print(f"⚠️  未找到调优参数: {args.tuned_path}")
            print("将使用默认参数")
    else:
        print(f"使用默认参数: simthreshd1={args.simthreshd1}, cdfthreshd={args.cdfthreshd}")
    
    pipe_sparse = CogVideoXPipeline.from_pretrained(
        args.model_name,
        transformer=transformer_sparse,
        torch_dtype=dtype,
    ).to(device)
    
    pipe_sparse.enable_model_cpu_offload()
    print("✓ 稀疏版Pipeline创建完成")
    
    return pipe_dense, pipe_sparse


def test_single_prompt(
    pipe_dense,
    pipe_sparse,
    prompt,
    args,
    save_prefix="test"
):
    """
    测试单个prompt的输出差异
    """
    generator_dense = torch.Generator(device="cuda").manual_seed(args.seed)
    generator_sparse = torch.Generator(device="cuda").manual_seed(args.seed)
    
    # 1. 密集版生成
    print(f"\n  生成密集版视频...")
    with torch.no_grad():
        # 如果需要测试中间层，设置output_hidden_states
        if args.test_intermediate:
            # 注意：这需要修改pipeline来支持返回中间状态
            outputs_dense = pipe_dense(
                prompt.strip(),
                num_videos_per_prompt=1,
                num_inference_steps=args.num_inference_steps,
                num_frames=args.num_frames,
                guidance_scale=args.guidance_scale,
                generator=generator_dense,
            )
        else:
            outputs_dense = pipe_dense(
                prompt.strip(),
                num_videos_per_prompt=1,
                num_inference_steps=args.num_inference_steps,
                num_frames=args.num_frames,
                guidance_scale=args.guidance_scale,
                generator=generator_dense,
            )
    
    frames_dense = outputs_dense.frames[0]
    
    # 2. 稀疏版生成
    print(f"  生成稀疏版视频...")
    with torch.no_grad():
        outputs_sparse = pipe_sparse(
            prompt.strip(),
            num_videos_per_prompt=1,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            generator=generator_sparse,
        )
    
    frames_sparse = outputs_sparse.frames[0]
    
    # 3. 对比视频帧
    print(f"  对比生成的视频...")
    
    # 转换为tensor进行对比
    frames_dense_tensor = torch.from_numpy(
        np.array(frames_dense)
    ).float() / 255.0  # [T, H, W, C]
    
    frames_sparse_tensor = torch.from_numpy(
        np.array(frames_sparse)
    ).float() / 255.0
    
    # 计算帧级别的相似度
    frame_metrics = precision_metric(
        frames_sparse_tensor,
        frames_dense_tensor,
        verbose=False
    )
    
    # 计算每帧的差异
    per_frame_diffs = []
    for t in range(len(frames_dense)):
        frame_d = frames_dense_tensor[t]
        frame_s = frames_sparse_tensor[t]
        diff = (frame_d - frame_s).abs().mean().item()
        per_frame_diffs.append(diff)
    
    # 计算PSNR和SSIM（简化版本）
    mse = ((frames_dense_tensor - frames_sparse_tensor) ** 2).mean().item()
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    
    result = {
        'prompt': prompt,
        'num_frames': len(frames_dense),
        'frame_cosine': frame_metrics['Cossim'],
        'frame_l1': frame_metrics['L1'],
        'frame_rmse': frame_metrics['RMSE'],
        'psnr': psnr,
        'per_frame_diff_mean': np.mean(per_frame_diffs),
        'per_frame_diff_max': np.max(per_frame_diffs),
        'per_frame_diffs': per_frame_diffs,
    }
    
    # 4. 保存视频（如果需要）
    if args.save_videos:
        video_dir = os.path.join(args.output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        
        dense_path = os.path.join(video_dir, f"{save_prefix}_dense.mp4")
        sparse_path = os.path.join(video_dir, f"{save_prefix}_sparse.mp4")
        
        export_to_video(frames_dense, dense_path, fps=8)
        export_to_video(frames_sparse, sparse_path, fps=8)
        
        result['dense_video_path'] = dense_path
        result['sparse_video_path'] = sparse_path
        
        print(f"  ✓ 视频已保存: {dense_path}, {sparse_path}")
    
    return result


def test_all_prompts(pipe_dense, pipe_sparse, prompts, args):
    """
    测试所有prompts
    """
    print("\n" + "="*60)
    print("开始测试")
    print("="*60)
    
    all_results = []
    
    for idx, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"测试 {idx+1}/{len(prompts)}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        
        try:
            result = test_single_prompt(
                pipe_dense,
                pipe_sparse,
                prompt,
                args,
                save_prefix=f"prompt_{idx}"
            )
            all_results.append(result)
            
            # 打印当前结果
            print(f"\n  结果:")
            print(f"    帧相似度(Cosine): {result['frame_cosine']:.6f}")
            print(f"    帧L1误差: {result['frame_l1']:.6f}")
            print(f"    PSNR: {result['psnr']:.2f} dB")
            print(f"    平均帧差异: {result['per_frame_diff_mean']:.6f}")
            
        except Exception as e:
            print(f"  ✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 清理显存
        gc.collect()
        torch.cuda.empty_cache()
    
    return all_results


def compute_statistics(all_results):
    """
    计算统计摘要
    """
    print("\n" + "="*60)
    print("统计摘要")
    print("="*60)
    
    if not all_results:
        print("没有有效的测试结果")
        return {}
    
    stats = {}
    
    metrics = ['frame_cosine', 'frame_l1', 'frame_rmse', 'psnr', 
               'per_frame_diff_mean', 'per_frame_diff_max']
    
    for metric in metrics:
        values = [r[metric] for r in all_results]
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }
    
    # 打印表格
    print(f"\n{'指标':<25} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12}")
    print("-" * 73)
    
    metrics_display = [
        ('frame_cosine', '帧相似度(Cosine)'),
        ('frame_l1', '帧L1误差'),
        ('psnr', 'PSNR (dB)'),
        ('per_frame_diff_mean', '平均帧差异'),
    ]
    
    for key, display_name in metrics_display:
        s = stats[key]
        print(f"{display_name:<25} {s['mean']:<12.6f} {s['std']:<12.6f} "
              f"{s['min']:<12.6f} {s['max']:<12.6f}")
    
    return stats


def evaluate_quality(stats):
    """
    评估整体质量
    """
    print("\n" + "="*60)
    print("质量评估")
    print("="*60)
    
    if not stats:
        print("无法评估（没有统计数据）")
        return 0
    
    score = 0
    max_score = 100
    
    # 帧相似度 (40分)
    frame_cosine = stats['frame_cosine']['mean']
    if frame_cosine > 0.98:
        score += 40
        print(f"✓ 帧相似度: {frame_cosine:.4f} (优秀) - 40/40分")
    elif frame_cosine > 0.95:
        score += 30
        print(f"✓ 帧相似度: {frame_cosine:.4f} (良好) - 30/40分")
    elif frame_cosine > 0.90:
        score += 20
        print(f"⚠ 帧相似度: {frame_cosine:.4f} (可接受) - 20/40分")
    else:
        score += 10
        print(f"✗ 帧相似度: {frame_cosine:.4f} (需改进) - 10/40分")
    
    # PSNR (40分) - 越高越好
    psnr = stats['psnr']['mean']
    if psnr > 40:
        score += 40
        print(f"✓ PSNR: {psnr:.2f} dB (优秀) - 40/40分")
    elif psnr > 35:
        score += 30
        print(f"✓ PSNR: {psnr:.2f} dB (良好) - 30/40分")
    elif psnr > 30:
        score += 20
        print(f"⚠ PSNR: {psnr:.2f} dB (可接受) - 20/40分")
    else:
        score += 10
        print(f"✗ PSNR: {psnr:.2f} dB (需改进) - 10/40分")
    
    # 帧L1误差 (20分) - 越小越好
    frame_l1 = stats['frame_l1']['mean']
    if frame_l1 < 0.02:
        score += 20
        print(f"✓ 帧L1误差: {frame_l1:.6f} (优秀) - 20/20分")
    elif frame_l1 < 0.05:
        score += 15
        print(f"✓ 帧L1误差: {frame_l1:.6f} (良好) - 15/20分")
    elif frame_l1 < 0.10:
        score += 10
        print(f"⚠ 帧L1误差: {frame_l1:.6f} (可接受) - 10/20分")
    else:
        score += 5
        print(f"✗ 帧L1误差: {frame_l1:.6f} (需改进) - 5/20分")
    
    # 总评
    print(f"\n{'='*60}")
    print(f"总分: {score}/{max_score}")
    print(f"{'='*60}")
    
    if score >= 90:
        print("🎉 优秀！视频质量几乎无损")
    elif score >= 75:
        print("✓ 良好，视频质量保持得很好")
    elif score >= 60:
        print("⚠ 可接受，有轻微质量下降")
    else:
        print("✗ 需要改进，建议调整参数或重新调优")
    
    return score


def save_results(args, all_results, stats, score):
    """
    保存结果到文件
    """
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成文件名
    if args.use_tuned:
        filename = "cogvideo_accuracy_tuned.json"
    else:
        filename = "cogvideo_accuracy_default.json"
    
    output_path = os.path.join(args.output_dir, filename)
    
    # 汇总结果
    results = {
        'config': {
            'model_name': args.model_name,
            'use_tuned': args.use_tuned,
            'tuned_path': args.tuned_path if args.use_tuned else None,
            'l1': args.l1,
            'pv_l1': args.pv_l1,
            'num_prompts': len(all_results),
            'num_inference_steps': args.num_inference_steps,
            'num_frames': args.num_frames,
            'seed': args.seed,
        },
        'statistics': stats,
        'detailed_results': all_results,
        'overall_score': score
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 结果已保存至: {output_path}")


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("CogVideoX模型准确性测试")
    print("="*80)
    print(f"模型: {args.model_name}")
    print(f"使用调优参数: {args.use_tuned}")
    if args.use_tuned:
        print(f"调优参数路径: {args.tuned_path}")
    print(f"测试prompts数: {args.num_prompts}")
    print(f"推理步数: {args.num_inference_steps}")
    print(f"生成帧数: {args.num_frames}")
    
    # 1. 加载prompts
    prompts = load_prompts(args.prompt_file, args.num_prompts)
    
    # 2. 创建pipelines
    pipe_dense, pipe_sparse = create_pipelines(args)
    
    # 3. 测试所有prompts
    all_results = test_all_prompts(pipe_dense, pipe_sparse, prompts, args)
    
    # 4. 计算统计
    stats = compute_statistics(all_results)
    
    # 5. 评估质量
    score = evaluate_quality(stats)
    
    # 6. 保存结果
    save_results(args, all_results, stats, score)
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
    
    if args.save_videos:
        print(f"\n生成的视频保存在: {os.path.join(args.output_dir, 'videos')}")


if __name__ == "__main__":
    main()

