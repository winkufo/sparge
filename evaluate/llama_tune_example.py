"""
LLaMA 3 模型稀疏化调优脚本

用法:
    # 调优
    python evaluate/llama_tune_example.py --tune --model_out_path llama3_tuned.pt
    
    # 并行调优（更快）
    python evaluate/llama_tune_example.py --tune --parallel_tune --model_out_path llama3_tuned.pt
    
    # 推理
    python evaluate/llama_tune_example.py --model_out_path llama3_tuned.pt --generate
"""

import torch
import os
import gc
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json

from modify_model.modify_llama import (
    set_spas_sage_attn_llama,
    enable_tune_mode,
    get_sparsity_statistics
)
from spas_sage_attn.autotune import (
    extract_sparse_attention_state_dict,
    load_sparse_attention_state_dict,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LLaMA 3 Sparse Attention Tuning")
    
    # 模型配置
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model name or path"
    )
    
    # 调优相关
    parser.add_argument("--tune", action="store_true", help="运行调优")
    parser.add_argument("--parallel_tune", action="store_true", help="启用并行调优")
    parser.add_argument("--l1", type=float, default=0.06, help="QK稀疏化的L1误差上限")
    parser.add_argument("--pv_l1", type=float, default=0.07, help="PV稀疏化的L1误差上限")
    
    # 数据配置
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="调优数据集"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="数据集配置"
    )
    parser.add_argument(
        "--num_tune_samples",
        type=int,
        default=5,
        help="调优样本数量"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="最大序列长度"
    )
    
    # 层选择
    parser.add_argument(
        "--layer_range",
        type=str,
        default=None,
        help="只调优指定层范围，格式: '10,32' 表示第10-31层"
    )
    
    # 输出配置
    parser.add_argument(
        "--model_out_path",
        type=str,
        default="evaluate/models_dict/llama3_tuned.pt",
        help="调优参数保存路径"
    )
    parser.add_argument(
        "--stats_out_path",
        type=str,
        default="evaluate/models_dict/llama3_stats.json",
        help="稀疏度统计保存路径"
    )
    
    # 推理配置
    parser.add_argument("--generate", action="store_true", help="生成文本测试")
    parser.add_argument("--compile", action="store_true", help="使用torch.compile加速")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    return args


def load_tune_data(dataset_name, dataset_config, split="train", num_samples=5, min_length=500):
    """
    加载调优数据
    
    Args:
        dataset_name: 数据集名称
        dataset_config: 数据集配置
        split: 数据集split
        num_samples: 样本数量
        min_length: 最小文本长度
    """
    print(f"\n加载调优数据: {dataset_name} ({dataset_config})")
    
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    
    # 筛选合适的文本
    texts = []
    for item in dataset:
        text = item['text'] if 'text' in item else str(item)
        if len(text) > min_length:
            texts.append(text)
        
        if len(texts) >= num_samples:
            break
    
    if len(texts) < num_samples:
        print(f"⚠️  只找到 {len(texts)} 个合适的样本（期望 {num_samples}）")
    else:
        print(f"✓ 成功加载 {len(texts)} 个样本")
    
    return texts


def tune_model(args):
    """
    调优模型
    """
    print("\n" + "="*60)
    print("开始LLaMA模型调优")
    print("="*60)
    
    # 设置环境变量
    if args.parallel_tune:
        os.environ['PARALLEL_TUNE'] = '1'
        print("✓ 启用并行调优")
    
    os.environ["TUNE_MODE"] = "1"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16  # LLaMA推荐使用bfloat16
    
    # 1. 加载tokenizer
    print(f"\n[1/6] 加载tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 加载模型
    print(f"\n[2/6] 加载模型: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto"
    )
    model.eval()
    
    # 3. 替换attention层
    print(f"\n[3/6] 替换attention层为稀疏版本")
    
    layer_range = None
    if args.layer_range:
        start, end = map(int, args.layer_range.split(','))
        layer_range = (start, end)
        print(f"只替换层 {start}-{end-1}")
    
    model = set_spas_sage_attn_llama(
        model,
        verbose=args.verbose,
        l1=args.l1,
        pv_l1=args.pv_l1,
        layer_range=layer_range
    )
    
    enable_tune_mode(model, True)
    
    # 4. 加载调优数据
    print(f"\n[4/6] 加载调优数据")
    tune_texts = load_tune_data(
        args.dataset,
        args.dataset_config,
        num_samples=args.num_tune_samples
    )
    
    # 5. 运行调优
    print(f"\n[5/6] 运行调优（这可能需要10-30分钟）")
    print("-" * 60)
    
    with torch.no_grad():
        for idx, text in enumerate(tqdm(tune_texts, desc="调优进度")):
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=args.max_length,
                truncation=True,
                padding=False
            ).to(device)
            
            seq_len = inputs.input_ids.size(1)
            
            if seq_len < 256:
                print(f"⚠️  样本 {idx} 太短 ({seq_len} tokens)，跳过")
                continue
            
            if args.verbose:
                print(f"\n调优样本 {idx+1}/{len(tune_texts)}, 长度={seq_len}")
            
            # 前向传播会触发自动调优
            try:
                outputs = model(**inputs)
            except Exception as e:
                print(f"✗ 样本 {idx} 调优失败: {e}")
                continue
            
            # 清理显存
            del inputs, outputs
            gc.collect()
            torch.cuda.empty_cache()
    
    # 6. 保存调优结果
    print(f"\n[6/6] 保存调优结果")
    
    # 保存attention参数
    os.makedirs(os.path.dirname(args.model_out_path), exist_ok=True)
    saved_state_dict = extract_sparse_attention_state_dict(model, verbose=args.verbose)
    torch.save(saved_state_dict, args.model_out_path)
    print(f"✓ 调优参数已保存至: {args.model_out_path}")
    
    # 保存统计信息
    stats = get_sparsity_statistics(model)
    if stats:
        os.makedirs(os.path.dirname(args.stats_out_path), exist_ok=True)
        with open(args.stats_out_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ 统计信息已保存至: {args.stats_out_path}")
        
        # 打印统计摘要
        print("\n" + "="*60)
        print("调优统计摘要")
        print("="*60)
        
        overall_sparsity = sum(s['mean_sparsity'] for s in stats) / len(stats)
        print(f"总体平均稀疏度: {overall_sparsity:.2%}")
        
        print(f"\n{'层':<6} {'稀疏度':<12} {'CDF阈值':<12} {'Sim阈值':<12}")
        print("-" * 48)
        for s in stats[:5]:  # 只显示前5层
            print(f"{s['layer_idx']:<6} {s['mean_sparsity']:<12.2%} "
                  f"{s['cdfthreshd']:<12.4f} {s['simthreshd1']:<12.4f}")
        if len(stats) > 5:
            print(f"... (还有 {len(stats)-5} 层)")
    
    print("\n" + "="*60)
    print("✓ 调优完成!")
    print("="*60)


def generate_text(args):
    """
    使用调优后的模型生成文本
    """
    print("\n" + "="*60)
    print("文本生成测试")
    print("="*60)
    
    os.environ["TUNE_MODE"] = ""  # 关闭调优模式
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    # 1. 加载模型
    print(f"\n[1/3] 加载模型")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto"
    )
    
    # 2. 应用稀疏attention
    print(f"\n[2/3] 应用稀疏attention")
    
    layer_range = None
    if args.layer_range:
        start, end = map(int, args.layer_range.split(','))
        layer_range = (start, end)
    
    model = set_spas_sage_attn_llama(
        model,
        verbose=args.verbose,
        l1=args.l1,
        pv_l1=args.pv_l1,
        layer_range=layer_range
    )
    
    # 加载调优好的参数
    if os.path.exists(args.model_out_path):
        print(f"✓ 加载调优参数: {args.model_out_path}")
        saved_state_dict = torch.load(args.model_out_path)
        load_sparse_attention_state_dict(model, saved_state_dict, verbose=args.verbose)
    else:
        print(f"⚠️  未找到调优参数文件: {args.model_out_path}")
        print("将使用默认参数")
    
    # 可选：编译加速
    if args.compile:
        print("✓ 启用torch.compile加速")
        model = torch.compile(model, mode="reduce-overhead")
    
    model.eval()
    
    # 3. 生成文本
    print(f"\n[3/3] 生成文本")
    print("-" * 60)
    
    test_prompts = [
        "The capital of France is",
        "In the field of machine learning,",
        "Once upon a time, there was a",
        "The theory of relativity states that",
        "Artificial intelligence is"
    ]
    
    with torch.no_grad():
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            print("─" * 40)
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated: {generated_text}")
    
    print("\n" + "="*60)
    print("✓ 生成完成")
    print("="*60)


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("LLaMA 3 稀疏化调优系统")
    print("="*80)
    print(f"模型: {args.model_name}")
    print(f"L1误差上限: {args.l1}")
    print(f"PV L1误差上限: {args.pv_l1}")
    
    if args.tune:
        # 调优模式
        tune_model(args)
    elif args.generate:
        # 生成模式
        generate_text(args)
    else:
        print("\n请指定 --tune 或 --generate")
        print("示例:")
        print("  调优: python llama_tune_example.py --tune --model_out_path llama3_tuned.pt")
        print("  生成: python llama_tune_example.py --generate --model_out_path llama3_tuned.pt")


if __name__ == "__main__":
    main()

