"""
快速LLaMA测试脚本 - 只测试最关键的指标

用于快速验证稀疏化是否工作
运行时间: ~2-3分钟

用法:
    python tests/quick_llama_test.py
    python tests/quick_llama_test.py --model meta-llama/Llama-3.2-1B
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import argparse
from tqdm import tqdm

from spas_sage_attn import spas_sage2_attn_meansim_cuda
from spas_sage_attn.utils import precision_metric


def quick_attention_test(model_name="meta-llama/Llama-3.2-1B"):
    """
    快速测试：直接在真实attention tensor上测试
    """
    print(f"\n{'='*60}")
    print("快速LLaMA稀疏化测试")
    print(f"{'='*60}")
    
    print("\n[1/3] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    print("\n[2/3] 获取真实的Q, K, V...")
    
    # 使用一段真实文本
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to intelligence displayed by animals and humans. Leading AI 
    textbooks define the field as the study of "intelligent agents": any 
    system that perceives its environment and takes actions that maximize 
    its chance of achieving its goals. The term "artificial intelligence" 
    had previously been used to describe machines that mimic and display 
    "human" cognitive skills that are associated with the human mind, such 
    as "learning" and "problem-solving". This definition has since been 
    rejected by major AI researchers who now describe AI in terms of 
    rationality and acting rationally.
    """
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    
    # Hook to capture Q, K, V
    qkv_cache = {}
    
    def capture_qkv(module, input, output):
        # 捕获attention层的输入
        qkv_cache['captured'] = True
    
    # 注册hook到第一个attention层
    first_attn = None
    for name, module in model.named_modules():
        if 'self_attn' in name and hasattr(module, 'q_proj'):
            first_attn = module
            break
    
    if first_attn is None:
        print("未找到attention层")
        return
    
    # 前向传播获取hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1]  # 第一层的输出
    
    # 手动计算Q, K, V
    with torch.no_grad():
        query_states = first_attn.q_proj(hidden_states)
        key_states = first_attn.k_proj(hidden_states)
        value_states = first_attn.v_proj(hidden_states)
        
        # Reshape
        bsz, seq_len, _ = hidden_states.shape
        query_states = query_states.view(bsz, seq_len, first_attn.num_heads, first_attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, first_attn.num_key_value_heads, first_attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seq_len, first_attn.num_key_value_heads, first_attn.head_dim).transpose(1, 2)
        
        # GQA: repeat k/v if needed
        if first_attn.num_key_value_heads != first_attn.num_heads:
            num_groups = first_attn.num_heads // first_attn.num_key_value_heads
            key_states = key_states.repeat_interleave(num_groups, dim=1)
            value_states = value_states.repeat_interleave(num_groups, dim=1)
    
    print(f"捕获到的tensor形状:")
    print(f"  Q: {query_states.shape}")
    print(f"  K: {key_states.shape}")
    print(f"  V: {value_states.shape}")
    
    print("\n[3/3] 测试稀疏attention...")
    
    # 标准attention
    with torch.no_grad():
        attn_dense = F.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=True
        )
    
    # 稀疏attention (多个配置)
    configs = [
        (0.3, 0.95, "高精度"),
        (0.6, 0.98, "中精度"),
        (0.8, 0.99, "高稀疏"),
    ]
    
    print(f"\n{'配置':<12} {'稀疏度':<12} {'Cosine':<12} {'L1':<12} {'评估':<12}")
    print("-" * 72)
    
    for simthreshd1, cdfthreshd, desc in configs:
        with torch.no_grad():
            attn_sparse, sparsity = spas_sage2_attn_meansim_cuda(
                query_states,
                key_states,
                value_states,
                is_causal=True,
                simthreshd1=simthreshd1,
                cdfthreshd=cdfthreshd,
                return_sparsity=True
            )
        
        # 计算指标
        metrics = precision_metric(attn_sparse, attn_dense, verbose=False)
        
        # 评估
        if metrics['Cossim'] > 0.98 and sparsity > 0.3:
            evaluation = "优秀 ✓"
        elif metrics['Cossim'] > 0.95 and sparsity > 0.2:
            evaluation = "良好 ✓"
        elif metrics['Cossim'] > 0.90:
            evaluation = "可接受 ⚠"
        else:
            evaluation = "需改进 ✗"
        
        print(f"{desc:<12} {sparsity:<12.2%} {metrics['Cossim']:<12.6f} "
              f"{metrics['L1']:<12.6f} {evaluation:<12}")
    
    print("\n" + "="*60)
    print("测试完成!")
    print("\n关键观察:")
    print("1. 如果稀疏度 > 30%，说明真实数据确实有稀疏性")
    print("2. 如果Cosine > 0.95，说明稀疏化不影响准确性")
    print("3. 对比随机数据，真实数据的稀疏度应该明显更高")
    print("="*60)


def compare_random_vs_real():
    """
    对比随机数据和真实数据的稀疏度
    这证明你的观察是对的！
    """
    print(f"\n{'='*60}")
    print("对比实验: 随机数据 vs 真实数据")
    print(f"{'='*60}")
    
    model_name = "meta-llama/Llama-3.2-1B"
    
    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # 获取真实Q, K, V (与上面类似的代码)
    text = "The quick brown fox jumps over the lazy dog. " * 20
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    first_attn = None
    for name, module in model.named_modules():
        if 'self_attn' in name and hasattr(module, 'q_proj'):
            first_attn = module
            break
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1]
        
        q_real = first_attn.q_proj(hidden_states)
        k_real = first_attn.k_proj(hidden_states)
        v_real = first_attn.v_proj(hidden_states)
        
        bsz, seq_len, _ = hidden_states.shape
        q_real = q_real.view(bsz, seq_len, first_attn.num_heads, first_attn.head_dim).transpose(1, 2)
        k_real = k_real.view(bsz, seq_len, first_attn.num_key_value_heads, first_attn.head_dim).transpose(1, 2)
        v_real = v_real.view(bsz, seq_len, first_attn.num_key_value_heads, first_attn.head_dim).transpose(1, 2)
        
        if first_attn.num_key_value_heads != first_attn.num_heads:
            num_groups = first_attn.num_heads // first_attn.num_key_value_heads
            k_real = k_real.repeat_interleave(num_groups, dim=1)
            v_real = v_real.repeat_interleave(num_groups, dim=1)
    
    # 生成随机数据（相同形状）
    q_random = torch.randn_like(q_real)
    k_random = torch.randn_like(k_real)
    v_random = torch.randn_like(v_real)
    
    print("\n测试配置: simthreshd1=0.6, cdfthreshd=0.98")
    print("-" * 60)
    
    # 测试真实数据
    with torch.no_grad():
        _, sparsity_real = spas_sage2_attn_meansim_cuda(
            q_real, k_real, v_real,
            is_causal=True,
            simthreshd1=0.6,
            cdfthreshd=0.98,
            return_sparsity=True
        )
    
    # 测试随机数据
    with torch.no_grad():
        _, sparsity_random = spas_sage2_attn_meansim_cuda(
            q_random, k_random, v_random,
            is_causal=True,
            simthreshd1=0.6,
            cdfthreshd=0.98,
            return_sparsity=True
        )
    
    print(f"真实数据稀疏度: {sparsity_real:.2%}")
    print(f"随机数据稀疏度: {sparsity_random:.2%}")
    print(f"差异: {(sparsity_real - sparsity_random)*100:+.1f} 百分点")
    
    print("\n" + "="*60)
    print("结论:")
    print("="*60)
    if sparsity_real > sparsity_random + 0.1:
        print("✓ 真实数据的稀疏度明显更高！")
        print("  这证实了你的观察：真实模型有结构化的attention pattern")
        print("  随机数据无法反映真实的稀疏化效果")
    else:
        print("⚠ 真实数据和随机数据稀疏度接近")
        print("  可能需要调整超参数或使用更长的序列")


def main():
    parser = argparse.ArgumentParser(description='快速LLaMA稀疏化测试')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B',
                       help='模型名称')
    parser.add_argument('--compare', action='store_true',
                       help='对比随机数据和真实数据')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_random_vs_real()
    else:
        quick_attention_test(args.model)


if __name__ == "__main__":
    main()

