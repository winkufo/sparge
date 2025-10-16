"""
快速测试脚本 - 用于日常开发验证

用法:
    python tests/quick_test.py
    python tests/quick_test.py --seq_len 2048 --sparse_level high
"""

import torch
import argparse
from spas_sage_attn import spas_sage2_attn_meansim_cuda
from spas_sage_attn.utils import precision_metric
import torch.nn.functional as F


def quick_test(
    seq_len=1024,
    sparse_level='medium',
    is_causal=False,
    device='cuda'
):
    """快速准确性测试"""
    
    # 稀疏度配置
    configs = {
        'low': (0.8, 0.99, "低稀疏度-高精度"),
        'medium': (0.6, 0.98, "中稀疏度-中精度"),
        'high': (0.3, 0.95, "高稀疏度-低精度")
    }
    
    simthreshd1, cdfthreshd, desc = configs[sparse_level]
    
    print("\n" + "="*60)
    print(f"快速测试: {desc}")
    print(f"序列长度: {seq_len}, Causal: {is_causal}")
    print("="*60)
    
    # 生成测试数据
    batch_size = 2
    num_heads = 8
    head_dim = 64
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                   dtype=torch.float16, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device=device)
    
    print(f"\n输入形状: Q={q.shape}, K={k.shape}, V={v.shape}")
    
    # 标准attention
    print("\n计算标准attention (baseline)...")
    with torch.no_grad():
        o_baseline = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    
    # 稀疏attention
    print("计算稀疏attention...")
    with torch.no_grad():
        o_sparse, sparsity = spas_sage2_attn_meansim_cuda(
            q, k, v,
            is_causal=is_causal,
            simthreshd1=simthreshd1,
            cdfthreshd=cdfthreshd,
            return_sparsity=True
        )
    
    # 精度评估
    print("\n精度指标:")
    print("-" * 60)
    metrics = precision_metric(o_sparse, o_baseline, verbose=True)
    print(f"稀疏度: {sparsity:.2%}")
    
    # 判定结果
    print("\n测试结果:")
    print("-" * 60)
    passed = True
    
    if metrics['Cossim'] > 0.95:
        print(f"✓ Cosine相似度 {metrics['Cossim']:.6f} > 0.95")
    else:
        print(f"✗ Cosine相似度 {metrics['Cossim']:.6f} <= 0.95")
        passed = False
    
    if metrics['L1'] < 0.1:
        print(f"✓ L1误差 {metrics['L1']:.6f} < 0.1")
    else:
        print(f"✗ L1误差 {metrics['L1']:.6f} >= 0.1")
        passed = False
    
    if sparsity > 0.05:
        print(f"✓ 稀疏度 {sparsity:.2%} > 5%")
    else:
        print(f"⚠ 稀疏度 {sparsity:.2%} <= 5% (稀疏效果不明显)")
    
    print("\n" + "="*60)
    if passed:
        print("✓ 测试通过!")
    else:
        print("✗ 测试失败!")
    print("="*60)
    
    return passed


def benchmark_speed(seq_len=2048, device='cuda'):
    """简单的速度对比"""
    import time
    
    print("\n" + "="*60)
    print(f"速度测试 (序列长度: {seq_len})")
    print("="*60)
    
    batch_size = 2
    num_heads = 8
    head_dim = 64
    num_runs = 10
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device=device)
    
    # Warm up
    for _ in range(3):
        _ = F.scaled_dot_product_attention(q, k, v)
        _ = spas_sage2_attn_meansim_cuda(q, k, v)
    
    torch.cuda.synchronize()
    
    # 标准attention
    start = time.time()
    for _ in range(num_runs):
        o_baseline = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
    baseline_time = (time.time() - start) / num_runs
    
    # 稀疏attention
    start = time.time()
    for _ in range(num_runs):
        o_sparse = spas_sage2_attn_meansim_cuda(q, k, v)
        torch.cuda.synchronize()
    sparse_time = (time.time() - start) / num_runs
    
    speedup = baseline_time / sparse_time
    
    print(f"标准attention: {baseline_time*1000:.2f} ms")
    print(f"稀疏attention: {sparse_time*1000:.2f} ms")
    print(f"加速比: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description='快速测试脚本')
    parser.add_argument('--seq_len', type=int, default=1024, 
                       help='序列长度 (default: 1024)')
    parser.add_argument('--sparse_level', type=str, default='medium',
                       choices=['low', 'medium', 'high'],
                       help='稀疏度级别 (default: medium)')
    parser.add_argument('--causal', action='store_true',
                       help='使用causal mask')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行速度测试')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (default: cuda)')
    
    args = parser.parse_args()
    
    # 准确性测试
    passed = quick_test(
        seq_len=args.seq_len,
        sparse_level=args.sparse_level,
        is_causal=args.causal,
        device=args.device
    )
    
    # 速度测试
    if args.benchmark:
        benchmark_speed(seq_len=args.seq_len, device=args.device)
    
    return 0 if passed else 1


if __name__ == "__main__":
    exit(main())

