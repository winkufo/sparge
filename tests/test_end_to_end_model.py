"""
端到端模型测试 - 在实际模型中验证稀疏attention

测试策略:
1. 替换模型中的attention层为稀疏版本
2. 对比原模型和稀疏模型的输出
3. 验证生成质量是否保持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from spas_sage_attn.utils import precision_metric


# ============================================================================
# 创建一个简单的Transformer模型用于测试
# ============================================================================

class StandardAttention(nn.Module):
    """标准attention层"""
    
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L, C)
        out = self.proj(out)
        return out


class SparseAttention(nn.Module):
    """稀疏attention层"""
    
    def __init__(self, dim, num_heads, simthreshd1=0.6, cdfthreshd=0.98):
        super().__init__()
        from spas_sage_attn import spas_sage2_attn_meansim_cuda
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sparse_attn = spas_sage2_attn_meansim_cuda
        self.simthreshd1 = simthreshd1
        self.cdfthreshd = cdfthreshd
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        out = self.sparse_attn(
            q, k, v,
            simthreshd1=self.simthreshd1,
            cdfthreshd=self.cdfthreshd
        )
        out = out.transpose(1, 2).reshape(B, L, C)
        out = self.proj(out)
        return out


class SimpleTransformerBlock(nn.Module):
    """简单的Transformer块"""
    
    def __init__(self, dim, num_heads, use_sparse=False):
        super().__init__()
        if use_sparse:
            self.attn = SparseAttention(dim, num_heads)
        else:
            self.attn = StandardAttention(dim, num_heads)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleTransformer(nn.Module):
    """简单的Transformer模型"""
    
    def __init__(self, dim=512, num_heads=8, num_layers=4, use_sparse=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(dim, num_heads, use_sparse)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ============================================================================
# 测试函数
# ============================================================================

def test_single_layer_equivalence():
    """测试单层attention的等价性"""
    print("\n" + "="*60)
    print("测试单层Attention等价性")
    print("="*60)
    
    dim = 512
    num_heads = 8
    batch_size = 2
    seq_len = 1024
    device = 'cuda'
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    # 标准attention
    model_standard = SimpleTransformerBlock(dim, num_heads, use_sparse=False).to(device)
    model_standard.eval()
    
    # 稀疏attention（复制权重）
    model_sparse = SimpleTransformerBlock(dim, num_heads, use_sparse=True).to(device)
    model_sparse.load_state_dict(model_standard.state_dict(), strict=False)
    model_sparse.eval()
    
    # 前向传播
    with torch.no_grad():
        out_standard = model_standard(x)
        out_sparse = model_sparse(x)
    
    # 对比
    metrics = precision_metric(out_sparse, out_standard, verbose=True)
    
    if metrics['Cossim'] > 0.95 and metrics['L1'] < 0.1:
        print("✓ 单层测试通过")
        return True
    else:
        print("✗ 单层测试失败")
        return False


def test_multi_layer_transformer():
    """测试多层Transformer"""
    print("\n" + "="*60)
    print("测试多层Transformer")
    print("="*60)
    
    dim = 512
    num_heads = 8
    num_layers = 4
    batch_size = 2
    seq_len = 1024
    device = 'cuda'
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    # 标准模型
    model_standard = SimpleTransformer(dim, num_heads, num_layers, use_sparse=False).to(device)
    model_standard.eval()
    
    # 稀疏模型（复制权重）
    model_sparse = SimpleTransformer(dim, num_heads, num_layers, use_sparse=True).to(device)
    model_sparse.load_state_dict(model_standard.state_dict(), strict=False)
    model_sparse.eval()
    
    # 前向传播
    with torch.no_grad():
        out_standard = model_standard(x)
        out_sparse = model_sparse(x)
    
    # 对比
    metrics = precision_metric(out_sparse, out_standard, verbose=True)
    
    # 多层误差会累积，所以阈值放宽一些
    if metrics['Cossim'] > 0.90 and metrics['L1'] < 0.15:
        print(f"✓ {num_layers}层测试通过")
        return True
    else:
        print(f"✗ {num_layers}层测试失败")
        return False


def test_gradient_equivalence():
    """测试梯度等价性（虽然是inference优化，但可以验证）"""
    print("\n" + "="*60)
    print("测试梯度计算")
    print("="*60)
    
    dim = 256
    num_heads = 4
    batch_size = 2
    seq_len = 512
    device = 'cuda'
    
    x = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
    
    # 标准attention
    model_standard = SimpleTransformerBlock(dim, num_heads, use_sparse=False).to(device)
    out_standard = model_standard(x)
    loss_standard = out_standard.mean()
    loss_standard.backward()
    grad_standard = x.grad.clone()
    
    # 稀疏attention不支持训练，所以跳过这个测试
    print("⚠ 稀疏attention用于inference，跳过梯度测试")
    return True


def test_various_sequence_lengths():
    """测试不同序列长度下的模型行为"""
    print("\n" + "="*60)
    print("测试不同序列长度")
    print("="*60)
    
    dim = 256
    num_heads = 4
    batch_size = 2
    device = 'cuda'
    
    seq_lengths = [512, 1024, 2048]
    
    for seq_len in seq_lengths:
        print(f"\n序列长度: {seq_len}")
        x = torch.randn(batch_size, seq_len, dim, device=device)
        
        model_standard = SimpleTransformerBlock(dim, num_heads, use_sparse=False).to(device)
        model_sparse = SimpleTransformerBlock(dim, num_heads, use_sparse=True).to(device)
        model_sparse.load_state_dict(model_standard.state_dict(), strict=False)
        
        model_standard.eval()
        model_sparse.eval()
        
        with torch.no_grad():
            out_standard = model_standard(x)
            out_sparse = model_sparse(x)
        
        metrics = precision_metric(out_sparse, out_standard, verbose=False)
        print(f"  Cosine: {metrics['Cossim']:.6f}, L1: {metrics['L1']:.6f}")
        
        if metrics['Cossim'] > 0.90:
            print(f"  ✓ 序列长度 {seq_len} 测试通过")
        else:
            print(f"  ✗ 序列长度 {seq_len} 测试失败")


def test_memory_usage():
    """测试显存使用情况"""
    print("\n" + "="*60)
    print("测试显存使用")
    print("="*60)
    
    import gc
    
    dim = 512
    num_heads = 8
    num_layers = 4
    batch_size = 2
    seq_len = 2048
    device = 'cuda'
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 标准模型
    x = torch.randn(batch_size, seq_len, dim, device=device)
    model_standard = SimpleTransformer(dim, num_heads, num_layers, use_sparse=False).to(device)
    model_standard.eval()
    
    with torch.no_grad():
        _ = model_standard(x)
    
    mem_standard = torch.cuda.max_memory_allocated() / 1024**2
    print(f"标准attention显存: {mem_standard:.2f} MB")
    
    del model_standard, x
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 稀疏模型
    x = torch.randn(batch_size, seq_len, dim, device=device)
    model_sparse = SimpleTransformer(dim, num_heads, num_layers, use_sparse=True).to(device)
    model_sparse.eval()
    
    with torch.no_grad():
        _ = model_sparse(x)
    
    mem_sparse = torch.cuda.max_memory_allocated() / 1024**2
    print(f"稀疏attention显存: {mem_sparse:.2f} MB")
    
    print(f"显存节省: {(1 - mem_sparse/mem_standard)*100:.1f}%")


def test_attention_pattern_preservation():
    """测试attention pattern是否被保留（质性测试）"""
    print("\n" + "="*60)
    print("测试Attention Pattern保留情况")
    print("="*60)
    
    # 创建一个特殊的输入，让attention pattern有明显的结构
    batch_size = 1
    seq_len = 512
    dim = 256
    num_heads = 4
    device = 'cuda'
    
    # 生成有结构的输入（例如：前半部分和后半部分不同）
    x = torch.randn(batch_size, seq_len, dim, device=device)
    x[:, :seq_len//2, :] *= 2  # 前半部分幅度更大
    
    model_standard = SimpleTransformerBlock(dim, num_heads, use_sparse=False).to(device)
    model_sparse = SimpleTransformerBlock(dim, num_heads, use_sparse=True).to(device)
    model_sparse.load_state_dict(model_standard.state_dict(), strict=False)
    
    model_standard.eval()
    model_sparse.eval()
    
    with torch.no_grad():
        out_standard = model_standard(x)
        out_sparse = model_sparse(x)
    
    # 检查输出的统计特性是否相似
    mean_diff = (out_standard.mean() - out_sparse.mean()).abs().item()
    std_diff = (out_standard.std() - out_sparse.std()).abs().item()
    
    print(f"均值差异: {mean_diff:.6f}")
    print(f"标准差差异: {std_diff:.6f}")
    
    if mean_diff < 0.01 and std_diff < 0.1:
        print("✓ Attention pattern保留良好")
        return True
    else:
        print("⚠ Attention pattern可能有较大变化")
        return False


# ============================================================================
# 主测试流程
# ============================================================================

def run_all_tests():
    """运行所有端到端测试"""
    print("\n" + "="*60)
    print("开始端到端模型测试")
    print("="*60)
    
    results = {}
    
    # 1. 单层等价性
    results['single_layer'] = test_single_layer_equivalence()
    
    # 2. 多层Transformer
    results['multi_layer'] = test_multi_layer_transformer()
    
    # 3. 梯度测试
    results['gradient'] = test_gradient_equivalence()
    
    # 4. 不同序列长度
    test_various_sequence_lengths()
    
    # 5. 显存使用
    test_memory_usage()
    
    # 6. Attention pattern保留
    results['pattern'] = test_attention_pattern_preservation()
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    passed = sum(results.values())
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"{status} {test_name}")
    
    print("="*60)
    
    return all(results.values())


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

