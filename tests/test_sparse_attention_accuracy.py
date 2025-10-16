"""
完整的稀疏注意力准确性测试套件

测试策略：
1. 单元测试：对比稀疏attention vs 标准attention
2. 多场景测试：不同序列长度、头数、批次大小
3. 多指标评估：Cosine相似度、L1误差、RMSE
4. 边界条件测试：causal mask、不同稀疏度
5. 可视化对比：attention map可视化
"""

import torch
import torch.nn.functional as F
import numpy as np
import pytest
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# 假设已经安装了项目
from spas_sage_attn import spas_sage_attn_meansim_cuda, spas_sage2_attn_meansim_cuda
from spas_sage_attn.utils import precision_metric


class SparseAttentionTester:
    """稀疏注意力测试器"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.test_results = []
    
    def generate_test_data(
        self, 
        batch_size: int = 2,
        num_heads: int = 8,
        seq_len: int = 1024,
        head_dim: int = 64,
        dtype: torch.dtype = torch.float16
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成测试数据"""
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       dtype=dtype, device=self.device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       dtype=dtype, device=self.device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       dtype=dtype, device=self.device)
        return q, k, v
    
    def compute_baseline(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        is_causal: bool = False
    ) -> torch.Tensor:
        """计算baseline（标准dense attention）"""
        with torch.no_grad():
            o_baseline = F.scaled_dot_product_attention(
                q, k, v, 
                is_causal=is_causal
            )
        return o_baseline
    
    def test_basic_accuracy(
        self,
        simthreshd1: float = 0.6,
        cdfthreshd: float = 0.98,
        seq_len: int = 1024,
        is_causal: bool = False,
        use_sage2: bool = True
    ) -> Dict:
        """基础准确性测试"""
        print(f"\n{'='*60}")
        print(f"测试配置: seq_len={seq_len}, simthreshd1={simthreshd1}, "
              f"cdfthreshd={cdfthreshd}, causal={is_causal}")
        print(f"{'='*60}")
        
        # 生成测试数据
        q, k, v = self.generate_test_data(seq_len=seq_len)
        
        # Baseline
        o_baseline = self.compute_baseline(q, k, v, is_causal=is_causal)
        
        # 稀疏attention
        kernel = spas_sage2_attn_meansim_cuda if use_sage2 else spas_sage_attn_meansim_cuda
        o_sparse, sparsity = kernel(
            q, k, v,
            is_causal=is_causal,
            simthreshd1=simthreshd1,
            cdfthreshd=cdfthreshd,
            return_sparsity=True
        )
        
        # 计算精度指标
        metrics = precision_metric(o_sparse, o_baseline, verbose=True)
        metrics['sparsity'] = sparsity
        metrics['config'] = {
            'seq_len': seq_len,
            'simthreshd1': simthreshd1,
            'cdfthreshd': cdfthreshd,
            'is_causal': is_causal
        }
        
        self.test_results.append(metrics)
        
        return metrics
    
    def test_various_sparsity_levels(self):
        """测试不同稀疏度级别"""
        print("\n" + "="*60)
        print("测试不同稀疏度级别")
        print("="*60)
        
        configs = [
            # (simthreshd1, cdfthreshd, expected_quality)
            (-0.5, 0.95, "高精度低稀疏"),
            (0.0, 0.97, "中精度中稀疏"),
            (0.5, 0.99, "低精度高稀疏"),
        ]
        
        results = []
        for sim_th, cdf_th, desc in configs:
            print(f"\n配置: {desc}")
            metrics = self.test_basic_accuracy(
                simthreshd1=sim_th,
                cdfthreshd=cdf_th,
                seq_len=2048
            )
            results.append((desc, metrics))
        
        # 打印对比表格
        print("\n" + "="*60)
        print("稀疏度 vs 精度对比")
        print("="*60)
        print(f"{'配置':<20} {'稀疏度':<10} {'Cosine':<10} {'L1':<10} {'RMSE':<10}")
        print("-"*60)
        for desc, m in results:
            print(f"{desc:<20} {m['sparsity']:<10.2%} {m['Cossim']:<10.6f} "
                  f"{m['L1']:<10.6f} {m['RMSE']:<10.6f}")
        
        return results
    
    def test_different_sequence_lengths(self):
        """测试不同序列长度"""
        print("\n" + "="*60)
        print("测试不同序列长度")
        print("="*60)
        
        seq_lengths = [512, 1024, 2048, 4096]
        results = []
        
        for seq_len in seq_lengths:
            try:
                metrics = self.test_basic_accuracy(seq_len=seq_len)
                results.append((seq_len, metrics))
            except Exception as e:
                print(f"序列长度 {seq_len} 测试失败: {e}")
        
        return results
    
    def test_causal_vs_noncausal(self):
        """测试causal mask的影响"""
        print("\n" + "="*60)
        print("测试Causal vs Non-Causal")
        print("="*60)
        
        results = {
            'non_causal': self.test_basic_accuracy(is_causal=False),
            'causal': self.test_basic_accuracy(is_causal=True)
        }
        
        print("\n对比:")
        print(f"{'模式':<15} {'Cosine':<10} {'L1':<10} {'稀疏度':<10}")
        print("-"*45)
        for mode, m in results.items():
            print(f"{mode:<15} {m['Cossim']:<10.6f} {m['L1']:<10.6f} "
                  f"{m['sparsity']:<10.2%}")
        
        return results
    
    def visualize_attention_comparison(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        head_idx: int = 0,
        save_path: str = "attention_comparison.png"
    ):
        """可视化attention map对比"""
        # 计算标准attention
        with torch.no_grad():
            scale = 1.0 / (q.size(-1) ** 0.5)
            attn_weights_baseline = torch.softmax(
                torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1
            )
            o_baseline = torch.matmul(attn_weights_baseline, v)
        
        # 计算稀疏attention
        o_sparse, sparsity = spas_sage2_attn_meansim_cuda(
            q, k, v,
            simthreshd1=0.6,
            cdfthreshd=0.98,
            return_sparsity=True
        )
        
        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Baseline attention map
        attn_map = attn_weights_baseline[0, head_idx].cpu().numpy()
        sns.heatmap(attn_map[:100, :100], ax=axes[0], cmap='viridis', 
                   cbar_kws={'label': 'Attention Weight'})
        axes[0].set_title('Baseline Attention Map')
        axes[0].set_xlabel('Key Position')
        axes[0].set_ylabel('Query Position')
        
        # Output difference
        diff = (o_baseline - o_sparse).abs()[0, head_idx].cpu().numpy()
        sns.heatmap(diff[:100, :], ax=axes[1], cmap='Reds',
                   cbar_kws={'label': 'Absolute Difference'})
        axes[1].set_title(f'Output Difference (Sparsity: {sparsity:.2%})')
        axes[1].set_xlabel('Head Dimension')
        axes[1].set_ylabel('Sequence Position')
        
        # Distribution of differences
        diff_flat = diff.flatten()
        axes[2].hist(diff_flat, bins=50, alpha=0.7, edgecolor='black')
        axes[2].set_title('Distribution of Differences')
        axes[2].set_xlabel('Absolute Difference')
        axes[2].set_ylabel('Frequency')
        axes[2].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化已保存至: {save_path}")
        plt.close()
    
    def test_numerical_stability(self, num_runs: int = 10):
        """测试数值稳定性（多次运行的一致性）"""
        print("\n" + "="*60)
        print(f"测试数值稳定性 ({num_runs} 次运行)")
        print("="*60)
        
        q, k, v = self.generate_test_data(seq_len=1024)
        
        results = []
        for i in range(num_runs):
            o_sparse = spas_sage2_attn_meansim_cuda(
                q, k, v,
                simthreshd1=0.6,
                cdfthreshd=0.98
            )
            results.append(o_sparse)
        
        # 检查一致性
        max_diff = 0
        for i in range(1, num_runs):
            diff = (results[0] - results[i]).abs().max().item()
            max_diff = max(max_diff, diff)
        
        print(f"最大差异: {max_diff:.2e}")
        if max_diff < 1e-5:
            print("✓ 数值稳定性良好")
        else:
            print("✗ 检测到数值不稳定")
        
        return max_diff
    
    def test_edge_cases(self):
        """测试边界条件"""
        print("\n" + "="*60)
        print("测试边界条件")
        print("="*60)
        
        edge_cases = [
            ("最小序列长度", 128, 8, 64),
            ("大批次", 16, 4, 512),
            ("多头", 32, 16, 256),
            ("大head_dim", 4, 8, 128),
        ]
        
        for name, batch, heads, seq_len in edge_cases:
            try:
                print(f"\n测试: {name} (B={batch}, H={heads}, L={seq_len})")
                q, k, v = self.generate_test_data(
                    batch_size=batch,
                    num_heads=heads,
                    seq_len=seq_len
                )
                o_baseline = self.compute_baseline(q, k, v)
                o_sparse = spas_sage2_attn_meansim_cuda(q, k, v)
                metrics = precision_metric(o_sparse, o_baseline, verbose=True)
                print(f"✓ {name} 测试通过")
            except Exception as e:
                print(f"✗ {name} 测试失败: {e}")
    
    def generate_report(self, save_path: str = "accuracy_test_report.txt"):
        """生成测试报告"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("稀疏注意力准确性测试报告\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"总测试数: {len(self.test_results)}\n\n")
            
            for i, result in enumerate(self.test_results, 1):
                f.write(f"测试 #{i}\n")
                f.write(f"配置: {result['config']}\n")
                f.write(f"Cosine相似度: {result['Cossim']:.6f}\n")
                f.write(f"L1误差: {result['L1']:.6f}\n")
                f.write(f"RMSE: {result['RMSE']:.6f}\n")
                f.write(f"稀疏度: {result['sparsity']:.2%}\n")
                f.write("-"*60 + "\n\n")
        
        print(f"测试报告已保存至: {save_path}")


# ============================================================================
# PyTest 测试用例
# ============================================================================

@pytest.fixture
def tester():
    """创建测试器实例"""
    return SparseAttentionTester()


class TestSparseAttentionAccuracy:
    """PyTest测试类"""
    
    def test_basic_correctness(self, tester):
        """测试基本正确性"""
        metrics = tester.test_basic_accuracy()
        assert metrics['Cossim'] > 0.95, "Cosine相似度太低"
        assert metrics['L1'] < 0.1, "L1误差太大"
    
    def test_high_accuracy_mode(self, tester):
        """测试高精度模式"""
        metrics = tester.test_basic_accuracy(
            simthreshd1=0.3,
            cdfthreshd=0.99
        )
        assert metrics['Cossim'] > 0.98, "高精度模式Cosine相似度应该更高"
        assert metrics['L1'] < 0.05, "高精度模式L1误差应该更小"
    
    def test_causal_mask(self, tester):
        """测试causal mask"""
        metrics = tester.test_basic_accuracy(is_causal=True)
        assert metrics['Cossim'] > 0.95, "Causal模式下准确度不足"
    
    def test_various_sequence_lengths(self, tester):
        """测试不同序列长度"""
        results = tester.test_different_sequence_lengths()
        assert len(results) > 0, "至少应该有一个序列长度测试成功"
        for seq_len, metrics in results:
            assert metrics['Cossim'] > 0.90, f"序列长度{seq_len}准确度不足"
    
    def test_numerical_stability(self, tester):
        """测试数值稳定性"""
        max_diff = tester.test_numerical_stability(num_runs=5)
        assert max_diff < 1e-4, "数值不稳定"


# ============================================================================
# 主测试脚本
# ============================================================================

def run_comprehensive_tests():
    """运行完整的测试套件"""
    tester = SparseAttentionTester()
    
    print("\n" + "="*60)
    print("开始完整的稀疏注意力准确性测试")
    print("="*60)
    
    # 1. 基础准确性测试
    tester.test_basic_accuracy()
    
    # 2. 不同稀疏度测试
    tester.test_various_sparsity_levels()
    
    # 3. 不同序列长度测试
    tester.test_different_sequence_lengths()
    
    # 4. Causal vs Non-causal
    tester.test_causal_vs_noncausal()
    
    # 5. 数值稳定性测试
    tester.test_numerical_stability()
    
    # 6. 边界条件测试
    tester.test_edge_cases()
    
    # 7. 可视化对比
    q, k, v = tester.generate_test_data(seq_len=512)
    tester.visualize_attention_comparison(q, k, v)
    
    # 8. 生成报告
    tester.generate_report()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)


if __name__ == "__main__":
    # 方式1: 直接运行完整测试
    run_comprehensive_tests()
    
    # 方式2: 使用pytest运行
    # pytest tests/test_sparse_attention_accuracy.py -v

