# 稀疏注意力准确性测试指南

本目录包含了完整的测试套件，用于验证SpargeAttn稀疏注意力的输出准确性。

## 📋 测试文件概览

| 文件 | 用途 | 适用场景 |
|------|------|----------|
| `test_sparse_attention_accuracy.py` | 完整的准确性测试套件 | 全面测试，CI/CD |
| `quick_test.py` | 快速验证脚本 | 日常开发，快速验证 |
| `test_end_to_end_model.py` | 端到端模型测试 | 集成测试，模型级验证 |
| `test_llama_output_accuracy.py` | **LLaMA真实模型测试** | **最重要：测试真实输出差异** |
| `quick_llama_test.py` | LLaMA快速测试 | 快速验证真实数据稀疏度 |

**⚠️ 重要说明：**
- 前3个文件使用**随机数据**，无法反映真实稀疏化效果
- **推荐使用** `test_llama_output_accuracy.py` 在真实模型上测试
- 真实模型的attention pattern有结构，稀疏度会**明显更高**

## 🚀 快速开始

### 🌟 推荐：测试真实LLaMA模型

**这是最准确的测试方式**，因为它使用真实模型的attention pattern：

```bash
# 完整测试（需要5-10分钟）
python tests/test_llama_output_accuracy.py

# 快速测试（2分钟）
python tests/quick_llama_test.py

# 对比随机数据vs真实数据
python tests/quick_llama_test.py --compare
```

**输出示例**：
```
测试输出差异
============================================================
汇总统计 (所有样本平均)
============================================================
logits_cosine       : 0.987654 (±0.003210)
logits_l1           : 0.043210 (±0.012345)
token_accuracy      : 0.945000 (±0.023456)
top5_overlap        : 0.980000 (±0.015432)

最终评估
============================================================
关键指标:
  Logits相似度: 0.9877
  Token准确率: 94.50%
  平均稀疏度: 42.35%

评估:
  ✓ 良好！可以用于生产
  ✓ 稀疏度很好 (42.4%)
```

---

### 1. 快速验证随机数据（仅用于功能测试）

⚠️ **注意**：随机数据的稀疏度会很低，不代表真实效果

```bash
cd /path/to/SpargeAttn
python tests/quick_test.py
```

**高级用法：**

```bash
# 测试长序列
python tests/quick_test.py --seq_len 4096

# 测试高稀疏度
python tests/quick_test.py --sparse_level high

# 测试causal attention
python tests/quick_test.py --causal

# 运行速度benchmark
python tests/quick_test.py --benchmark
```

**输出示例：**

```
============================================================
快速测试: 中稀疏度-中精度
序列长度: 1024, Causal: False
============================================================

输入形状: Q=torch.Size([2, 8, 1024, 64]), ...

计算标准attention (baseline)...
计算稀疏attention...

精度指标:
------------------------------------------------------------
Cossim: 0.9876, L1: 0.0543, RMSE:0.0892
稀疏度: 45.23%

测试结果:
------------------------------------------------------------
✓ Cosine相似度 0.9876 > 0.95
✓ L1误差 0.0543 < 0.1
✓ 稀疏度 45.23% > 5%

============================================================
✓ 测试通过!
============================================================
```

### 2. 完整测试套件

运行所有测试，生成详细报告：

```bash
python tests/test_sparse_attention_accuracy.py
```

这将：
- ✅ 测试基础准确性
- ✅ 测试不同稀疏度级别
- ✅ 测试不同序列长度
- ✅ 测试causal vs non-causal
- ✅ 测试数值稳定性
- ✅ 测试边界条件
- ✅ 生成可视化对比图
- ✅ 生成测试报告

**输出文件：**
- `attention_comparison.png` - attention map对比图
- `accuracy_test_report.txt` - 详细测试报告

### 3. 使用PyTest运行

```bash
# 安装pytest
pip install pytest pytest-cov

# 运行所有测试
pytest tests/test_sparse_attention_accuracy.py -v

# 运行特定测试
pytest tests/test_sparse_attention_accuracy.py::TestSparseAttentionAccuracy::test_basic_correctness -v

# 生成覆盖率报告
pytest tests/ --cov=spas_sage_attn --cov-report=html
```

### 4. 端到端模型测试

测试在完整模型中的表现：

```bash
python tests/test_end_to_end_model.py
```

这将测试：
- 单层Transformer block
- 多层Transformer模型
- 不同序列长度下的稳定性
- 显存使用对比
- Attention pattern保留情况

## 🎯 测试方法论

### 为什么测试输出差异而不是困惑度？

**问题**：你可能想问，为什么不测试困惑度(Perplexity)？

**答案**：困惑度是评估**单个模型好坏**的指标，但我们要测试的是**稀疏化前后的差异**。

```python
# ❌ 错误的测试思路
ppl_dense = compute_perplexity(dense_model)    # 10.5
ppl_sparse = compute_perplexity(sparse_model)  # 10.8
# → 困惑度变化了，但这不能直接说明稀疏化的影响

# ✅ 正确的测试思路  
# 相同输入 → 对比输出
same_input = "The capital of France is"
logits_dense = dense_model(same_input)
logits_sparse = sparse_model(same_input)
difference = cosine_similarity(logits_dense, logits_sparse)
# → 0.987，说明输出几乎一致！
```

### 核心测试指标

**我们测试的是：密集模型 vs 稀疏模型在相同输入下的输出差异**

| 指标 | 含义 | 好的标准 |
|------|------|---------|
| **Logits Cosine相似度** | 输出向量的方向一致性 | > 0.95 |
| **Logits L1误差** | 输出数值的相对误差 | < 0.10 |
| **Token准确率** | 预测的token完全相同的比例 | > 90% |
| **Top-k重叠率** | 前k个候选token的重叠 | > 95% |
| **生成一致性** | 生成的文本是否相同 | 定性评估 |

### 真实数据 vs 随机数据

**关键发现**：你的观察完全正确！

```python
# 随机数据测试
q_random = torch.randn(1, 8, 1024, 64)
k_random = torch.randn(1, 8, 1024, 64)
v_random = torch.randn(1, 8, 1024, 64)
sparsity_random = test_sparse_attn(q_random, k_random, v_random)
# → 稀疏度: 15% (块内相似度很低，几乎不能稀疏)

# 真实LLaMA数据测试
q_real, k_real, v_real = extract_from_llama(real_text)
sparsity_real = test_sparse_attn(q_real, k_real, v_real)
# → 稀疏度: 45% (attention pattern有结构，可以大量稀疏！)
```

**结论**：
- ✅ 必须在真实模型上测试
- ❌ 随机数据只能测试"功能是否正常"
- ✅ 真实数据才能测试"稀疏化效果如何"

---

## 📊 理解测试指标

### 1. **Cosine相似度 (Cosine Similarity)**

```python
sim = F.cosine_similarity(output_sparse.flatten(), output_baseline.flatten())
```

- **范围**: -1 到 1
- **含义**: 两个输出向量的方向相似度
- **阈值建议**:
  - `> 0.99`: 极高精度（几乎完全一致）
  - `> 0.95`: 高精度（推荐）
  - `> 0.90`: 中等精度
  - `< 0.90`: 低精度（需要调整参数）

### 2. **L1相对误差**

```python
l1 = (output_sparse - output_baseline).abs().sum() / output_baseline.abs().sum()
```

- **范围**: 0 到 ∞
- **含义**: 相对平均绝对误差
- **阈值建议**:
  - `< 0.05`: 极高精度
  - `< 0.10`: 高精度（推荐）
  - `< 0.15`: 中等精度
  - `> 0.15`: 低精度

### 3. **RMSE (均方根误差)**

```python
rmse = torch.sqrt(torch.mean((output_sparse - output_baseline) ** 2))
```

- **范围**: 0 到 ∞
- **含义**: 误差的标准差
- **特点**: 对大误差更敏感

### 4. **稀疏度 (Sparsity)**

```python
sparsity = 1 - (有效块数 / 总块数)
```

- **范围**: 0 到 1
- **含义**: 跳过的计算比例
- **期望**: 
  - 更高的稀疏度 = 更快的速度
  - 需要与精度权衡

## 🎯 测试策略建议

### 场景1: 开发新功能

```bash
# 快速验证功能是否工作
python tests/quick_test.py

# 通过后再运行完整测试
python tests/test_sparse_attention_accuracy.py
```

### 场景2: 调整超参数

创建自定义测试：

```python
from tests.test_sparse_attention_accuracy import SparseAttentionTester

tester = SparseAttentionTester()

# 测试你的参数组合
for simth in [0.3, 0.5, 0.7]:
    for cdfth in [0.95, 0.97, 0.99]:
        metrics = tester.test_basic_accuracy(
            simthreshd1=simth,
            cdfthreshd=cdfth
        )
        print(f"simth={simth}, cdfth={cdfth}: "
              f"Cosine={metrics['Cossim']:.4f}, "
              f"Sparsity={metrics['sparsity']:.2%}")
```

### 场景3: 集成到新模型

```python
# 1. 先用快速测试验证基本功能
python tests/quick_test.py --seq_len 2048

# 2. 创建模型级测试（参考test_end_to_end_model.py）
# 3. 对比生成结果（如图像、视频质量）
```

### 场景4: CI/CD集成

```yaml
# .github/workflows/test.yml 示例
name: Accuracy Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        python setup.py install
    
    - name: Run quick test
      run: python tests/quick_test.py
    
    - name: Run pytest
      run: pytest tests/test_sparse_attention_accuracy.py -v
```

## 🔍 调试失败的测试

### 如果Cosine相似度低

```python
# 1. 检查输入数据类型
print(q.dtype, k.dtype, v.dtype)  # 应该是float16或bfloat16

# 2. 降低稀疏度
metrics = tester.test_basic_accuracy(
    simthreshd1=0.3,   # 降低阈值
    cdfthreshd=0.99     # 提高保留比例
)

# 3. 可视化差异
tester.visualize_attention_comparison(q, k, v)
```

### 如果L1误差大

```python
# 检查是否有NaN或Inf
print(f"Output has NaN: {torch.isnan(output).any()}")
print(f"Output has Inf: {torch.isinf(output).any()}")

# 检查数值范围
print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
```

### 如果稀疏度太低

```python
# 1. 检查序列长度（太短可能稀疏不起来）
# 最小推荐: 512

# 2. 调整参数
simthreshd1 = 0.7   # 更高的阈值
cdfthreshd = 0.95   # 更低的CDF阈值
```

## 📈 可视化工具

### 生成Attention Map对比

```python
from tests.test_sparse_attention_accuracy import SparseAttentionTester

tester = SparseAttentionTester()
q, k, v = tester.generate_test_data(seq_len=512)
tester.visualize_attention_comparison(q, k, v, head_idx=0, 
                                     save_path="my_comparison.png")
```

### 分析稀疏模式

```python
from spas_sage_attn.utils import get_block_map_meansim

# 获取块稀疏mask
block_map = get_block_map_meansim(
    q, k,
    simthreshd1=0.6,
    cdfthreshd=0.98,
    return_lut=False
)

# 可视化稀疏模式
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.imshow(block_map[0, 0].cpu().numpy(), cmap='binary')
plt.title('Block Sparse Pattern (Head 0)')
plt.xlabel('Key Blocks')
plt.ylabel('Query Blocks')
plt.colorbar()
plt.savefig('sparse_pattern.png')
```

## 🎓 最佳实践

1. **始终先运行快速测试**
   ```bash
   python tests/quick_test.py
   ```

2. **调参时使用网格搜索**
   ```python
   for s in [0.3, 0.5, 0.7]:
       for c in [0.95, 0.97, 0.99]:
           test_basic_accuracy(simthreshd1=s, cdfthreshd=c)
   ```

3. **关注精度-稀疏度权衡**
   - 目标: Cosine > 0.95 且 Sparsity > 30%

4. **在真实数据上验证**
   - 合成数据测试通过后，用实际模型数据测试

5. **记录测试结果**
   ```python
   tester.generate_report(save_path="my_test_report.txt")
   ```

## 🤝 贡献测试用例

欢迎添加新的测试用例！请确保：

1. ✅ 测试覆盖特定场景
2. ✅ 包含清晰的注释
3. ✅ 提供预期结果
4. ✅ 可以独立运行

示例PR结构：
```
tests/
  test_long_sequence.py      # 新测试文件
  README.md                  # 更新文档
```

## 📝 常见问题

**Q: 测试运行很慢怎么办？**

A: 使用快速测试或减小测试规模：
```bash
python tests/quick_test.py --seq_len 512
```

**Q: 如何测试特定head_dim？**

A: 修改测试数据生成器：
```python
q, k, v = tester.generate_test_data(head_dim=128)
```

**Q: 能测试FP32输入吗？**

A: 稀疏kernel会自动转换为FP16/BF16，建议直接用FP16测试。

**Q: 如何测试自己的数据？**

A:
```python
# 加载你的数据
my_q = torch.load('my_q.pt')
my_k = torch.load('my_k.pt')
my_v = torch.load('my_v.pt')

# 运行测试
o_baseline = tester.compute_baseline(my_q, my_k, my_v)
o_sparse = spas_sage2_attn_meansim_cuda(my_q, my_k, my_v)
metrics = precision_metric(o_sparse, o_baseline)
```

## 📧 联系与支持

如有问题，请：
1. 查看项目主README
2. 提交GitHub Issue
3. 参考论文: [SpargeAttn Paper](https://arxiv.org/abs/2502.18137)

