# 模型适配说明

本目录包含了为不同模型适配SpargeAttn的代码。

## 📁 文件说明

| 文件 | 模型类型 | 注意力类型 | 特殊处理 |
|------|---------|-----------|---------|
| `modify_cogvideo.py` | CogVideoX (视频生成) | 非因果 | Attention Processor |
| `modify_llama.py` | LLaMA (语言模型) | 因果 | GQA + RoPE |
| `modify_flux.py` | Flux (图像生成) | 非因果 | - |
| `modify_hunyuan.py` | HunyuanDiT (图像生成) | 非因果 | - |
| `modify_wan.py` | Wan2V (视频生成) | 非因果 | - |

---

## 🔧 LLaMA适配说明

### 关键修正

根据[transformers官方实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)，修正了以下问题：

#### 1. `hidden_size` 属性

**问题**：LLamaAttention没有直接的`hidden_size`属性

```python
# ❌ 错误
self.hidden_size = original_attn.hidden_size  # 不存在

# ✅ 正确
self.hidden_size = original_attn.config.hidden_size  # 从config获取
```

#### 2. `apply_rotary_pos_emb` 导入

**问题**：需要从transformers导入

```python
# ✅ 正确导入
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
```

#### 3. 版本兼容性

不同版本的transformers API可能不同：

```python
# 兼容处理
try:
    # 新版本API
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin)
except:
    # 旧版本API  
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
```

---

## ✅ 验证实现

运行验证脚本确保实现正确：

```bash
python evaluate/verify_llama_implementation.py
```

**预期输出**：
```
验证LLaMA稀疏化实现
============================================================

[1/5] 加载原始模型...
✓ 模型加载成功

[2/5] 检查模型结构...
✓ 找到 32 层
Attention类型: LlamaAttention
Attention属性:
  ✓ config: LlamaConfig
  ✓ num_heads: 32
  ✓ head_dim: 64
  ✓ num_key_value_heads: 8
  ✓ num_key_value_groups: 4
  ✓ hidden_size (from config): 2048

[3/5] 应用稀疏化...
✓ Replaced layer 0 attention with sparse version
✓ Replaced layer 1 attention with sparse version
...
总共替换了 32/32 个attention层
✓ 稀疏化应用成功

[4/5] 测试前向传播...
✓ 前向传播成功
  输出形状: torch.Size([1, 7, 32000])
  输出范围: [-15.23, 12.45]

[5/5] 测试生成...
✓ 生成成功
  输入: The capital of France is
  输出: The capital of France is Paris.

============================================================
✓ 所有验证通过！LLaMA实现正确
============================================================
```

---

## 🎯 不同模型的适配要点

### CogVideoX

```python
# 特点：
- 使用Attention Processor机制
- 非因果attention (is_causal=False)
- 有text和image两部分（RoPE只在image部分）

# 实现方式：
class SageAttnCogVideoXAttnProcessor:
    def __call__(self, attn, hidden_states, ...):
        # 在processor中处理attention
        ...

block.attn1.set_processor(processor)
```

### LLaMA

```python
# 特点：
- 直接的Attention模块
- 因果attention (is_causal=True)
- GQA (Grouped Query Attention)
- RoPE应用在全序列

# 实现方式：
class SparseLlamaAttention(nn.Module):
    def __init__(self, original_attn):
        # 复制所有projection和rotary_emb
        ...
    
    def forward(self, hidden_states, ...):
        # 完整实现attention逻辑
        ...

layer.self_attn = SparseLlamaAttention(original_attn)
```

### 关键差异总结

| 方面 | CogVideoX | LLaMA |
|------|-----------|-------|
| **替换方式** | 设置processor | 替换整个模块 |
| **is_causal** | False | True |
| **特殊组件** | RoPE部分应用 | GQA + RoPE全应用 |
| **难度** | 简单 | 中等 |

---

## 🔍 调试技巧

### 如果遇到属性错误

```python
# 检查对象有哪些属性
obj = model.model.layers[0].self_attn
print(dir(obj))

# 检查config
print(obj.config.__dict__)
```

### 如果遇到维度错误

```python
# 打印中间tensor的形状
print(f"Q shape: {query_states.shape}")
print(f"K shape: {key_states.shape}")
print(f"V shape: {value_states.shape}")
```

### 如果遇到RoPE错误

```python
# 检查transformers版本
import transformers
print(f"transformers version: {transformers.__version__}")

# 推荐版本: >= 4.36.0
```

---

## 📚 参考资源

- [Transformers LLaMA实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- [CogVideoX实现](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/cogvideox_transformer_3d.py)
- [SpargeAttn论文](https://arxiv.org/abs/2502.18137)

