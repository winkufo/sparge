## LLaMA 3 调优指南

### 📋 与CogVideoX的关键差异

| 维度 | CogVideoX | LLaMA 3 |
|------|-----------|---------|
| **模型类型** | Diffusion视频生成 | Causal语言模型 |
| **Attention类型** | 非因果 (is_causal=False) | 因果 (is_causal=True) |
| **架构特点** | 使用Attention Processor | 直接替换Attention层 |
| **数据格式** | 视频生成prompts | 文本数据集 |
| **调优数据** | 5个视频prompt | 5-10个文本样本 |
| **序列长度** | 可变（视频帧数） | 通常2048 tokens |
| **特殊组件** | RoPE在图像部分 | RoPE在全序列 + GQA |

### 🚀 快速开始

#### 1. 调优（首次使用）

```bash
# 基础调优
python evaluate/llama_tune_example.py \
    --tune \
    --model_name meta-llama/Llama-3.2-1B \
    --model_out_path evaluate/models_dict/llama3_1b_tuned.pt \
    --l1 0.06 \
    --pv_l1 0.07

# 并行调优（推荐，快很多）
python evaluate/llama_tune_example.py \
    --tune \
    --parallel_tune \
    --model_name meta-llama/Llama-3.2-1B \
    --model_out_path evaluate/models_dict/llama3_1b_tuned.pt

# 只调优部分层（例如：只调优后半部分层）
python evaluate/llama_tune_example.py \
    --tune \
    --parallel_tune \
    --layer_range 16,32 \
    --model_out_path evaluate/models_dict/llama3_1b_partial.pt
```

**调优时间**：
- 顺序调优：20-30分钟
- 并行调优：5-10分钟（多GPU）

#### 2. 推理（使用调优好的参数）

```bash
# 生成文本测试
python evaluate/llama_tune_example.py \
    --generate \
    --model_out_path evaluate/models_dict/llama3_1b_tuned.pt

# 启用torch.compile进一步加速
python evaluate/llama_tune_example.py \
    --generate \
    --compile \
    --model_out_path evaluate/models_dict/llama3_1b_tuned.pt
```

#### 3. 在你自己的代码中使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate.modify_model.modify_llama import set_spas_sage_attn_llama
from spas_sage_attn.autotune import load_sparse_attention_state_dict
import torch

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 应用稀疏attention
model = set_spas_sage_attn_llama(
    model,
    l1=0.06,
    pv_l1=0.07
)

# 加载调优好的参数
tuned_params = torch.load("llama3_1b_tuned.pt")
load_sparse_attention_state_dict(model, tuned_params)

# 正常使用
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
```

---

## 📊 技术细节

### 1. LLaMA特有的处理

#### Grouped Query Attention (GQA)

LLaMA 3使用GQA来减少KV cache：

```python
# LLaMA-3.2-1B配置示例
num_heads = 32           # Query heads
num_kv_heads = 8         # Key/Value heads（更少！）
num_kv_groups = 4        # 32 / 8 = 4

# 我们的实现自动处理GQA
key_states = self._repeat_kv(key_states, self.num_kv_groups)
value_states = self._repeat_kv(value_states, self.num_kv_groups)
```

#### Rotary Position Embedding (RoPE)

LLaMA使用RoPE而非绝对位置编码：

```python
# 在Q和K上应用旋转位置编码
cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
query_states, key_states = apply_rotary_pos_emb(
    query_states, key_states, cos, sin, position_ids
)
```

#### Causal Mask

LLaMA是自回归模型，必须使用causal mask：

```python
# 强制使用causal attention
attn_output = sparse_attn(
    query, key, value,
    is_causal=True  # ← 关键！
)
```

### 2. 与CogVideoX实现的差异

#### CogVideoX的实现

```python
# cogvideo: 使用Attention Processor模式
class SageAttnCogVideoXAttnProcessor:
    def __call__(self, attn, hidden_states, ...):
        # 在processor中处理attention
        query = attn.to_q(hidden_states)
        # ...
        hidden_states = attn.inner_attention(q, k, v, is_causal=False)

# 设置processor
block.attn1.set_processor(SageAttnCogVideoXAttnProcessor())
```

#### LLaMA的实现

```python
# llama: 直接替换整个Attention模块
class SparseLlamaAttention(nn.Module):
    def __init__(self, original_attn):
        # 复制所有projection层
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        # ...
    
    def forward(self, hidden_states, ...):
        # 完整的attention逻辑
        # 包括RoPE、GQA、KV cache等

# 直接替换
layer.self_attn = SparseLlamaAttention(original_attn)
```

**原因**：
- CogVideoX的Attention已经有processor机制，可以插入自定义逻辑
- LLaMA的Attention是完整封装的，需要整体替换

---

## 🎯 调优策略

### 策略1：全层调优（默认）

```bash
python evaluate/llama_tune_example.py --tune
# 调优所有32层（LLaMA-3.2-1B）
```

**适用场景**：
- 追求最佳性能
- 有足够的调优时间
- 全面部署

**预期效果**：
- 平均稀疏度：40-50%
- 精度损失：<2%

### 策略2：分层调优

不同层对稀疏化的敏感度不同：

```bash
# 只调优后半部分（16-31层）
python evaluate/llama_tune_example.py --tune --layer_range 16,32

# 只调优中间层（8-24层）
python evaluate/llama_tune_example.py --tune --layer_range 8,24
```

**经验规则**：
- 前几层（0-7）：提取基础特征，建议保持密集或低稀疏度
- 中间层（8-23）：可以中等稀疏（40-50%）
- 后期层（24+）：可以高度稀疏（50-60%）

### 策略3：调整精度约束

```bash
# 高精度（低稀疏）
python evaluate/llama_tune_example.py --tune --l1 0.04 --pv_l1 0.05

# 平衡（推荐）
python evaluate/llama_tune_example.py --tune --l1 0.06 --pv_l1 0.07

# 高稀疏（可能影响精度）
python evaluate/llama_tune_example.py --tune --l1 0.10 --pv_l1 0.12
```

---

## 📈 评估调优效果

### 方法1：使用我们提供的测试脚本

```bash
# 测试调优前后的输出差异
python tests/test_llama_output_accuracy.py \
    --model meta-llama/Llama-3.2-1B \
    --output before_tune.json

# 调优
python evaluate/llama_tune_example.py --tune --model_out_path tuned.pt

# 测试调优后
python tests/test_llama_output_accuracy.py \
    --model meta-llama/Llama-3.2-1B \
    --model_out_path tuned.pt \
    --output after_tune.json

# 对比结果
python compare_results.py before_tune.json after_tune.json
```

### 方法2：查看稀疏度统计

```python
import json

# 读取统计文件
with open('evaluate/models_dict/llama3_stats.json') as f:
    stats = json.load(f)

# 分析
for layer_stats in stats:
    print(f"Layer {layer_stats['layer_idx']}: "
          f"稀疏度 {layer_stats['mean_sparsity']:.2%}")
```

### 方法3：实际任务测试

```python
# 在你的下游任务上测试
from your_task import evaluate_model

# 密集模型
score_dense = evaluate_model(dense_model)

# 稀疏模型
score_sparse = evaluate_model(sparse_model)

# 对比
print(f"精度差异: {(score_sparse - score_dense):.2%}")
```

---

## 🔧 常见问题

### Q1: 调优时显存不足

**A**: 减少样本长度或使用gradient checkpointing

```bash
# 减少序列长度
python evaluate/llama_tune_example.py --tune --max_length 1024

# 或使用小样本数
python evaluate/llama_tune_example.py --tune --num_tune_samples 3
```

### Q2: 调优很慢

**A**: 使用并行调优

```bash
# 确保启用并行模式
python evaluate/llama_tune_example.py --tune --parallel_tune

# 这会利用所有可用GPU并行处理不同的head
```

### Q3: 不同大小的LLaMA模型需要重新调优吗？

**A**: 是的！

```bash
# LLaMA-3.2-1B
python evaluate/llama_tune_example.py --tune \
    --model_name meta-llama/Llama-3.2-1B

# LLaMA-3.2-3B（需要重新调优）
python evaluate/llama_tune_example.py --tune \
    --model_name meta-llama/Llama-3.2-3B \
    --model_out_path llama3_3b_tuned.pt
```

### Q4: 调优后的参数能跨任务使用吗？

**A**: 可以，但建议在目标任务数据上重新调优

```bash
# 通用调优（WikiText）
python evaluate/llama_tune_example.py --tune --dataset wikitext

# 特定任务调优（代码）
python evaluate/llama_tune_example.py --tune --dataset codeparrot/github-code
```

### Q5: 如何知道调优是否成功？

**A**: 看这几个指标：

1. **稀疏度** > 30%（至少有效果）
2. **Logits Cosine相似度** > 0.95（精度保持）
3. **实际生成质量**（最终检验）

```bash
# 调优后立即测试生成
python evaluate/llama_tune_example.py --generate --model_out_path tuned.pt
```

---

## 📝 最佳实践总结

### ✅ 推荐做法

1. **首次使用**：用默认参数调优
   ```bash
   python evaluate/llama_tune_example.py --tune --parallel_tune
   ```

2. **生产部署**：在代表性数据上调优
   - 使用与实际应用相似的文本
   - 5-10个样本即可

3. **定期评估**：模型更新后重新调优
   - Fine-tuning后重新调优
   - 换数据域重新调优

4. **保存多个版本**：不同精度-速度配置
   ```bash
   # 高精度版
   python evaluate/llama_tune_example.py --tune --l1 0.04 \
       --model_out_path llama3_high_accuracy.pt
   
   # 高速度版
   python evaluate/llama_tune_example.py --tune --l1 0.08 \
       --model_out_path llama3_high_speed.pt
   ```

### ❌ 避免的做法

1. ❌ 每次推理都调优（太慢）
2. ❌ 用太短的文本调优（<256 tokens）
3. ❌ 跨模型共用调优参数
4. ❌ 不验证就直接部署

---

## 🎓 进阶：自定义调优

如果默认脚本不满足需求，可以自定义：

```python
from transformers import AutoModelForCausalLM
from evaluate.modify_model.modify_llama import set_spas_sage_attn_llama, enable_tune_mode
from spas_sage_attn.autotune import extract_sparse_attention_state_dict

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("your_model")

# 2. 自定义层范围和参数
model = set_spas_sage_attn_llama(
    model,
    l1=0.05,           # 自定义精度
    pv_l1=0.06,
    layer_range=(10, 30)  # 只替换10-29层
)

# 3. 启用调优
enable_tune_mode(model, True)

# 4. 用你自己的数据调优
for text in your_custom_data:
    inputs = tokenizer(text, return_tensors="pt")
    model(**inputs)  # 触发调优

# 5. 保存
params = extract_sparse_attention_state_dict(model)
torch.save(params, "custom_tuned.pt")
```

---

## 📚 相关文档

- [测试指南](../tests/README.md)
- [调优原理](../docs/tuning_guide.md)
- [CogVideoX调优](cogvideo_example.py)

