# CogVideoX准确性测试指南

## 🎯 测试目标

对比CogVideoX使用SpargeAttn前后的输出差异：
- ✅ 密集版（原始attention）
- ✅ 稀疏版（SpargeAttn）

**核心问题**：稀疏化会影响视频生成质量吗？

---

## 🚀 快速开始

### 1. 基础测试（不调优）

```bash
# 测试默认参数的效果
python evaluate/cogvideo_accuracy_test.py \
    --num_prompts 3 \
    --save_videos
```

**预期输出**：
```
测试 1/3
Prompt: A panda eating bamboo in a lush forest.
============================================================
  生成密集版视频...
  生成稀疏版视频...
  对比生成的视频...

  结果:
    帧相似度(Cosine): 0.987654
    帧L1误差: 0.043210
    PSNR: 38.45 dB
    平均帧差异: 0.023456

统计摘要
============================================================
帧相似度(Cosine)         0.985432    0.005123    ...
帧L1误差                0.045678    0.008234    ...
PSNR (dB)               37.82       1.234       ...

质量评估
============================================================
总分: 85/100
✓ 良好，视频质量保持得很好
```

### 2. 测试调优后的模型

```bash
# 使用调优后的参数
python evaluate/cogvideo_accuracy_test.py \
    --use_tuned \
    --tuned_path evaluate/models_dict/CogVideoX-2b_0.06_0.07.pt \
    --num_prompts 5 \
    --save_videos
```

### 3. 完整测试（所有prompts）

```bash
# 使用完整的prompt列表
python evaluate/cogvideo_accuracy_test.py \
    --prompt_file evaluate/datasets/video/prompts.txt \
    --num_prompts 10 \
    --save_videos \
    --output_dir evaluate/results/full_test
```

---

## 📊 测试指标说明

### 1. 帧相似度 (Frame Cosine Similarity)

```python
# 对所有视频帧计算Cosine相似度
similarity = cosine_similarity(frames_sparse, frames_dense)
```

**含义**：两个视频在像素空间的整体相似度

**阈值**：
- `> 0.98`: 几乎完全相同
- `> 0.95`: 非常相似（人眼难以区分）
- `> 0.90`: 相似（可能有细微差别）
- `< 0.90`: 有明显差异

### 2. PSNR (峰值信噪比)

```python
PSNR = 10 * log10(MAX^2 / MSE)
```

**含义**：视频质量的客观指标，越高越好

**阈值**：
- `> 40 dB`: 优秀（几乎无损）
- `> 35 dB`: 良好（高质量）
- `> 30 dB`: 可接受（有损但可用）
- `< 30 dB`: 质量下降明显

### 3. 帧L1误差

```python
L1 = mean(|frames_sparse - frames_dense|)
```

**含义**：像素级的平均绝对误差

**阈值**：
- `< 0.02`: 优秀
- `< 0.05`: 良好
- `< 0.10`: 可接受
- `> 0.10`: 需改进

---

## 📁 输出文件

测试完成后会生成：

```
evaluate/results/cogvideo_accuracy/
├── cogvideo_accuracy_default.json  # 不调优的结果
├── cogvideo_accuracy_tuned.json    # 调优后的结果
└── videos/                          # 生成的视频（如果启用--save_videos）
    ├── prompt_0_dense.mp4          # 密集版
    ├── prompt_0_sparse.mp4         # 稀疏版
    ├── prompt_1_dense.mp4
    ├── prompt_1_sparse.mp4
    └── ...
```

### 结果JSON格式

```json
{
  "config": {
    "model_name": "THUDM/CogVideoX-2b",
    "use_tuned": true,
    "num_prompts": 5,
    "seed": 42
  },
  "statistics": {
    "frame_cosine": {
      "mean": 0.9876,
      "std": 0.0034,
      "min": 0.9821,
      "max": 0.9912
    },
    "psnr": {
      "mean": 38.45,
      ...
    }
  },
  "detailed_results": [
    {
      "prompt": "A panda eating bamboo...",
      "frame_cosine": 0.9854,
      "psnr": 37.82,
      ...
    }
  ],
  "overall_score": 85
}
```

---

## 🔍 对比调优前后

### 测试流程

```bash
# 1. 测试不调优版本
python evaluate/cogvideo_accuracy_test.py \
    --num_prompts 5 \
    --save_videos \
    --output_dir evaluate/results/before_tune

# 2. 运行调优
python evaluate/cogvideo_example.py \
    --use_spas_sage_attn \
    --tune \
    --parallel_tune \
    --model_out_path evaluate/models_dict/my_tuned.pt

# 3. 测试调优后版本
python evaluate/cogvideo_accuracy_test.py \
    --use_tuned \
    --tuned_path evaluate/models_dict/my_tuned.pt \
    --num_prompts 5 \
    --save_videos \
    --output_dir evaluate/results/after_tune

# 4. 对比结果
python tools/compare_results.py \
    evaluate/results/before_tune/cogvideo_accuracy_default.json \
    evaluate/results/after_tune/cogvideo_accuracy_tuned.json
```

---

## 🎬 人工评估视频质量

虽然客观指标很重要，但最终还是要看人眼效果：

### 并排对比

```bash
# 安装ffmpeg
brew install ffmpeg  # macOS
# 或 apt-get install ffmpeg  # Linux

# 生成对比视频（左右并排）
ffmpeg -i evaluate/results/cogvideo_accuracy/videos/prompt_0_dense.mp4 \
       -i evaluate/results/cogvideo_accuracy/videos/prompt_0_sparse.mp4 \
       -filter_complex hstack \
       compare_0.mp4
```

### 评估清单

观看对比视频，检查：
- [ ] 整体内容是否一致？
- [ ] 物体运动是否流畅？
- [ ] 细节是否保留？
- [ ] 是否有伪影（artifacts）？
- [ ] 颜色是否一致？

---

## 📈 预期结果

### 不调优（默认参数）

```
配置: simthreshd1=0.6, cdfthreshd=0.98

预期:
  帧相似度: 0.95-0.97
  PSNR: 35-38 dB
  视觉质量: 良好，轻微差异
  稀疏度: 30-40%
```

### 调优后

```
配置: 自动搜索的最优参数

预期:
  帧相似度: 0.97-0.99
  PSNR: 38-42 dB
  视觉质量: 优秀，几乎无差异
  稀疏度: 40-50%
```

---

## 🔧 常见问题

### Q1: 测试很慢怎么办？

**A**: 减少测试样本或推理步数

```bash
# 快速测试（3个prompt，30步）
python evaluate/cogvideo_accuracy_test.py \
    --num_prompts 3 \
    --num_inference_steps 30
```

### Q2: 显存不足

**A**: 生成完一个就清理显存

```python
# 脚本已经自动处理了
gc.collect()
torch.cuda.empty_cache()
```

如果还不够：
```bash
# 减少帧数
python evaluate/cogvideo_accuracy_test.py --num_frames 25
```

### Q3: 如何解读PSNR？

**A**: PSNR是视频/图像质量的标准指标

```
PSNR > 40 dB: 几乎看不出差异
PSNR 35-40 dB: 高质量，细看可能有差异
PSNR 30-35 dB: 可接受，有可见差异
PSNR < 30 dB: 质量明显下降
```

### Q4: 帧相似度和PSNR不一致怎么办？

**A**: 这是正常的，它们关注不同方面

```
情况1: Cosine高，PSNR低
→ 整体结构相似，但细节有差异
→ 建议：人眼检查细节

情况2: Cosine低，PSNR高
→ 数值接近，但结构不同（罕见）
→ 建议：检查是否有bug
```

---

## 💡 最佳实践

### 1. 测试流程

```bash
# Step 1: 快速测试（3个prompt）
python evaluate/cogvideo_accuracy_test.py --num_prompts 3

# Step 2: 如果效果好，完整测试
python evaluate/cogvideo_accuracy_test.py --num_prompts 10 --save_videos

# Step 3: 人工观看对比视频

# Step 4: 如果不满意，调优后重测
python evaluate/cogvideo_example.py --tune
python evaluate/cogvideo_accuracy_test.py --use_tuned --save_videos
```

### 2. 选择测试prompts

**建议prompts类型**：
- ✅ 包含运动（测试时序一致性）
- ✅ 包含细节（测试细节保留）
- ✅ 不同场景（室内、室外、特写等）
- ✅ 不同主体（人、动物、物体等）

### 3. 判断标准

```
可以部署的条件：
  帧相似度 > 0.95 AND
  PSNR > 35 dB AND
  人眼检查OK
```

---

## 📝 报告模板

测试完成后，可以这样总结：

```markdown
# CogVideoX稀疏化测试报告

## 配置
- 模型: THUDM/CogVideoX-2b
- 调优: 是/否
- 测试prompts: 10个

## 客观指标
- 帧相似度: 0.9876 (±0.0034)
- PSNR: 38.45 dB (±1.23)
- 帧L1误差: 0.0432 (±0.0082)

## 稀疏度
- 平均稀疏度: 45.2%
- 理论加速: 1.8x

## 主观评估
- 5个视频人工对比
- 结论: 4个完全无差异，1个有轻微差异

## 结论
✓ 可以部署
```

---

## 🎓 进阶：自定义评估

如果你需要特定的评估指标：

```python
import torch
from evaluate.cogvideo_accuracy_test import create_pipelines, load_prompts

# 加载模型
pipe_dense, pipe_sparse = create_pipelines(args)

# 自定义测试
my_prompts = ["Your specific prompt"]

for prompt in my_prompts:
    # 生成
    frames_dense = pipe_dense(prompt, ...).frames[0]
    frames_sparse = pipe_sparse(prompt, ...).frames[0]
    
    # 自定义评估
    your_metric = compute_your_metric(frames_dense, frames_sparse)
```

---

## 📚 相关资源

- [CogVideoX调优示例](cogvideo_example.py)
- [模型修改代码](modify_model/modify_cogvideo.py)
- [SpargeAttn论文](https://arxiv.org/abs/2502.18137)

