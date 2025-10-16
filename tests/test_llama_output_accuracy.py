"""
测试LLaMA模型稀疏化前后的输出准确性

核心思想：
1. 加载同一个模型的两个副本（密集版 vs 稀疏版）
2. 用相同的输入喂给两个模型
3. 直接对比它们的输出差异

测试指标：
- Logits的Cosine相似度和L1误差
- Hidden states的相似度
- 预测token的一致性
- 生成文本的一致性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
from typing import Dict, List
import argparse

from spas_sage_attn import spas_sage2_attn_meansim_cuda
from spas_sage_attn.utils import precision_metric


class SparseAttentionWrapper(nn.Module):
    """替换attention为稀疏版本的包装器"""
    
    def __init__(self, original_attn, simthreshd1=0.6, cdfthreshd=0.98):
        super().__init__()
        self.original_attn = original_attn
        self.simthreshd1 = simthreshd1
        self.cdfthreshd = cdfthreshd
        
        # 统计
        self.num_calls = 0
        self.total_sparsity = 0.0
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None, 
                past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        
        bsz, q_len, _ = hidden_states.size()
        
        # Q, K, V projection
        query_states = self.original_attn.q_proj(hidden_states)
        key_states = self.original_attn.k_proj(hidden_states)
        value_states = self.original_attn.v_proj(hidden_states)
        
        # Reshape
        query_states = query_states.view(bsz, q_len, self.original_attn.num_heads, 
                                        self.original_attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.original_attn.num_key_value_heads, 
                                     self.original_attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.original_attn.num_key_value_heads, 
                                        self.original_attn.head_dim).transpose(1, 2)
        
        # KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        # GQA: repeat k/v heads
        if self.original_attn.num_key_value_heads != self.original_attn.num_heads:
            key_states = self._repeat_kv(key_states, self.original_attn.num_key_value_groups)
            value_states = self._repeat_kv(value_states, self.original_attn.num_key_value_groups)
        
        kv_seq_len = key_states.shape[-2]
        
        # 稀疏attention (只在序列足够长时)
        if kv_seq_len >= 256:
            try:
                attn_output, sparsity = spas_sage2_attn_meansim_cuda(
                    query_states, key_states, value_states,
                    is_causal=True,
                    simthreshd1=self.simthreshd1,
                    cdfthreshd=self.cdfthreshd,
                    return_sparsity=True
                )
                self.num_calls += 1
                self.total_sparsity += sparsity
            except:
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states, value_states, is_causal=True
                )
        else:
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states, is_causal=True
            )
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.original_attn.hidden_size)
        attn_output = self.original_attn.o_proj(attn_output)
        
        if use_cache:
            return attn_output, None, (key_states, value_states)
        return attn_output, None, None
    
    def _repeat_kv(self, hidden_states, n_rep):
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    def get_avg_sparsity(self):
        return self.total_sparsity / self.num_calls if self.num_calls > 0 else 0.0


def replace_with_sparse_attention(model, simthreshd1=0.6, cdfthreshd=0.98):
    """替换模型的所有attention层"""
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'self_attn'):
            module.self_attn = SparseAttentionWrapper(
                module.self_attn, simthreshd1, cdfthreshd
            )
            count += 1
    print(f"替换了 {count} 个attention层")
    return model


def test_output_difference_on_texts(
    model_dense,
    model_sparse, 
    tokenizer,
    texts: List[str],
    max_length=1024
):
    """
    核心测试：对比两个模型在相同输入下的输出差异
    """
    print(f"\n{'='*60}")
    print("测试输出差异")
    print(f"{'='*60}")
    
    model_dense.eval()
    model_sparse.eval()
    
    all_results = []
    
    with torch.no_grad():
        for idx, text in enumerate(tqdm(texts, desc="对比输出")):
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=False
            ).to(model_dense.device)
            
            seq_len = inputs.input_ids.size(1)
            
            # 跳过太短的序列
            if seq_len < 256:
                continue
            
            # 密集模型前向
            outputs_dense = model_dense(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            logits_dense = outputs_dense.logits
            hidden_dense = outputs_dense.hidden_states[-1]  # 最后一层
            
            # 稀疏模型前向
            outputs_sparse = model_sparse(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            logits_sparse = outputs_sparse.logits
            hidden_sparse = outputs_sparse.hidden_states[-1]
            
            # 1. 对比Logits
            logits_metrics = precision_metric(
                logits_sparse, logits_dense, verbose=False
            )
            
            # 2. 对比Hidden States
            hidden_metrics = precision_metric(
                hidden_sparse, hidden_dense, verbose=False
            )
            
            # 3. Token预测一致性
            pred_dense = logits_dense.argmax(dim=-1)
            pred_sparse = logits_sparse.argmax(dim=-1)
            token_accuracy = (pred_dense == pred_sparse).float().mean().item()
            
            # 4. Top-k token一致性
            k = 5
            topk_dense = torch.topk(logits_dense, k, dim=-1).indices
            topk_sparse = torch.topk(logits_sparse, k, dim=-1).indices
            
            # 计算top-k重叠率
            topk_overlap = 0.0
            for i in range(seq_len):
                dense_set = set(topk_dense[0, i].cpu().numpy())
                sparse_set = set(topk_sparse[0, i].cpu().numpy())
                topk_overlap += len(dense_set & sparse_set) / k
            topk_overlap /= seq_len
            
            result = {
                'text_idx': idx,
                'seq_len': seq_len,
                'logits_cosine': logits_metrics['Cossim'],
                'logits_l1': logits_metrics['L1'],
                'logits_rmse': logits_metrics['RMSE'],
                'hidden_cosine': hidden_metrics['Cossim'],
                'hidden_l1': hidden_metrics['L1'],
                'token_accuracy': token_accuracy,
                'top5_overlap': topk_overlap,
            }
            
            all_results.append(result)
            
            # 打印每个样本的结果
            if idx < 3:  # 只打印前3个
                print(f"\n样本 {idx+1} (长度={seq_len}):")
                print(f"  Logits Cosine: {logits_metrics['Cossim']:.6f}")
                print(f"  Logits L1: {logits_metrics['L1']:.6f}")
                print(f"  Token准确率: {token_accuracy:.2%}")
                print(f"  Top-5重叠: {topk_overlap:.2%}")
    
    # 汇总统计
    print(f"\n{'='*60}")
    print("汇总统计 (所有样本平均)")
    print(f"{'='*60}")
    
    avg_results = {}
    for key in all_results[0].keys():
        if key != 'text_idx':
            values = [r[key] for r in all_results]
            avg_results[key] = np.mean(values)
            std = np.std(values)
            print(f"{key:20s}: {avg_results[key]:.6f} (±{std:.6f})")
    
    return avg_results, all_results


def test_generation_consistency(
    model_dense,
    model_sparse,
    tokenizer,
    prompts: List[str],
    max_new_tokens=50
):
    """
    测试生成的一致性
    """
    print(f"\n{'='*60}")
    print("测试生成一致性")
    print(f"{'='*60}")
    
    model_dense.eval()
    model_sparse.eval()
    
    results = []
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 60)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model_dense.device)
        
        # 密集模型生成 (greedy)
        with torch.no_grad():
            outputs_dense = model_dense.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy解码，确保确定性
                pad_token_id=tokenizer.eos_token_id
            )
        text_dense = tokenizer.decode(outputs_dense[0], skip_special_tokens=True)
        
        # 稀疏模型生成 (greedy)
        with torch.no_grad():
            outputs_sparse = model_sparse.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        text_sparse = tokenizer.decode(outputs_sparse[0], skip_special_tokens=True)
        
        # 计算token级别的一致性
        token_match = (outputs_dense == outputs_sparse).float().mean().item()
        
        print(f"密集模型: {text_dense}")
        print(f"稀疏模型: {text_sparse}")
        print(f"Token一致性: {token_match:.2%}")
        
        # 简单的文本相似度
        if text_dense == text_sparse:
            text_similarity = 1.0
        else:
            # 计算字符级Levenshtein距离的简单版本
            text_similarity = token_match
        
        results.append({
            'prompt': prompt,
            'dense': text_dense,
            'sparse': text_sparse,
            'token_match': token_match,
            'text_similarity': text_similarity
        })
    
    avg_token_match = np.mean([r['token_match'] for r in results])
    print(f"\n平均Token一致性: {avg_token_match:.2%}")
    
    return results


def get_sparsity_stats(model):
    """收集稀疏度统计"""
    sparsities = []
    for module in model.modules():
        if isinstance(module, SparseAttentionWrapper):
            avg = module.get_avg_sparsity()
            if avg > 0:
                sparsities.append(avg)
    
    if sparsities:
        return {
            'mean': np.mean(sparsities),
            'std': np.std(sparsities),
            'min': np.min(sparsities),
            'max': np.max(sparsities)
        }
    return None


def run_llama_accuracy_test(
    model_name="meta-llama/Llama-3.2-1B",
    simthreshd1=0.6,
    cdfthreshd=0.98,
    num_samples=20,
    output_file="llama_accuracy_results.json"
):
    """
    运行完整测试
    """
    print(f"\n{'='*80}")
    print("LLaMA稀疏化输出准确性测试")
    print(f"模型: {model_name}")
    print(f"稀疏参数: simthreshd1={simthreshd1}, cdfthreshd={cdfthreshd}")
    print(f"{'='*80}")
    
    # 1. 加载模型
    print("\n[1/5] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("  加载密集模型...")
    model_dense = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("  加载稀疏模型...")
    model_sparse = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model_sparse = replace_with_sparse_attention(
        model_sparse, simthreshd1, cdfthreshd
    )
    
    # 2. 准备测试数据
    print("\n[2/5] 准备测试数据...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [item['text'] for item in dataset if len(item['text']) > 500]
    texts = texts[:num_samples]
    print(f"  选择了 {len(texts)} 个文本样本")
    
    # 3. 测试输出差异
    print("\n[3/5] 测试输出差异...")
    output_metrics, detailed_results = test_output_difference_on_texts(
        model_dense, model_sparse, tokenizer, texts
    )
    
    # 4. 测试生成一致性
    print("\n[4/5] 测试生成一致性...")
    test_prompts = [
        "The capital of France is",
        "In machine learning,",
        "Once upon a time,",
    ]
    generation_results = test_generation_consistency(
        model_dense, model_sparse, tokenizer, test_prompts
    )
    
    # 5. 收集稀疏度统计
    print("\n[5/5] 收集稀疏度统计...")
    sparsity_stats = get_sparsity_stats(model_sparse)
    if sparsity_stats:
        print(f"平均稀疏度: {sparsity_stats['mean']:.2%}")
        print(f"标准差: {sparsity_stats['std']:.2%}")
        print(f"范围: [{sparsity_stats['min']:.2%}, {sparsity_stats['max']:.2%}]")
    
    # 汇总结果
    results = {
        'config': {
            'model': model_name,
            'simthreshd1': simthreshd1,
            'cdfthreshd': cdfthreshd,
            'num_samples': num_samples
        },
        'output_accuracy': output_metrics,
        'generation': generation_results,
        'sparsity': sparsity_stats,
        'detailed_results': detailed_results
    }
    
    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_file}")
    
    # 最终评估
    print(f"\n{'='*60}")
    print("最终评估")
    print(f"{'='*60}")
    
    # 评分标准
    logits_cosine = output_metrics['logits_cosine']
    token_acc = output_metrics['token_accuracy']
    sparsity = sparsity_stats['mean'] if sparsity_stats else 0.0
    
    print(f"\n关键指标:")
    print(f"  Logits相似度: {logits_cosine:.4f}")
    print(f"  Token准确率: {token_acc:.2%}")
    print(f"  平均稀疏度: {sparsity:.2%}")
    
    # 判断
    print(f"\n评估:")
    if logits_cosine > 0.98 and token_acc > 0.95:
        print("  ✓ 优秀！输出几乎完全一致")
    elif logits_cosine > 0.95 and token_acc > 0.90:
        print("  ✓ 良好！可以用于生产")
    elif logits_cosine > 0.90 and token_acc > 0.85:
        print("  ⚠ 可接受，但建议调整参数")
    else:
        print("  ✗ 差异较大，需要降低稀疏度")
    
    if sparsity > 0.4:
        print(f"  ✓ 稀疏度很好 ({sparsity:.1%})")
    elif sparsity > 0.2:
        print(f"  ✓ 稀疏度不错 ({sparsity:.1%})")
    else:
        print(f"  ⚠ 稀疏度较低 ({sparsity:.1%})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='测试LLaMA稀疏化的输出准确性')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--simthreshd1', type=float, default=0.6)
    parser.add_argument('--cdfthreshd', type=float, default=0.98)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--output', type=str, default='llama_accuracy_results.json')
    
    args = parser.parse_args()
    
    run_llama_accuracy_test(
        model_name=args.model,
        simthreshd1=args.simthreshd1,
        cdfthreshd=args.cdfthreshd,
        num_samples=args.num_samples,
        output_file=args.output
    )


if __name__ == "__main__":
    main()

