"""
在真实LLaMA 3模型上测试SpargeAttn的准确性

测试策略：
1. 加载预训练LLaMA 3模型
2. 替换attention层为稀疏版本
3. 测试perplexity (困惑度) - 语言模型最关键指标
4. 测试生成质量 - 人类可读性评估
5. 测试各种NLP任务的准确度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
from typing import Dict, List, Optional
import argparse

from spas_sage_attn import spas_sage2_attn_meansim_cuda
from spas_sage_attn.utils import precision_metric


class SparseAttentionWrapper(nn.Module):
    """
    包装器：替换标准attention为稀疏attention
    保持接口兼容LLaMA的Attention模块
    """
    
    def __init__(self, original_attn, simthreshd1=0.6, cdfthreshd=0.98):
        super().__init__()
        self.original_attn = original_attn
        self.simthreshd1 = simthreshd1
        self.cdfthreshd = cdfthreshd
        self.use_sparse = True  # 可以动态切换
        
        # 统计信息
        self.total_calls = 0
        self.total_sparsity = 0.0
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        """
        与LLaMA Attention接口兼容的forward函数
        """
        if not self.use_sparse:
            # 使用原始attention
            return self.original_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs
            )
        
        # 获取原始attention的Q, K, V投影
        bsz, q_len, _ = hidden_states.size()
        
        # 使用原始模型的projection
        query_states = self.original_attn.q_proj(hidden_states)
        key_states = self.original_attn.k_proj(hidden_states)
        value_states = self.original_attn.v_proj(hidden_states)
        
        # Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, self.original_attn.num_heads, self.original_attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.original_attn.num_key_value_heads, self.original_attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.original_attn.num_key_value_heads, self.original_attn.head_dim).transpose(1, 2)
        
        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        # GQA: repeat k/v heads if necessary
        if self.original_attn.num_key_value_heads != self.original_attn.num_heads:
            key_states = self._repeat_kv(key_states, self.original_attn.num_key_value_groups)
            value_states = self._repeat_kv(value_states, self.original_attn.num_key_value_groups)
        
        # 使用稀疏attention
        kv_seq_len = key_states.shape[-2]
        
        # 只在序列足够长时使用稀疏
        if kv_seq_len >= 512:
            try:
                attn_output, sparsity = spas_sage2_attn_meansim_cuda(
                    query_states,
                    key_states,
                    value_states,
                    is_causal=True,  # LLaMA使用causal mask
                    simthreshd1=self.simthreshd1,
                    cdfthreshd=self.cdfthreshd,
                    return_sparsity=True
                )
                
                # 统计稀疏度
                self.total_calls += 1
                self.total_sparsity += sparsity
                
            except Exception as e:
                # Fallback到标准attention
                print(f"Sparse attention failed: {e}, using dense attention")
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states, value_states, is_causal=True
                )
        else:
            # 短序列使用标准attention
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states, is_causal=True
            )
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.original_attn.hidden_size)
        
        # Output projection
        attn_output = self.original_attn.o_proj(attn_output)
        
        if use_cache:
            return attn_output, None, (key_states, value_states)
        else:
            return attn_output, None, None
    
    def _repeat_kv(self, hidden_states, n_rep):
        """GQA: repeat k/v heads"""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    def get_avg_sparsity(self):
        """获取平均稀疏度"""
        if self.total_calls == 0:
            return 0.0
        return self.total_sparsity / self.total_calls


def replace_attention_with_sparse(model, simthreshd1=0.6, cdfthreshd=0.98):
    """
    替换模型中所有的attention层为稀疏版本
    """
    replaced_count = 0
    
    for name, module in model.named_modules():
        # LLaMA 3的attention模块通常命名为 'self_attn'
        if hasattr(module, 'self_attn'):
            original_attn = module.self_attn
            module.self_attn = SparseAttentionWrapper(
                original_attn,
                simthreshd1=simthreshd1,
                cdfthreshd=cdfthreshd
            )
            replaced_count += 1
            print(f"替换了 {name}.self_attn")
    
    print(f"\n总共替换了 {replaced_count} 个attention层")
    return model


def compute_perplexity(
    model,
    tokenizer,
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    max_samples=100,
    max_length=2048
):
    """
    计算模型的困惑度（Perplexity）
    
    这是评估语言模型最重要的指标：
    - PPL越低越好
    - PPL = exp(average negative log-likelihood)
    """
    print(f"\n{'='*60}")
    print(f"计算Perplexity on {dataset_name}")
    print(f"{'='*60}")
    
    # 加载数据集
    if dataset_name == "wikitext":
        dataset = load_dataset(dataset_name, dataset_config, split="test")
        texts = [item['text'] for item in dataset if len(item['text']) > 100]
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 限制样本数
    texts = texts[:max_samples]
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="计算困惑度"):
            # Tokenize
            encodings = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True
            ).to(model.device)
            
            input_ids = encodings.input_ids
            
            if input_ids.size(1) < 2:
                continue
            
            # Forward pass
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            # 累积
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)
    
    # 计算平均loss和perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_samples': len(texts),
        'num_tokens': total_tokens
    }


def test_generation_quality(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens=100,
    temperature=0.7
):
    """
    测试生成质量
    
    通过实际生成文本来评估模型是否还能生成合理的内容
    """
    print(f"\n{'='*60}")
    print("测试生成质量")
    print(f"{'='*60}")
    
    model.eval()
    generations = []
    
    with torch.no_grad():
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            print("-" * 60)
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated: {generated_text}")
            
            generations.append({
                'prompt': prompt,
                'generated': generated_text
            })
    
    return generations


def test_output_consistency(
    model_dense,
    model_sparse,
    tokenizer,
    test_texts: List[str],
    max_length=1024
):
    """
    对比密集模型和稀疏模型的输出一致性
    
    这是最直接的测试：相同输入，输出应该接近
    """
    print(f"\n{'='*60}")
    print("测试输出一致性（最关键测试）")
    print(f"{'='*60}")
    
    model_dense.eval()
    model_sparse.eval()
    
    all_metrics = []
    
    with torch.no_grad():
        for text in tqdm(test_texts, desc="对比输出"):
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True
            ).to(model_dense.device)
            
            if inputs.input_ids.size(1) < 128:
                continue  # 跳过太短的文本
            
            # Dense model
            outputs_dense = model_dense(**inputs, output_hidden_states=True)
            logits_dense = outputs_dense.logits
            hidden_dense = outputs_dense.hidden_states[-1]
            
            # Sparse model
            outputs_sparse = model_sparse(**inputs, output_hidden_states=True)
            logits_sparse = outputs_sparse.logits
            hidden_sparse = outputs_sparse.hidden_states[-1]
            
            # 对比logits
            logits_metrics = precision_metric(
                logits_sparse,
                logits_dense,
                verbose=False
            )
            
            # 对比hidden states
            hidden_metrics = precision_metric(
                hidden_sparse,
                hidden_dense,
                verbose=False
            )
            
            # Token-level accuracy
            pred_dense = logits_dense.argmax(dim=-1)
            pred_sparse = logits_sparse.argmax(dim=-1)
            token_accuracy = (pred_dense == pred_sparse).float().mean().item()
            
            metrics = {
                'logits_cosine': logits_metrics['Cossim'],
                'logits_l1': logits_metrics['L1'],
                'hidden_cosine': hidden_metrics['Cossim'],
                'hidden_l1': hidden_metrics['L1'],
                'token_accuracy': token_accuracy,
                'seq_len': inputs.input_ids.size(1)
            }
            
            all_metrics.append(metrics)
    
    # 汇总统计
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    print(f"\n平均指标:")
    print(f"{'指标':<25} {'值':<10}")
    print("-" * 35)
    print(f"{'Logits Cosine相似度':<25} {avg_metrics['logits_cosine']:.6f}")
    print(f"{'Logits L1误差':<25} {avg_metrics['logits_l1']:.6f}")
    print(f"{'Hidden Cosine相似度':<25} {avg_metrics['hidden_cosine']:.6f}")
    print(f"{'Hidden L1误差':<25} {avg_metrics['hidden_l1']:.6f}")
    print(f"{'Token准确率':<25} {avg_metrics['token_accuracy']:.2%}")
    print(f"{'平均序列长度':<25} {avg_metrics['seq_len']:.0f}")
    
    return avg_metrics, all_metrics


def collect_sparsity_statistics(model):
    """
    收集所有稀疏attention层的稀疏度统计
    """
    print(f"\n{'='*60}")
    print("稀疏度统计")
    print(f"{'='*60}")
    
    sparsities = []
    
    for name, module in model.named_modules():
        if isinstance(module, SparseAttentionWrapper):
            avg_sparsity = module.get_avg_sparsity()
            if module.total_calls > 0:
                print(f"{name}: 平均稀疏度 {avg_sparsity:.2%} "
                      f"(调用次数: {module.total_calls})")
                sparsities.append(avg_sparsity)
    
    if sparsities:
        overall_avg = np.mean(sparsities)
        print(f"\n整体平均稀疏度: {overall_avg:.2%}")
        return overall_avg
    else:
        print("未收集到稀疏度数据")
        return 0.0


def run_comprehensive_llama_test(
    model_name="meta-llama/Llama-3.2-1B",
    simthreshd1=0.6,
    cdfthreshd=0.98,
    max_samples=50,
    output_file="llama_test_results.json"
):
    """
    运行完整的LLaMA测试套件
    """
    print(f"\n{'='*80}")
    print(f"开始LLaMA 3模型稀疏化准确性测试")
    print(f"模型: {model_name}")
    print(f"稀疏参数: simthreshd1={simthreshd1}, cdfthreshd={cdfthreshd}")
    print(f"{'='*80}")
    
    # 1. 加载模型和tokenizer
    print("\n[1/7] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载原始模型（用于对比）
    print("加载密集模型（ground truth）...")
    model_dense = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 加载稀疏模型
    print("加载稀疏模型...")
    model_sparse = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model_sparse = replace_attention_with_sparse(
        model_sparse,
        simthreshd1=simthreshd1,
        cdfthreshd=cdfthreshd
    )
    
    results = {}
    
    # 2. 测试Perplexity（密集模型）
    print("\n[2/7] 计算密集模型的Perplexity...")
    ppl_dense = compute_perplexity(
        model_dense,
        tokenizer,
        max_samples=max_samples
    )
    results['perplexity_dense'] = ppl_dense
    
    # 3. 测试Perplexity（稀疏模型）
    print("\n[3/7] 计算稀疏模型的Perplexity...")
    ppl_sparse = compute_perplexity(
        model_sparse,
        tokenizer,
        max_samples=max_samples
    )
    results['perplexity_sparse'] = ppl_sparse
    
    # 4. 对比困惑度
    ppl_increase = ppl_sparse['perplexity'] - ppl_dense['perplexity']
    ppl_increase_pct = (ppl_increase / ppl_dense['perplexity']) * 100
    
    print(f"\n{'='*60}")
    print("Perplexity对比")
    print(f"{'='*60}")
    print(f"密集模型: {ppl_dense['perplexity']:.4f}")
    print(f"稀疏模型: {ppl_sparse['perplexity']:.4f}")
    print(f"增加: {ppl_increase:+.4f} ({ppl_increase_pct:+.2f}%)")
    
    if abs(ppl_increase_pct) < 2:
        print("✓ Perplexity变化 < 2%, 优秀！")
    elif abs(ppl_increase_pct) < 5:
        print("✓ Perplexity变化 < 5%, 良好")
    elif abs(ppl_increase_pct) < 10:
        print("⚠ Perplexity变化 < 10%, 可接受")
    else:
        print("✗ Perplexity变化 > 10%, 需要调整参数")
    
    results['perplexity_comparison'] = {
        'increase': ppl_increase,
        'increase_pct': ppl_increase_pct
    }
    
    # 5. 输出一致性测试
    print("\n[4/7] 测试输出一致性...")
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_texts = [item['text'] for item in test_dataset if len(item['text']) > 200][:20]
    
    consistency_metrics, _ = test_output_consistency(
        model_dense,
        model_sparse,
        tokenizer,
        test_texts
    )
    results['consistency'] = consistency_metrics
    
    # 6. 生成质量测试
    print("\n[5/7] 测试生成质量...")
    test_prompts = [
        "The capital of France is",
        "In machine learning, the most important concept is",
        "Once upon a time, there was a",
    ]
    
    print("\n密集模型生成:")
    gen_dense = test_generation_quality(model_dense, tokenizer, test_prompts)
    
    print("\n稀疏模型生成:")
    gen_sparse = test_generation_quality(model_sparse, tokenizer, test_prompts)
    
    results['generations'] = {
        'dense': gen_dense,
        'sparse': gen_sparse
    }
    
    # 7. 稀疏度统计
    print("\n[6/7] 收集稀疏度统计...")
    avg_sparsity = collect_sparsity_statistics(model_sparse)
    results['sparsity'] = avg_sparsity
    
    # 8. 生成报告
    print("\n[7/7] 生成测试报告...")
    results['config'] = {
        'model_name': model_name,
        'simthreshd1': simthreshd1,
        'cdfthreshd': cdfthreshd,
        'max_samples': max_samples
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n完整结果已保存至: {output_file}")
    
    # 最终评分
    print(f"\n{'='*60}")
    print("最终评估")
    print(f"{'='*60}")
    
    score = 0
    max_score = 0
    
    # Perplexity评分 (40分)
    max_score += 40
    if abs(ppl_increase_pct) < 2:
        score += 40
        print("✓ Perplexity: 40/40")
    elif abs(ppl_increase_pct) < 5:
        score += 30
        print("✓ Perplexity: 30/40")
    elif abs(ppl_increase_pct) < 10:
        score += 20
        print("⚠ Perplexity: 20/40")
    else:
        score += 10
        print("✗ Perplexity: 10/40")
    
    # 输出一致性评分 (40分)
    max_score += 40
    if consistency_metrics['logits_cosine'] > 0.98:
        score += 40
        print("✓ 输出一致性: 40/40")
    elif consistency_metrics['logits_cosine'] > 0.95:
        score += 30
        print("✓ 输出一致性: 30/40")
    elif consistency_metrics['logits_cosine'] > 0.90:
        score += 20
        print("⚠ 输出一致性: 20/40")
    else:
        score += 10
        print("✗ 输出一致性: 10/40")
    
    # 稀疏度评分 (20分)
    max_score += 20
    if avg_sparsity > 0.4:
        score += 20
        print(f"✓ 稀疏度: 20/20 ({avg_sparsity:.1%})")
    elif avg_sparsity > 0.3:
        score += 15
        print(f"✓ 稀疏度: 15/20 ({avg_sparsity:.1%})")
    elif avg_sparsity > 0.2:
        score += 10
        print(f"⚠ 稀疏度: 10/20 ({avg_sparsity:.1%})")
    else:
        score += 5
        print(f"✗ 稀疏度: 5/20 ({avg_sparsity:.1%})")
    
    print(f"\n总分: {score}/{max_score}")
    
    if score >= 85:
        print("🎉 优秀！可以部署到生产环境")
    elif score >= 70:
        print("✓ 良好，适合大多数应用")
    elif score >= 50:
        print("⚠ 可接受，建议调整参数")
    else:
        print("✗ 需要改进")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='测试LLaMA模型的稀疏化准确性')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B',
                       help='模型名称或路径')
    parser.add_argument('--simthreshd1', type=float, default=0.6,
                       help='相似度阈值')
    parser.add_argument('--cdfthreshd', type=float, default=0.98,
                       help='CDF阈值')
    parser.add_argument('--max_samples', type=int, default=50,
                       help='测试样本数')
    parser.add_argument('--output', type=str, default='llama_test_results.json',
                       help='输出文件路径')
    
    args = parser.parse_args()
    
    results = run_comprehensive_llama_test(
        model_name=args.model,
        simthreshd1=args.simthreshd1,
        cdfthreshd=args.cdfthreshd,
        max_samples=args.max_samples,
        output_file=args.output
    )
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()

