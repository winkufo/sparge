"""
修改LLaMA模型，替换attention为稀疏版本

关键差异：
- LLaMA使用causal attention (is_causal=True)
- LLaMA有GQA (Grouped Query Attention)
- LLaMA的attention直接在模型内部，不使用processor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel
from spas_sage_attn.autotune import SparseAttentionMeansim
from typing import Optional, Tuple


class SparseLlamaAttention(nn.Module):
    """
    稀疏版本的LLaMA Attention
    
    保持与原始LlamaAttention完全兼容的接口
    """
    
    def __init__(
        self,
        original_attn: LlamaAttention,
        l1: float = 0.06,
        pv_l1: float = 0.07,
        layer_idx: int = 0
    ):
        super().__init__()
        
        # 保存原始attention的所有属性
        self.original_attn = original_attn
        self.layer_idx = layer_idx
        
        # 复制必要的属性
        self.config = original_attn.config
        self.hidden_size = original_attn.hidden_size
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim
        self.num_key_value_heads = original_attn.num_key_value_heads
        self.num_key_value_groups = original_attn.num_key_value_groups
        self.max_position_embeddings = original_attn.max_position_embeddings
        self.rope_theta = original_attn.rope_theta
        self.is_causal = True
        
        # 复制所有的projection层和rotary embedding
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj
        self.rotary_emb = original_attn.rotary_emb
        
        # 稀疏attention模块
        self.sparse_attn = SparseAttentionMeansim(
            l1=l1,
            pv_l1=pv_l1,
            rearrange_kwargs={}
        )
        
        # 标记是否在调优模式
        self.tune_mode = False
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        GQA: 重复k/v heads以匹配query heads数量
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # Q, K, V projection
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # RoPE (Rotary Position Embedding)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = self.original_attn.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        
        # KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # GQA: repeat k/v heads if necessary
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # Sparse Attention
        kv_seq_len = key_states.shape[-2]
        
        # 只在序列足够长时使用稀疏attention
        if kv_seq_len >= 256:
            try:
                attn_output = self.sparse_attn(
                    query_states,
                    key_states,
                    value_states,
                    is_causal=True,
                    tensor_layout="HND",
                    tune_mode=self.tune_mode
                )
            except Exception as e:
                # Fallback to dense attention
                print(f"Sparse attention failed at layer {self.layer_idx}: {e}")
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
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value


def set_spas_sage_attn_llama(
    model,
    verbose: bool = False,
    l1: float = 0.06,
    pv_l1: float = 0.07,
    layer_range: Optional[Tuple[int, int]] = None
):
    """
    为LLaMA模型设置稀疏attention
    
    Args:
        model: LLaMA模型
        verbose: 是否打印详细信息
        l1: QK稀疏化的L1误差上限
        pv_l1: PV稀疏化的L1误差上限
        layer_range: 只替换指定范围的层，例如 (10, 32) 表示只替换第10-31层
                    如果为None，则替换所有层
    """
    
    # 获取模型的layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # AutoModelForCausalLM
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        # LlamaModel
        layers = model.layers
    else:
        raise ValueError("Cannot find layers in the model")
    
    num_layers = len(layers)
    
    # 确定要替换的层范围
    if layer_range is not None:
        start_layer, end_layer = layer_range
        start_layer = max(0, start_layer)
        end_layer = min(num_layers, end_layer)
    else:
        start_layer, end_layer = 0, num_layers
    
    replaced_count = 0
    
    for layer_idx in range(start_layer, end_layer):
        layer = layers[layer_idx]
        
        if hasattr(layer, 'self_attn'):
            original_attn = layer.self_attn
            
            # 创建稀疏attention
            sparse_attn = SparseLlamaAttention(
                original_attn,
                l1=l1,
                pv_l1=pv_l1,
                layer_idx=layer_idx
            )
            
            # 替换
            layer.self_attn = sparse_attn
            replaced_count += 1
            
            if verbose:
                print(f"✓ Replaced layer {layer_idx} attention with sparse version")
    
    print(f"\n总共替换了 {replaced_count}/{num_layers} 个attention层 (层 {start_layer}-{end_layer-1})")
    
    return model


def enable_tune_mode(model, enable: bool = True):
    """
    启用或禁用调优模式
    
    在调优模式下，SparseAttentionMeansim会自动搜索最优参数
    """
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        return
    
    for layer in layers:
        if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, SparseLlamaAttention):
            layer.self_attn.tune_mode = enable
            layer.self_attn.sparse_attn.tune_mode = enable
    
    mode_str = "启用" if enable else "禁用"
    print(f"{mode_str}调优模式")


def get_sparsity_statistics(model):
    """
    获取稀疏度统计信息
    """
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        return None
    
    stats = []
    
    for layer_idx, layer in enumerate(layers):
        if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, SparseLlamaAttention):
            sparse_attn = layer.self_attn.sparse_attn
            if hasattr(sparse_attn, 'tuning_sparsity') and sparse_attn.tuning_sparsity is not None:
                mean_sparsity = sparse_attn.tuning_sparsity.mean().item()
                stats.append({
                    'layer_idx': layer_idx,
                    'mean_sparsity': mean_sparsity,
                    'cdfthreshd': sparse_attn.cdfthreshd.mean().item() if sparse_attn.cdfthreshd is not None else 0,
                    'simthreshd1': sparse_attn.simthreshd1.mean().item() if sparse_attn.simthreshd1 is not None else 0,
                })
    
    return stats

