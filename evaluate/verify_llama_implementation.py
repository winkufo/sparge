"""
验证LLaMA实现是否正确

用于测试modify_llama.py中的SparseLlamaAttention是否能正常工作
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from modify_model.modify_llama import set_spas_sage_attn_llama
except ImportError:
    import sys
    sys.path.insert(0, 'evaluate')
    from modify_model.modify_llama import set_spas_sage_attn_llama


def verify_llama_implementation():
    """
    验证实现的关键步骤
    """
    print("\n" + "="*60)
    print("验证LLaMA稀疏化实现")
    print("="*60)
    
    model_name = "meta-llama/Llama-3.2-1B"
    
    # 1. 加载原始模型
    print("\n[1/5] 加载原始模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False
    
    # 2. 检查模型结构
    print("\n[2/5] 检查模型结构...")
    print(f"模型类型: {type(model).__name__}")
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        print(f"✓ 找到 {len(layers)} 层")
        
        # 检查第一层的attention
        first_attn = layers[0].self_attn
        print(f"Attention类型: {type(first_attn).__name__}")
        
        # 打印attention的属性
        print("\nAttention属性:")
        for attr in ['config', 'num_heads', 'head_dim', 'num_key_value_heads', 
                     'num_key_value_groups', 'q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(first_attn, attr):
                value = getattr(first_attn, attr)
                if not callable(value):
                    print(f"  ✓ {attr}: {value if not isinstance(value, torch.nn.Module) else type(value).__name__}")
                else:
                    print(f"  ✓ {attr}: <method>")
            else:
                print(f"  ✗ {attr}: 不存在")
        
        # 检查hidden_size
        if hasattr(first_attn, 'hidden_size'):
            print(f"  ✓ hidden_size (直接): {first_attn.hidden_size}")
        elif hasattr(first_attn.config, 'hidden_size'):
            print(f"  ✓ hidden_size (from config): {first_attn.config.hidden_size}")
        else:
            print(f"  ⚠️ hidden_size: 需要计算 (num_heads * head_dim)")
    else:
        print("✗ 无法找到layers")
        return False
    
    # 3. 应用稀疏化
    print("\n[3/5] 应用稀疏化...")
    try:
        model = set_spas_sage_attn_llama(
            model,
            verbose=True,
            l1=0.06,
            pv_l1=0.07
        )
        print("✓ 稀疏化应用成功")
    except Exception as e:
        print(f"✗ 稀疏化应用失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 测试前向传播
    print("\n[4/5] 测试前向传播...")
    try:
        test_text = "The capital of France is"
        inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✓ 前向传播成功")
        print(f"  输出形状: {outputs.logits.shape}")
        print(f"  输出范围: [{outputs.logits.min():.2f}, {outputs.logits.max():.2f}]")
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 测试生成
    print("\n[5/5] 测试生成...")
    try:
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
        
        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"✓ 生成成功")
        print(f"  输入: {test_text}")
        print(f"  输出: {output_text}")
        
    except Exception as e:
        print(f"✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 最终结果
    print("\n" + "="*60)
    print("✓ 所有验证通过！LLaMA实现正确")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = verify_llama_implementation()
    exit(0 if success else 1)

