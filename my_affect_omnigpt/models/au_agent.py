#!/usr/bin/env python3
"""
AU Agent模块
用于从OpenFace AU检测结果生成自然语言描述
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, List, Optional
import numpy as np


class AUAgent:
    """AU Agent - 将AU检测结果转换为自然语言描述"""
    
    def __init__(
        self,
        base_model_path: str,
        lora_weights_path: str = None,
        device: str = "cuda",
        use_lora: bool = True
    ):
        """
        初始化AU Agent
        
        Args:
            base_model_path: 基础模型路径（Qwen2.5-7B）
            lora_weights_path: LoRA权重路径（如果use_lora=True）
            device: 设备
            use_lora: 是否使用LoRA微调权重
        """
        self.device = device
        self.use_lora = use_lora
        
        print(f"[AU Agent] Loading from {base_model_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        # 加载模型
        if use_lora and lora_weights_path:
            # 加载基础模型 + LoRA
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(base_model, lora_weights_path)
            self.model = self.model.merge_and_unload()
            print(f"[AU Agent] LoRA weights loaded from {lora_weights_path}")
        else:
            # 只加载基础模型
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        
        # 移动到指定设备
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[AU Agent] Model loaded successfully")
    
    def parse_openface_csv(self, csv_row: Dict) -> Dict[str, float]:
        """
        解析OpenFace CSV行，提取AU强度
        
        Args:
            csv_row: OpenFace CSV的一行数据（字典）
        
        Returns:
            AU强度字典，如 {"AU01": 0.98, "AU02": 1.5, ...}
        """
        au_values = {}
        
        # OpenFace AU强度列名: AU01_r, AU02_r, ...
        for key, value in csv_row.items():
            if key.endswith('_r'):  # 强度（intensity）
                au_id = key[:-2]  # 去掉_r
                try:
                    au_values[au_id] = float(value)
                except:
                    continue
        
        return au_values
    
    
    def generate_description(
        self,
        au_values: Dict[str, float],
        au_description: Optional[str] = None,
        max_length: int = 256,
        temperature: float = 0.7
    ) -> str:
        """
        生成AU描述（图1 EmoChat使用阶段：从AU值生成）
        
        Args:
            au_values: AU强度字典 {"AU01": 0.98, "AU02_r": 1.5, ...}
            au_description: 外部提供的AU描述（来自MER-Factory），如果提供则使用，否则自动生成
            max_length: 最大生成长度
            temperature: 采样温度
        
        Returns:
            自然语言客观肌肉运动描述（无情感词）
        """
        # 过滤低强度AU（移除_r后缀）
        significant_aus = {}
        for k, v in au_values.items():
            au_id = k.replace('_r', '')
            if v > 0.5:
                significant_aus[au_id] = v
        
        if not significant_aus:
            return "neutral expression with minimal facial movement"
        
        # AU名称映射
        au_name_map = {
            'AU01': 'Inner brow raiser',
            'AU02': 'Outer brow raiser',
            'AU04': 'Brow lowerer',
            'AU05': 'Upper lid raiser',
            'AU06': 'Cheek raiser',
            'AU07': 'Lid tightener',
            'AU09': 'Nose wrinkler',
            'AU10': 'Upper lip raiser',
            'AU12': 'Lip corner puller (smile)',
            'AU14': 'Dimpler',
            'AU15': 'Lip corner depressor',
            'AU17': 'Chin raiser',
            'AU20': 'Lip stretcher',
            'AU23': 'Lip tightener',
            'AU25': 'Lips part',
            'AU26': 'Jaw drop',
            'AU45': 'Blink'
        }
        
        # 构建AU result（AU值 + AU描述）
        au_values_list = []
        
        for au_id, value in significant_aus.items():
            au_values_list.append(f"{au_id}: {value:.2f}")
        
        au_values_text = ", ".join(au_values_list)
        
        # 如果外部提供了au_description（来自MER-Factory），直接使用
        if au_description:
            au_descriptions_text = au_description
        else:
            # 否则自动生成AU描述
            au_descriptions_list = []
            for au_id, value in significant_aus.items():
                au_name = au_name_map.get(au_id, au_id)
                au_descriptions_list.append(f"{au_name} (intensity: {value:.2f})")
            au_descriptions_text = ", ".join(au_descriptions_list)
        
        # 构建输入（推理时只使用AU result，不包含Emotion和Prompt）
        # AffectGPT推理时只需要客观的肌肉运动描述
        input_text = f"""AU values: {au_values_text}
AU descriptions: {au_descriptions_text}"""
        
        # instruction（简化的指令）
        instruction = "Generate a detailed and objective facial muscle movement description based on the Action Unit detections. Focus only on the physical movements without inferring emotions."
        
        # 使用Qwen2.5的chat template
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text}
        ]
        
        full_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([full_prompt], return_tensors="pt").to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取描述部分
        if "Description:" in response:
            description = response.split("Description:")[-1].strip()
        else:
            description = response.strip()
        
        return description
    
    def batch_generate_descriptions(
        self,
        au_values_list: List[Dict[str, float]],
        batch_size: int = 8
    ) -> List[str]:
        """
        批量生成AU描述
        
        Args:
            au_values_list: AU强度字典列表
            batch_size: 批处理大小
        
        Returns:
            描述列表
        """
        descriptions = []
        
        for i in range(0, len(au_values_list), batch_size):
            batch = au_values_list[i:i+batch_size]
            batch_descs = [self.generate_description(au_vals) for au_vals in batch]
            descriptions.extend(batch_descs)
        
        return descriptions


def create_au_agent(config) -> AUAgent:
    """
    工厂函数：根据配置创建AU Agent
    
    Args:
        config: 配置对象，包含：
            - au_agent_base_model: 基础模型路径
            - au_agent_lora_weights: LoRA权重路径（可选）
            - au_agent_use_lora: 是否使用LoRA
    
    Returns:
        AUAgent实例
    """
    base_model = getattr(config, 'au_agent_base_model', '/home/project/Dataset/Emotion/tools/transformer/LLM/Qwen2.5-7B-Instruct')
    lora_weights = getattr(config, 'au_agent_lora_weights', None)
    use_lora = getattr(config, 'au_agent_use_lora', lora_weights is not None)
    
    return AUAgent(
        base_model_path=base_model,
        lora_weights_path=lora_weights,
        use_lora=use_lora
    )
