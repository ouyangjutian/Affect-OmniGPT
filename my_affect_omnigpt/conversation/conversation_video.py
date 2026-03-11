import re
import copy
import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any
from PIL import Image
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from my_affectgpt.common.registry import registry
from my_affectgpt.processors import Blip2ImageEvalProcessor
from my_affectgpt.processors.video_processor import ToTHWC, ToUint8, load_video, load_face
from my_affectgpt.models.ImageBind.data import load_audio, transform_audio
from my_affectgpt.datasets.builders.image_text_pair_builder import *
from my_affectgpt.models.au_agent import AUAgent
import config

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


default_conversation = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


######################################################
# ============== (only for inference) ============== #
######################################################
class Chat:
    def __init__(self, model, model_cfg, device='cuda:0'):
        self.device = device
        self.model = model
        self.tokenizer = model.llama_tokenizer
       
        # 采用更加通用的格式，单独的 “#” 和在文本中的“a#” 采用的编码符号是不同的
        id_tre_jin = self.tokenizer('###', add_special_tokens=False)['input_ids'][0] # 835
        id_two_jin = self.tokenizer('a##', add_special_tokens=False)['input_ids'][1] # 2277
        id_one_jin = self.tokenizer('a#', add_special_tokens=False)['input_ids'][1]  # 29937
        stop_words_ids = [torch.tensor([self.tokenizer.eos_token_id]).to(self.device),
                          torch.tensor([id_tre_jin]).to(self.device),
                          torch.tensor([id_two_jin, id_one_jin]).to(self.device),
                          torch.tensor([id_one_jin, id_two_jin]).to(self.device)] # three id for "### / # ## "
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        # 设置其他参数
        self.num_video_query_token = model_cfg.num_video_query_token
        self.num_audio_query_token = model_cfg.num_audio_query_token
        self.num_multi_query_token = model_cfg.num_multi_query_token
        self.num_image_query_token = model_cfg.num_image_query_token
        self.num_au_query_token = getattr(model_cfg, 'num_au_query_token', 8)  # AU query token数量
        
        # 初始化AU Agent（如果配置中启用）
        self.use_au_agent = getattr(model_cfg, 'use_au_agent', False)
        if self.use_au_agent:
            au_base_model = getattr(model_cfg, 'au_agent_base_model', '/home/project/Dataset/Emotion/tools/transformer/LLM/Qwen2.5-7B-Instruct')
            au_lora_weights = getattr(model_cfg, 'au_agent_lora_weights', '/home/project/AffectGPT/AffectGPT/output/au_agent_qwen2.5_7b_lora')
            print(f"[Chat] Loading AU Agent from {au_lora_weights}")
            self.au_agent = AUAgent(
                base_model_path=au_base_model,
                lora_weights_path=au_lora_weights,
                device=device,
                use_lora=True
            )
        else:
            self.au_agent = None

    def to_token_ids(self, text, max_length):
        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   padding="longest",
                                   max_length=max_length,
                                   truncation=True,
                                   add_special_tokens=False).input_ids[0]
        return input_ids

    def replace_token_for_multimodal(self, prompt):
        replace_token = config.DEFAULT_FRAME_PATCH_TOKEN * self.num_video_query_token
        prompt = prompt.replace(config.DEFAULT_FRAME_PATCH_TOKEN, replace_token)
        replace_token = config.DEFAULT_FACE_PATCH_TOKEN * self.num_video_query_token
        prompt = prompt.replace(config.DEFAULT_FACE_PATCH_TOKEN, replace_token)
        replace_token = config.DEFAULT_AUDIO_PATCH_TOKEN * self.num_audio_query_token
        prompt = prompt.replace(config.DEFAULT_AUDIO_PATCH_TOKEN, replace_token)
        replace_token = config.DEFAULT_MULTI_PATCH_TOKEN * self.num_multi_query_token
        prompt = prompt.replace(config.DEFAULT_MULTI_PATCH_TOKEN, replace_token)
        replace_token = config.DEFAULT_IMAGE_PATCH_TOKEN * self.num_image_query_token
        prompt = prompt.replace(config.DEFAULT_IMAGE_PATCH_TOKEN, replace_token)
        # 🎯 Lexical信息直接作为文本嵌入，不需要token替换
        return prompt
   
    def postprocess_audio(self, sample_data):
        if 'audio' not in sample_data or sample_data['audio'] is None:
            return None, None
        
        audio = sample_data['audio'].unsqueeze(0).to(self.device)
        raw_audio = sample_data['raw_audio'].unsqueeze(0).to(self.device)
        audio_hiddens, audio_llms = self.model.encode_audio_merge(audio, raw_audio)
        return audio_hiddens, audio_llms

    def postprocess_face(self, sample_data):
        if 'face' not in sample_data or sample_data['face'] is None:
            return None, None
        
        # 检查是否为预提取特征
        is_preextracted = sample_data.get('face_preextracted', False)
        
        face = sample_data['face'].unsqueeze(0).to(self.device)
        raw_face = sample_data['raw_face'].unsqueeze(0).to(self.device)
        
        # 传递is_preextracted标志
        face_hiddens, face_llms = self.model.encode_video_merge(face, raw_face, is_preextracted=is_preextracted)
        return face_hiddens, face_llms
    
    def postprocess_frame(self, sample_data):
        if 'frame' not in sample_data or sample_data['frame'] is None:
            return None, None
        
        # 检查是否为预提取特征
        is_preextracted = sample_data.get('frame_preextracted', False)
        
        video = sample_data['frame'].unsqueeze(0).to(self.device)
        raw_video = sample_data['raw_frame'].unsqueeze(0).to(self.device)
        
        # 传递is_preextracted标志
        frame_hiddens, frame_llms = self.model.encode_video_merge(video, raw_video, is_preextracted=is_preextracted)
        return frame_hiddens, frame_llms

    def postprocess_image(self, sample_data):
        if 'image' not in sample_data or sample_data['image'] is None:
            return None, None
        
        image = sample_data['image'].unsqueeze(0).to(self.device)
        raw_image = sample_data['raw_image'].unsqueeze(0).to(self.device)
        image_hiddens, image_llms = self.model.encode_image_merge(image, raw_image)
        return image_hiddens, image_llms


    def postprocess_multi(self, video_hiddens, audio_hiddens):
        if video_hiddens is None or audio_hiddens is None:
            return None, None

        multi_hiddens, multi_llms = self.model.encode_multi_merge(video_hiddens, audio_hiddens)
        return multi_hiddens, multi_llms

    def postprocess_au(self, sample_data):
        """AU模态处理"""
        if sample_data is None or 'au' not in sample_data or sample_data['au'] is None:
            return None, None
        
        # 如果使用AU Agent，生成自然语言描述
        if self.use_au_agent and self.au_agent is not None:
            # 解析sample_data['au']结构
            au_values = None
            au_description = None
            
            if isinstance(sample_data['au'], dict):
                # MER-Factory格式：{'au_values': {...}, 'au_description': "..."}
                if 'au_values' in sample_data['au'] or 'active_aus' in sample_data['au']:
                    au_values = sample_data['au'].get('au_values') or sample_data['au'].get('active_aus')
                    au_description = sample_data['au'].get('au_description', None)
                else:
                    # 直接是AU值字典：{'AU01_r': 0.98, ...}
                    au_values = sample_data['au']
            elif hasattr(sample_data['au'], 'values'):
                au_values = sample_data['au']['values']
                au_description = sample_data['au'].get('description', None)
            else:
                # 回退到预提取特征模式
                au = sample_data['au'].unsqueeze(0).to(self.device)
                au_hiddens, au_llms = self.model.encode_au_merge(au, is_preextracted=True)
                return au_hiddens, au_llms
            
            # 使用AU Agent生成描述
            try:
                facial_description = self.au_agent.generate_description(
                    au_values=au_values,
                    au_description=au_description,  # 如果MER-Factory提供了则使用
                    temperature=0.7
                )
                
                # 将描述转换为text tokens（类似text模态处理）
                au_text = f"<au>\n{facial_description}\n</au>"
                au_llms = self.to_token_ids(au_text, max_length=512).to(self.device)
                
                # AU作为text处理，不需要特征向量
                return None, au_llms
                
            except Exception as e:
                print(f"⚠️ AU Agent生成失败: {e}，回退到预提取特征模式")
                # 回退
                if torch.is_tensor(sample_data['au']):
                    au = sample_data['au'].unsqueeze(0).to(self.device)
                    au_hiddens, au_llms = self.model.encode_au_merge(au, is_preextracted=True)
                    return au_hiddens, au_llms
                else:
                    return None, None
        else:
            # 预提取特征模式或CLIP实时编码模式
            # 检查AU数据类型
            if sample_data['au'] is None:
                print(f"⚠️ AU数据为None，跳过AU处理")
                return None, None
            
            if isinstance(sample_data['au'], dict):
                print(f"⚠️ AU数据是字典格式，但未启用AU Agent。请检查配置：")
                print(f"   1. 训练配置文件中是否正确设置了AU模式")
                print(f"   2. 推理时是否设置了use_au_clip_realtime=True")
                print(f"   跳过AU处理")
                return None, None
            
            if not torch.is_tensor(sample_data['au']):
                print(f"⚠️ AU数据类型错误: {type(sample_data['au'])}，应该是torch.Tensor")
                print(f"   跳过AU处理")
                return None, None
            
            # 正常处理张量AU特征
            au = sample_data['au'].unsqueeze(0).to(self.device)
            au_hiddens, au_llms = self.model.encode_au_merge(au, is_preextracted=True)
            return au_hiddens, au_llms

    
    # 整体过程就是在模拟inference过程 => 尝试完全按照 training 的方式进行读写
    def answer_sample(self, prompt, img_list, num_beams=1, temperature=1.0, do_sample=True,  top_p=0.9,
                    max_new_tokens=1000, min_length=1, max_length=2000, repetition_penalty=1.0, length_penalty=1.0):
        
        
        IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_IMAGE_PATCH_TOKEN]
        AUDIO_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_AUDIO_PATCH_TOKEN]
        FRAME_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_FRAME_PATCH_TOKEN]
        FACE_PATCH_TOKEN_ID  = self.tokenizer.get_vocab()[config.DEFAULT_FACE_PATCH_TOKEN]
        MULTI_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_MULTI_PATCH_TOKEN]
        # 🎯 Lexical信息直接作为文本嵌入，不需要PATCH_TOKEN

        ###### step1: => (input_id, attention_mask) 
        ## replace and add
        prompt = self.replace_token_for_multimodal(prompt)
        input_id = self.to_token_ids(prompt, max_length)
        print (prompt)
        
        ## length limits
        current_max_len = len(input_id) + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        input_id = input_id[begin_idx:]
        attention_mask=input_id.ne(self.tokenizer.pad_token_id).to(self.device)

        ###### step2: (input_ids) => (inputs_embeds)
        temp_input_id = copy.deepcopy(input_id).to(self.device)
        temp_input_id[temp_input_id == FRAME_PATCH_TOKEN_ID] = 0
        temp_input_id[temp_input_id == FACE_PATCH_TOKEN_ID]  = 0
        temp_input_id[temp_input_id == AUDIO_PATCH_TOKEN_ID] = 0
        temp_input_id[temp_input_id == MULTI_PATCH_TOKEN_ID] = 0
        temp_input_id[temp_input_id == IMAGE_PATCH_TOKEN_ID] = 0
        # 🎯 AU不再使用token占位符，改为caption文本直接嵌入prompt
        cur_input_embeds = self.model.llama_model.model.model.embed_tokens(temp_input_id)
        cur_input_ids = input_id
        
        # replace <ImageHere>, <AudioHere>, <FrameHere>, <FaceHere> with features
        # 🎯 AU不再作为特征模态，改为caption文本直接嵌入prompt
        cur_idx = 0
        for (patch_token_id, query_token_number, embeds) in [(FRAME_PATCH_TOKEN_ID, self.num_video_query_token, img_list['frame']),
                                                            (FACE_PATCH_TOKEN_ID,  self.num_video_query_token, img_list['face']),
                                                            (AUDIO_PATCH_TOKEN_ID, self.num_audio_query_token, img_list['audio']),
                                                            (MULTI_PATCH_TOKEN_ID, self.num_multi_query_token, img_list['multi']),
                                                            (IMAGE_PATCH_TOKEN_ID, self.num_image_query_token, img_list['image']),
                                                            ]:
            if (cur_input_ids == patch_token_id).sum() != 0:
                assert embeds is not None, f'Some input info is missing.'
                cur_features = embeds[cur_idx]
                if (cur_input_ids == patch_token_id).sum() != query_token_number:
                    raise ValueError("The number of audio patch tokens should be the same as the number of audio patches.")
                masked_indices = torch.where(cur_input_ids == patch_token_id)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start+query_token_number, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                cur_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], 
                                            cur_features, 
                                            cur_input_embeds[mask_index_start+query_token_number:]), dim=0)
                    
        cur_input_embeds = cur_input_embeds.unsqueeze(0) 
        attention_mask = attention_mask.unsqueeze(0) 
        ###### step3: (inputs_embeds, attention_masks) => response
        outputs = self.model.llama_model.generate(
            inputs_embeds=cur_input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )

        ###### step4: convert to batch samples 
        # maybe <bos> aaa <stop token> bbb <eos>
        response = self.tokenizer.decode(outputs[0], add_special_tokens=False)
        if response.find(self.tokenizer.bos_token) != -1:
            response = response.split(self.tokenizer.bos_token)[1]
        if response.find(self.tokenizer.eos_token) != -1:
            response = response.split(self.tokenizer.eos_token)[0]
        response = response.rsplit('###', 1)[0] # split from stop tokens '###'
        response = response.split('Assistant:')[-1].strip()
        return response
