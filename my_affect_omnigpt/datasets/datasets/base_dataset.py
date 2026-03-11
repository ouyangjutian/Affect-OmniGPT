import os
import tqdm
import copy
import random
import pandas as pd
from typing import Dict, Optional, Sequence, Iterable
import json

import torch
from torch.utils.data import Dataset, ConcatDataset
from my_affectgpt.models.tokenizer import load_tokenizer_from_LLM

import torch
from PIL import Image
import numpy as np

import transformers
from my_affectgpt.processors.video_processor import load_video, load_face
from my_affectgpt.models.ImageBind.data import load_audio, transform_audio
import config

class BaseDataset():
    def __init__(self, vis_processor=None, txt_processor=None, img_processor=None, model_cfg=None, dataset_cfg=None,
                vis_root=None, ann_path=None, wav_root=None, face_root=None, img_root=None):
        
        ####################################
        ## part1: common ones
        self.vis_root = vis_root
        self.img_root = img_root
        self.wav_root = wav_root
        self.ann_path = ann_path
        self.face_root = face_root
        self.vis_processor = vis_processor
        self.txt_processor = txt_processor
        self.img_processor = img_processor
        self.model_cfg = model_cfg
        self.dataset_cfg = dataset_cfg

        self.image_caption_prompt_candidates = ["Describe this image in detail.",
                                                "Take a look at this image and describe what you notice.",
                                                "Please provide a detailed description of the picture.",
                                                "Could you describe the contents of this image for me?"]

        self.audio_caption_prompt_candidates = ["Describe this audio in detail.",
                                                "Listen to this audio and describe what you hear.",
                                                "Please provide a detailed description of this audio.",
                                                "Could you describe the contents of this audio for me?"]

        ####################################
        ## part2: (model_cfg, dataset_cfg) specific ones
        if model_cfg is None or dataset_cfg is None: return
        
        self.max_length = model_cfg.max_length
        self.num_video_query_token = model_cfg.num_video_query_token
        self.num_audio_query_token = model_cfg.num_audio_query_token
        self.num_multi_query_token = model_cfg.num_multi_query_token
        self.num_image_query_token = model_cfg.num_image_query_token
        # 🎯 Nonverbal不再需要query token，直接作为文本嵌入prompt

        ## 控制视频采样的帧数
        self.n_frms = model_cfg.vis_processor.train.n_frms
        
        # Frame采样配置 - 可以通过dataset_cfg覆盖
        self.frame_n_frms = getattr(dataset_cfg, 'frame_n_frms', self.n_frms)  # Frame帧数，默认与n_frms相同
        self.frame_sampling = getattr(dataset_cfg, 'frame_sampling', 'uniform')  # Frame采样策略，默认uniform
        
        # MER-Factory输出路径 - 用于emotion_peak智能采样
        self.mer_factory_output = getattr(dataset_cfg, 'mer_factory_output', None)
        
        # 预提取特征配置 - 从dataset_cfg获取
        self.use_preextracted_features = getattr(dataset_cfg, 'use_preextracted_features', False)
        self.preextracted_root = getattr(dataset_cfg, 'preextracted_root', None)
        self.visual_encoder = getattr(dataset_cfg, 'visual_encoder', 'CLIP_VIT_LARGE')
        self.acoustic_encoder = getattr(dataset_cfg, 'acoustic_encoder', 'HUBERT_LARGE')
        self.clips_per_video = getattr(dataset_cfg, 'clips_per_video', 8)

        # 每个模态都需要一个明确的预提取标志；默认关闭（不再依赖全局 use_preextracted_features）
        def _resolve_preextract_flag(field_name: str) -> bool:
            if hasattr(dataset_cfg, field_name):
                return bool(getattr(dataset_cfg, field_name))
            return False

        self.use_preextracted_frame = _resolve_preextract_flag('use_preextracted_frame')
        self.use_preextracted_face = _resolve_preextract_flag('use_preextracted_face')
        self.use_preextracted_audio = _resolve_preextract_flag('use_preextracted_audio')
        
        # 🎯 Nonverbal配置 - 直接作为文本输入LLM（不需要CLIP编码）
        self.nonverbal_json = getattr(dataset_cfg, 'nonverbal_json', None)
        self.use_nonverbal_text = getattr(dataset_cfg, 'use_nonverbal_text', False)
        self._nonverbal_data = None  # 懒加载Nonverbal数据
        
        # 🎯 实时特征提取配置 - 从dataset_cfg获取
        self.use_realtime_extraction = getattr(dataset_cfg, 'use_realtime_extraction', False)
        self.extraction_server_host = getattr(dataset_cfg, 'extraction_server_host', 'localhost')
        self.extraction_server_port = getattr(dataset_cfg, 'extraction_server_port', 12345)
        self.feature_client = None
        
        # 初始化实时特征提取客户端
        if self.use_realtime_extraction:
            try:
                from simple_feature_client import SimpleFeatureClient
                self.feature_client = SimpleFeatureClient(
                    server_host=self.extraction_server_host,
                    server_port=self.extraction_server_port
                )
                if self.feature_client.connect():
                    print(f'[DATASET] 实时特征提取客户端已连接: {self.extraction_server_host}:{self.extraction_server_port}')
                else:
                    print(f'[DATASET] 实时特征提取客户端连接失败，将回退到实时处理模式')
                    self.feature_client = None
                    self.use_realtime_extraction = False
            except ImportError as e:
                print(f'[DATASET] 无法导入实时特征提取客户端: {e}')
                self.feature_client = None
                self.use_realtime_extraction = False
        
        print(f'====== Frame Sampling Config ======')
        print(f'Frame frames: {self.frame_n_frms}, Frame sampling: {self.frame_sampling}')
        print(f'Face frames: {self.n_frms}, Face sampling: uniform')
        
        if self.use_realtime_extraction:
            print(f'[DATASET] 实时特征提取模式已启用')
            print(f'Client status: {"Connected" if self.feature_client else "Failed"}')
        else:
            any_pre = any([
                self.use_preextracted_frame,
                self.use_preextracted_face,
                self.use_preextracted_audio,
            ])
            if any_pre:
                print(f'====== Preextracted Features Config (flags only, actual usage depends on face_or_frame) ======')
                print(f'Root: {self.preextracted_root}')
                print(f'Visual encoder: {self.visual_encoder}')
                print(f'Acoustic encoder: {self.acoustic_encoder}')
                print(f'Clips per video: {self.clips_per_video}')
                print(f'- Frame flag: {"ON" if self.use_preextracted_frame else "OFF"}')
                print(f'- Face flag:  {"ON" if self.use_preextracted_face else "OFF"}')
                print(f'- Audio flag: {"ON" if self.use_preextracted_audio else "OFF"}')
                print(f'- Nonverbal: Text Mode (embedded in prompt)')
            else:
                print(f'🔄 Real-time mode: ENABLED')

        # 这里token的设置和 affectgpt.py 中的一致 (所以这部分调用改成全局调用了)
        self.tokenizer = load_tokenizer_from_LLM(model_cfg.llama_model)
        self.IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_IMAGE_PATCH_TOKEN]
        self.AUDIO_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_AUDIO_PATCH_TOKEN]
        self.FRAME_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_FRAME_PATCH_TOKEN]
        self.FACE_PATCH_TOKEN_ID  = self.tokenizer.get_vocab()[config.DEFAULT_FACE_PATCH_TOKEN]
        self.MULTI_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_MULTI_PATCH_TOKEN]
        
        # 让模型只读取一定比例的文件
        if 'ratio' in dataset_cfg and dataset_cfg.ratio < 1:
            self.annotation = self.func_random_sample_subset(self.annotation, ratio=dataset_cfg.ratio)
            print(f'after sampled sample number: {len(self.annotation)}')

        ####################################
        ## part3: debug
        sample1 = self.__getitem__(random.randint(0, len(self)-1))
        sample2 = self.__getitem__(random.randint(0, len(self)-1))
        sample3 = self.__getitem__(random.randint(0, len(self)-1))
        self.func_visualize_samples(sample1)
        self.func_visualize_samples(sample2)
        self.func_visualize_samples(sample3)
        samples = [sample1, sample2, sample3]
        self.collater(samples)

        ## debug2: for all datasets (whether contains errors)
        # print ('Debug: whether all data are readable?')
        # for index in tqdm.tqdm(range(len(self))):
        #     sample = self.__getitem__(index)
        #     self.func_visualize_samples(sample)
        #     # print (sample['raw_audio'].shape)

        ## debug3: short version, only length
        print ('training sample number: ', len(self))
        ####################################

    def __len__(self):
        return len(self.annotation)
    
    def func_visualize_samples(self, sample):
        text_input = copy.deepcopy(sample['text_input'])
        input_convert = self.tokenizer.decode(text_input)
        print (input_convert)

        label = copy.deepcopy(sample['label'])
        label[label==config.IGNORE_INDEX] = self.tokenizer.bos_token_id
        output_convert = self.tokenizer.decode(label)
        print (output_convert)
    
    # to_token_ids: 开头不增加特殊符号，裁剪输入保证不超过 max_length
    def to_token_ids(self, text, max_length):
        input_ids = self.tokenizer(text, return_tensors="pt", padding="longest", max_length=max_length, 
                                truncation=True, add_special_tokens=False).input_ids[0]
        return input_ids
    
    def _load_nonverbal_data(self):
        """懒加载Nonverbal JSON数据"""
        if self._nonverbal_data is not None:
            return self._nonverbal_data
        
        if not self.nonverbal_json or not os.path.exists(self.nonverbal_json):
            if not hasattr(BaseDataset, '_warned_no_nonverbal_json'):
                print(f"⚠️ [Nonverbal] JSON文件不存在或未配置: {self.nonverbal_json}")
                BaseDataset._warned_no_nonverbal_json = True
            self._nonverbal_data = {}
            return self._nonverbal_data
        
        try:
            with open(self.nonverbal_json, 'r', encoding='utf-8') as f:
                self._nonverbal_data = json.load(f)
            print(f"✅ [Nonverbal] 加载成功: {self.nonverbal_json}")
            print(f"   包含数据集: {list(self._nonverbal_data.keys())}")
        except Exception as e:
            print(f"❌ [Nonverbal] 加载失败: {e}")
            self._nonverbal_data = {}
        
        return self._nonverbal_data
    
    def get_nonverbal_text(self, sample_name: str) -> Optional[str]:
        """
        从MER_UniBench_grained.json获取Nonverbal文本
        
        Args:
            sample_name: 样本名称
        
        Returns:
            Nonverbal文本，如果不存在则返回None
        """
        if not self.use_nonverbal_text:
            return None
        
        nonverbal_data = self._load_nonverbal_data()
        if not nonverbal_data:
            return None
        
        # 数据集名称映射
        dataset_name = getattr(self, 'dataset', '')
        dataset_name_mapping = {
            'MER2023': 'MER2023',
            'MER2024': 'MER2024',
            'MERCaptionPlus': 'MERCaptionPlus',
            'MELD': 'MELD',
            'IEMOCAPFour': 'IEMOCAP',
            'CMUMOSI': 'CMU-MOSI',
            'CMUMOSEI': 'CMU-MOSEI',
            'SIMS': 'CH-SIMS',
            'SIMSv2': 'CH-SIMS v2',
        }
        json_dataset_name = dataset_name_mapping.get(dataset_name, dataset_name)
        
        # 尝试多种数据集名称匹配
        for try_name in [json_dataset_name, dataset_name, dataset_name.upper(), dataset_name.lower()]:
            if try_name in nonverbal_data and sample_name in nonverbal_data[try_name]:
                caption = nonverbal_data[try_name][sample_name].get('caption', None)
                if caption:
                    return caption
        
        return None
    
    # 🎯 旧的AU CLIP编码相关函数已移除
    # Nonverbal信息现在直接作为文本嵌入prompt，参见 get_nonverbal_text()

    def func_map_valence_to_emotion(self, valence):
        if valence > 0:
            return 'positive'
        elif valence < 0:
            return 'negative'
        else:
            return 'neutral'
        

    def get_cur_label_type(self, label_type_candidates, label_type):
        if label_type == 'hybird':
            index = random.randint(0, len(label_type_candidates) -1)
            return label_type_candidates[index]
        else:
            assert label_type in label_type_candidates, f'error label type: {label_type} not in {label_type_candidates}'
            return label_type
        
    
    def func_random_prompts(self, candidates):
        index = random.randint(0, len(candidates) - 1)
        prompt = candidates[index]
        return prompt
    
    
    # 随机采样一个 annotations
    def func_random_sample_subset(self, annotations, ratio=0.1):
        annotations_subset = random.sample(annotations, int(len(annotations)*ratio))
        return annotations_subset


    ###########################################################
    ## 数据读取部分操作
    ###########################################################
    # all types: {audio, frame, face, image}
    def get_needed_data(self, face_or_frame):
        if face_or_frame == 'faceframe': # (face, frame, audio, text)
            needed_data = ['audio', 'frame', 'face']
        elif face_or_frame == 'face': # (face, audio, text)
            needed_data = ['audio', 'face']
        elif face_or_frame == 'frame': # (frame, audio, text)
            needed_data = ['audio', 'frame']
        elif face_or_frame == 'audioonly': # (audio)
            needed_data = ['audio']
        elif face_or_frame == 'textonly':  # (text)
            needed_data = []
        elif face_or_frame == 'auonly':  # (au_caption only)
            # 🎯 只有AU caption文本，没有其他模态
            needed_data = []
        elif face_or_frame == 'faceonly':  # (face)
            needed_data = ['face']
        elif face_or_frame == 'frameonly': # (frame)
            needed_data = ['frame']
        elif face_or_frame == 'multiface_text': # (multi, text)
            needed_data = ['face', 'audio', 'multi']
        elif face_or_frame == 'multiface_audio_face_text': # (multi, face, audio, text)
            needed_data = ['face', 'audio', 'multi']
        elif face_or_frame == 'image': # (image)
            needed_data = ['image']
        elif face_or_frame == 'multiframe_audio_frame_text': # (multi, face, audio, text)
            needed_data = ['frame', 'audio', 'multi']
        elif face_or_frame == 'multiframe_audio_frame_au_text': # (multi, frame, audio, au_caption_text)
            # 🎯 AU不再作为独立模态，改为caption文本直接嵌入prompt
            needed_data = ['frame', 'audio', 'multi']
        elif face_or_frame == 'multiface_audio_face_frame_text': # (multi, face, audio, text)
            needed_data = ['frame', 'face', 'audio', 'multi']
        elif face_or_frame == 'multiface_audio_face_frame_au_text': # (multi, frame, face, audio, au_caption_text)
            # 🎯 AU不再作为独立模态，改为caption文本直接嵌入prompt
            needed_data = ['frame', 'face', 'audio', 'multi']
        elif face_or_frame == 'multiface_audio_face_au_text': # (multi, face, audio, au_caption_text)
            # 🎯 AU不再作为独立模态，改为caption文本直接嵌入prompt
            needed_data = ['face', 'audio', 'multi']        
        elif face_or_frame == 'audio_text': # (audio, text)
            needed_data = ['audio']
        elif face_or_frame == 'audio_au_text': # (audio, au_caption, text)
            # 🎯 AU不再作为独立模态，改为caption文本直接嵌入prompt
            needed_data = ['audio']
        elif face_or_frame == 'face_text': # (face, text)
            needed_data = ['face']
        elif face_or_frame == 'frame_text': # (frame, text)
            needed_data = ['frame']
        elif face_or_frame == 'face_frame_au_text': # (face, frame, au_caption_text) - 无audio无multi
            # 🎯 AU不再作为独立模态，改为caption文本直接嵌入prompt
            needed_data = ['face', 'frame']
        return needed_data
    

    def read_frame_face_audio_text(self, video_path=None, face_npy=None, audio_path=None, image_path=None, sample_name=None):

        sample_data = {}

        # 🎯 预提取特征配置：支持独立模态标志 + 全局标志（向后兼容）
        # 优先级：独立标志 > 全局标志
        # global_preextracted = getattr(self, 'use_preextracted_features', False)  # 全局标志（训练时用）
        use_preextracted_frame = getattr(self, 'use_preextracted_frame')
        use_preextracted_face = getattr(self, 'use_preextracted_face')
        use_preextracted_audio = getattr(self, 'use_preextracted_audio')
        # 🎯 AU不再使用预提取特征，改为caption文本直接嵌入prompt
        
        preextracted_root = getattr(self, 'preextracted_root', None)
        
        # 数据集名称映射（处理特殊情况，所有模态共用）
        dataset_name_mapping = {
            'IEMOCAPFour': 'iemocap',  # IEMOCAPFour -> iemocap（与提取脚本保持一致）
        }
        dataset_name = getattr(self, 'dataset', 'unknown')
        dataset_name_lower = dataset_name_mapping.get(dataset_name, dataset_name.lower())

        def _resolve_preextracted_root(root: Optional[str]) -> Optional[str]:
            """确保返回的路径只包含一次数据集子目录，兼容用户传入已包含dataset名的root。"""
            if not root:
                return None
            normalized = os.path.normpath(root)
            if os.path.basename(normalized) == dataset_name_lower:
                return normalized
            return os.path.join(root, dataset_name_lower)

        preextracted_dataset_root = _resolve_preextracted_root(preextracted_root)

        # step1: read (raw_frame, frame) - 可配置的Frame采样策略
        frame, raw_frame = None, None
        if 'frame' in self.needed_data:
            # 🎯 新增：检查是否使用实时特征提取服务
            if hasattr(self, 'use_realtime_extraction') and self.use_realtime_extraction:
                # 实时特征提取模式 - 通过服务提取特征（保持数据增强）
                if hasattr(self, 'feature_client') and self.feature_client:
                    realtime_features = self.feature_client.extract_features(
                        sample_name=sample_name,
                        modalities=['frame'],
                        video_path=video_path,
                        n_frms=getattr(self, 'frame_n_frms', self.n_frms),
                        frame_sampling=getattr(self, 'frame_sampling', 'uniform')
                    )
                    if realtime_features and 'frame' in realtime_features:
                        frame_features = realtime_features['frame']  # [T, D] - 编码器输出特征
                        frame = torch.from_numpy(frame_features).float()
                        raw_frame = frame  # 分布式模式下使用相同数据
                        sample_data['frame_preextracted'] = True  # 标记为已提取特征（编码器输出）
                        pass  # 特征提取成功
                    else:
                        print(f"⚠️ 实时Frame特征提取失败: {sample_name}")
            elif use_preextracted_frame and preextracted_dataset_root and sample_name:
                # 预提取特征模式 - 直接加载.npy特征文件
                frame_n_frms = getattr(self, 'frame_n_frms', 1)
                frame_sampling = getattr(self, 'frame_sampling', 'uniform')
                visual_encoder = getattr(self, 'visual_encoder', 'CLIP_VIT_LARGE')
                
                frame_feat_dir = f'frame_{visual_encoder}_{frame_sampling}_{frame_n_frms}frms'
                frame_feat_path = os.path.join(preextracted_dataset_root, frame_feat_dir, f'{sample_name}.npy')
                
                if os.path.exists(frame_feat_path):
                    frame_features = np.load(frame_feat_path)  # [T, D]
                    frame = torch.from_numpy(frame_features).float()  # 转换为tensor
                    raw_frame = frame  # 预提取模式下raw_frame与frame相同
                    sample_data['frame_preextracted'] = True  # 标记为预提取特征
                    
                    # 首次加载时输出提示
                    if not hasattr(BaseDataset, '_logged_frame_preextract_success'):
                        print(f"✅ [Frame预提取] 成功加载预提取特征: {dataset_name_lower}/frame_{visual_encoder}_{frame_sampling}_{frame_n_frms}frms/")
                        BaseDataset._logged_frame_preextract_success = True
                else:
                    # 预提取特征文件不存在，回退到实时处理模式
                    if not hasattr(BaseDataset, '_warned_missing_preextract'):
                        print(f"⚠️ Frame预提取特征不存在: {frame_feat_path}")
                        print(f"   将回退到实时处理模式")
                        BaseDataset._warned_missing_preextract = True
                    
                    # 回退：实时加载视频
                    if video_path is not None:
                        frame_n_frms = getattr(self, 'frame_n_frms', self.n_frms)
                        frame_sampling = getattr(self, 'frame_sampling', 'uniform')
                        mer_factory_output = getattr(self, 'mer_factory_output', None)
                        
                        video_name = sample_name if sample_name else (os.path.splitext(os.path.basename(video_path))[0] if video_path else None)
                        
                        raw_frame, msg = load_video(
                            video_path=video_path,
                            n_frms=frame_n_frms,
                            height=224,
                            width=224,
                            sampling=frame_sampling,
                            return_msg=True,
                            video_name=video_name,
                            mer_factory_output=mer_factory_output
                        )
                        frame = self.vis_processor.transform(raw_frame)
                        sample_data['frame_preextracted'] = False  # ✅ 显式设置为False
            else:
                # 实时处理模式 - 原有逻辑
                if video_path is not None:
                    frame_n_frms = getattr(self, 'frame_n_frms', self.n_frms)  # 默认使用n_frms
                    frame_sampling = getattr(self, 'frame_sampling', 'uniform')  # 默认使用uniform采样
                    mer_factory_output = getattr(self, 'mer_factory_output', None)  # MER-Factory输出路径
                    
                    # 提取video_name（不含扩展名）
                    video_name = None
                    if sample_name:
                        video_name = sample_name
                    elif video_path:
                        video_name = os.path.splitext(os.path.basename(video_path))[0]
                    
                    raw_frame, msg = load_video(
                        video_path=video_path,
                        n_frms=frame_n_frms,
                        height=224,
                        width=224,
                        sampling=frame_sampling,
                        return_msg=True,
                        video_name=video_name,  # 传递video_name用于智能采样
                        mer_factory_output=mer_factory_output  # 传递MER-Factory路径
                    )
                    frame = self.vis_processor.transform(raw_frame) # [3, frame_n_frms, 224, 224]
                    sample_data['frame_preextracted'] = False  # ✅ 显式设置为False
        # 只有当frame特征有效时才添加到样本中
        if frame is not None:
            sample_data['frame'] = frame
            sample_data['raw_frame'] = raw_frame
        else:
            # Frame特征无效，如果需要Frame模态则跳过此样本
            if 'frame' in self.needed_data:
                print(f"⚠️ Frame特征无效，跳过样本: {sample_name}")
                return None  # 返回None表示此样本无效，需要重新选择
            # 确保frame相关的标志也不设置
            if 'frame_preextracted' in sample_data:
                del sample_data['frame_preextracted']

        # step2: read (raw_face, face)
        face, raw_face = None, None
        if 'face' in self.needed_data:
            # 检查是否使用预提取特征
            if use_preextracted_face and preextracted_dataset_root and sample_name:
                # 预提取特征模式
                visual_encoder = getattr(self, 'visual_encoder', 'CLIP_VIT_LARGE')
                n_frms = getattr(self, 'n_frms', 8)
                face_feat_dir = f'face_{visual_encoder}_{n_frms}frms'
                face_feat_path = os.path.join(preextracted_dataset_root, face_feat_dir, f'{sample_name}.npy')
                
                if os.path.exists(face_feat_path):
                    face_features = np.load(face_feat_path)
                    face = torch.from_numpy(face_features).float()
                    raw_face = face
                    sample_data['face_preextracted'] = True
                else:
                    # 预提取文件不存在，fallback到实时处理
                    if face_npy is not None:
                        raw_face, msg = load_face(face_npy=face_npy, n_frms=self.n_frms, height=224, width=224, sampling="uniform", return_msg=True)
                        face = self.vis_processor.transform(raw_face)
                        sample_data['face_preextracted'] = False
            else:
                # 实时处理模式
                if face_npy is not None:
                    raw_face, msg = load_face(face_npy=face_npy, n_frms=self.n_frms, height=224, width=224, sampling="uniform", return_msg=True)
                    face = self.vis_processor.transform(raw_face)
                    sample_data['face_preextracted'] = False
        # 只有当face特征有效时才添加到样本中
        if face is not None:
            sample_data['face'] = face
            sample_data['raw_face'] = raw_face
        elif 'face' in self.needed_data:
            # Face特征无效，但需要Face模态时才打印警告
            print(f"⚠️ Face特征无效，跳过Face模态: {sample_name}")
            # 确保face相关的标志也不设置
            if 'face_preextracted' in sample_data:
                del sample_data['face_preextracted']

        # step3: read audio [需要针对没有 audio track 的 video 进行额外处理]
        audio, raw_audio = None, None
        if 'audio' in self.needed_data:
            # 检查是否使用预提取特征
            if use_preextracted_audio and preextracted_dataset_root and sample_name:
                # 预提取特征模式 - 尝试加载.npy特征文件
                acoustic_encoder = getattr(self, 'acoustic_encoder', 'HUBERT_LARGE')
                clips_per_video = getattr(self, 'clips_per_video', 8)
                audio_feat_dir = f'audio_{acoustic_encoder}_{clips_per_video}clips'
                audio_feat_path = os.path.join(preextracted_dataset_root, audio_feat_dir, f'{sample_name}.npy')
                
                if os.path.exists(audio_feat_path):
                    audio_features = np.load(audio_feat_path)  # [T, D]
                    audio = torch.from_numpy(audio_features).float()
                    raw_audio = audio
                    sample_data['audio_preextracted'] = True  # ✅ 标记为预提取
                else:
                    # 预提取文件不存在，fallback到实时处理
                    if audio_path is not None:
                        raw_audio = load_audio([audio_path], "cpu", clips_per_video=8)[0]
                        audio = transform_audio(raw_audio, "cpu")
                        sample_data['audio_preextracted'] = False  # ✅ 显式设置为False
            else:
                # 实时处理模式
                if audio_path is not None:
                    raw_audio = load_audio([audio_path], "cpu", clips_per_video=8)[0]
                    audio = transform_audio(raw_audio, "cpu")
                    sample_data['audio_preextracted'] = False  # ✅ 显式设置为False
        # 只有当audio特征有效时才添加到样本中
        if audio is not None:
            sample_data['audio'] = audio
            sample_data['raw_audio'] = raw_audio
        elif 'audio' in self.needed_data:
            # Audio特征无效，但需要Audio模态时才打印警告
            print(f"⚠️ Audio特征无效，跳过Audio模态: {sample_name}")
        
        # step4: read multi features (Face+Audio融合特征)
        # Multi特征总是在模型中实时融合，不使用预提取
        multi, raw_multi = None, None
        sample_data['multi'] = multi
        sample_data['raw_multi'] = raw_multi
        
        # 🎯 AU不再作为特征模态加载，改为caption文本直接嵌入prompt
        # AU caption通过get_au_caption_text()方法在生成prompt时获取
        
        # 注意：*_preextracted标志已在各模态加载时设置，不需要在这里重复设置
        
        # step4: read image
        image, raw_image = None, None
        if image_path is not None and 'image' in self.needed_data:
            ###### 支持两种类型的 image_path 输入 ######
            if not isinstance(image_path, Image.Image):
                raw_image = Image.open(image_path)
            else:
                raw_image = image_path
            ##########################################
            ## image process
            image = self.img_processor(raw_image.convert("RGB")) # [3, 224, 224] 这是 vis processor 默认下的处理，正常情况其实也不需要这个内容
            image = image.unsqueeze(dim=1) # [3, 1, 224, 224]
            raw_image = torch.from_numpy(np.array(raw_image.resize((224, 224)))) # [H, W, C] => 可能因为llava中的图片有些并不是一样大小的，使得转换过程中有些
            raw_image = raw_image.permute(2, 0, 1).unsqueeze(dim=1).float() # (C, T=1, H, W)
        sample_data['image'] = image
        sample_data['raw_image'] = raw_image
        # print (sample_data)

        return sample_data


    ###########################################################
    ## QA 获取
    ###########################################################
    ## 建立一个 qa 读取器，用于后续统一化的处理
    def func_get_qa_description(self, sample, question_only=False):
        question = "Please infer the person's emotional state and provide your reasoning process."

        if question_only:
            return question
        else:
            return {
                'question': question, 
                'answer':sample['description'],
                }
    
    def func_get_qa_ovlabel(self, sample, question_only=False):
        question = "Please recognize all possible emotional states of the character."

        if question_only:
            return question
        else:
            return {
                'question': question,
                'answer':  f"The character's emotional state is {sample['ovlabel']}."
                }
    
    def func_get_qa_onehot_w_candidates(self, sample, question_only=False):
        question = f"Please select the label that can best describe the person's emotional state from the provided candidate labels: {self.candidate_labels}."

        if question_only:
            return question
        else:
            return {
                'question': question,
                'answer':   f"The most likely label is {sample['onehot']}."
                }

    def func_get_qa_onehot_wo_candidates(self, sample, question_only=False):
        question = "Please recognize the character's most likely emotional state."

        if question_only:
            return question
        else:
            return {
                'question': question,
                'answer':  f"The character's emotional state is {sample['onehot']}."
                }

    def func_get_qa_valence(self, sample, question_only=False):
        question = f"Please identify the overall positive or negative emotional polarity of the main characters. " \
                 + f"The output should be a ﬂoating-point number ranging from {self.minval} to {self.maxval}. " \
                 + f"Here, {self.minval} indicates extremely negative emotions, 0 indicates neutral emotions, and {self.maxval} indicates extremely positive emotions. " \
                 + f"Please provide your judgment as a ﬂoating-point number."
        
        if question_only:
            return question
        else:
            return {
                'question': question,
                'answer': 'The valence score is %.2f.' %(sample['valence']),
                }

    def func_get_qa_sentiment(self, sample, question_only=False):
        question = "Please select the most likely sentiment label that can best describe the person's emotional state: positive, negative, neutral."
        
        if question_only:
            return question
        else:
            return {
                'question': question,
                'answer':  f"The character's sentiment state is {sample['sentiment']}.",
                }

    def func_get_qa_direct(self, sample):
        return {
            'question': sample['question'],
            'answer':   sample['answer'],
            }
    
    def func_get_qa_caption(self, sample, modality):
        if modality == 'image':
            return {
            'question': self.func_random_prompts(self.image_caption_prompt_candidates),
            'answer':   sample['caption'],
            }
        elif modality == 'audio':
            return {
            'question': self.func_random_prompts(self.audio_caption_prompt_candidates),
            'answer':   sample['caption'],
            }
    
    def func_get_qa_preference(self, sample):

        a1 = sample['preference']['a1']
        a2 = sample['preference']['a2']
        p  = sample['preference']['p']

        question = f"We provide two descriptions. a1: {a1} \t\t\t a2: {a2} Please select the one that best matches the video content."
        
        assert p in ['a1', 'a2', 'same']
        if p in ['a1', 'a2']:
            answer = f"The best one is {p}."
        else:
            answer = f'These two sentences describe the content of the video with the same accuracy.'

        return {
            'question': question,
            'answer':   answer,
            }

    # this (q, a) is used to determinate the reward value
    def func_get_description_reward(self, sample):
        reason = sample['description']
        reward = sample['reward']

        question = f"We have provided a description: {reason} \t\t\t Please evaluate and decide whether to accept or reject this description based on its alignment with the video content."

        assert reward in ['accept', 'reject']
        answer = f'{reward} this sentence.'

        return {
            'question': question,
            'answer':   answer,
        }

    ## 获取 <question, answer> 用于后续训练
    def get_qa_pairs(self, dataset, label_type, sample):
        
        '''
        self.  -> 数据集全局的内容
        sample -> 样本局部的内容
        '''
        # EMERFine 指的是 (training set) 那 332 samples，同时包含 ovlabel/description
        if dataset in ['EMERCoarse', 'EMERFine']:
            candidates = {
                'description': self.func_get_qa_description(sample),
                'ovlabel':     self.func_get_qa_ovlabel(sample),
            }
        
        elif dataset in ['EMERCoarseFilter']:
            candidates = {
                'description': self.func_get_qa_description(sample),
                'ovlabel':     self.func_get_qa_ovlabel(sample),
                'sentiment':   self.func_get_qa_sentiment(sample),
                'valence':     self.func_get_qa_valence(sample),
            }
        
        elif dataset in ['MERCaptionPlus', 'OVMERD']:
            candidates = {
                'description': self.func_get_qa_description(sample),
                'ovlabel':     self.func_get_qa_ovlabel(sample),
            }
        
        elif dataset in ['Preference']: # 带 preference 优化
            candidates = {
                'description': self.func_get_qa_description(sample),
                'ovlabel':     self.func_get_qa_ovlabel(sample),
                'sentiment':   self.func_get_qa_sentiment(sample),
                'valence':     self.func_get_qa_valence(sample),
                'preference':  self.func_get_qa_preference(sample),
            }

        elif dataset in ['Preference2', 'Preference4']: # 不带 preference 优化
            candidates = {
                'description': self.func_get_qa_description(sample),
                'ovlabel':     self.func_get_qa_ovlabel(sample),
                'sentiment':   self.func_get_qa_sentiment(sample),
                'valence':     self.func_get_qa_valence(sample),
            }
        
        elif dataset in ['Preference3']: # 不带 preference 优化
            candidates = {
                'reward': self.func_get_description_reward(sample),
            }
        
        ## case1: Zebang's labels
        elif dataset in ['MERRCoarse', 'MERRFine', 'MAFW']:
            candidates = {
                'description': self.func_get_qa_description(sample),
            }

        ## case2: onehot labels
        elif dataset in ['MER2023', 'MER2024', 'MELD', 'IEMOCAPFour']:
            candidates = {
                'onehot_w_candidates':  self.func_get_qa_onehot_w_candidates(sample),
                'onehot_wo_candidates': self.func_get_qa_onehot_wo_candidates(sample),
            }

        ## case3: valence scores
        elif dataset in ['CMUMOSI', 'CMUMOSEI', 'SIMS', 'SIMSv2']:
            candidates = {
                'valence':   self.func_get_qa_valence(sample),
                'sentiment': self.func_get_qa_sentiment(sample),
            }

        ## case4: instruction dataset
        elif dataset in ['VideoChat', 'LLaVA', 'EmoVIT']:
            candidates = {
                'qa':  self.func_get_qa_direct(sample),
            }

        elif dataset in ['MiniGPT4']:
            candidates = {
                'caption': self.func_get_qa_caption(sample, 'image'),
            }

        elif dataset in ['WavCaps', 'TextrolSpeech', 'PromptSpeech']:
            candidates = {
                'caption': self.func_get_qa_caption(sample, 'audio'),
            }

        return candidates[label_type] # 包含 question, answer 两部分内容


    def get_prompt_for_multimodal(self, face_or_frame, subtitle, user_message, nonverbal_text=None):
        """
        生成多模态prompt
        
        Args:
            face_or_frame: 模态组合类型
            subtitle: 字幕文本
            user_message: 用户问题
            nonverbal_text: Nonverbal文本（直接嵌入prompt，不需要CLIP编码）
        """
        # step5: get prompts for differet cases [可能存在三种数据加载情况，从而能够扩展至4种模态输入]
        if face_or_frame == 'faceframe': # (face, frame, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"Additionally, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'face': # (face, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'frame': # (frame, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'audioonly': # (audio)
            prompt = f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'textonly':  # (text)
            assert subtitle is not None
            prompt = f"###Human: The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'auonly':  # (au_caption only)
            # 🎯 只有AU caption文本，没有其他模态
            nonverbal_prompt = f"The nonverbal clues (facial action units and audio emotion clues) are: <Nonverbal>{nonverbal_text}</Nonverbal>. " if nonverbal_text else ""
            prompt = f"###Human: {nonverbal_prompt}" \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'faceonly':  # (face)
            prompt = f"###Human: We uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'frameonly': # (frame)
            prompt = f"###Human: We uniformly sample raw frames from the video: <Video><FrameHere></Video>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'image': # (image)
            prompt = f"###Human: The image content is as follows: <Image><ImageHere></Image>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        
        ## 这部分是为了和其他 MLLM 进行公平比较，所进行的 ablation study 部分
        elif face_or_frame == 'audio_text': # (audio, text)
            assert subtitle is not None
            prompt =  f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'audio_au_text': # (audio, au_caption, text)
            assert subtitle is not None
            # 🎯 Nonverbal信息（AU + Audio emotion）直接作为文本嵌入prompt
            nonverbal_prompt = f"The nonverbal clues (facial action units and audio emotion clues) are: <Nonverbal>{nonverbal_text}</Nonverbal>. " if nonverbal_text else ""
            prompt =  f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + nonverbal_prompt \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'face_text': # (audio, text)
            assert subtitle is not None
            prompt =  f"We uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'frame_text': # (audio, text)
            assert subtitle is not None
            prompt =  f"we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'face_frame_au_text': # (face, frame, nonverbal, text) - 无audio无multi
            assert subtitle is not None
            # 🎯 Nonverbal信息（AU caption）直接作为文本嵌入prompt
            nonverbal_prompt = f"The nonverbal clues (facial action units) are: <Nonverbal>{nonverbal_text}</Nonverbal>. " if nonverbal_text else ""
            prompt = f"###Human: We uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"Additionally, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + nonverbal_prompt \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
            
        ## 后面都是增加 <Multi> token 后的结果    
        elif face_or_frame == 'multiface_text': # (multi, text)
            assert subtitle is not None
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'multiface_audio_face_text': # (multi, face, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'multiframe_audio_frame_text': # (multi, face, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'multiframe_audio_frame_au_text': # (multi, frame, audio, nonverbal, text) - 无Face
            assert subtitle is not None
            # 🎯 Nonverbal信息（AU + Audio emotion）直接作为文本嵌入prompt，格式与Subtitle一致
            nonverbal_prompt = f"The nonverbal clues (facial action units and audio emotion clues) are: <Nonverbal>{nonverbal_text}</Nonverbal>. " if nonverbal_text else ""
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + nonverbal_prompt \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'multiface_audio_face_frame_text': # (multi, frame, face, audio, text)
            assert subtitle is not None
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'multiface_audio_face_frame_au_text': # (multi, frame, face, audio, nonverbal, text)
            assert subtitle is not None
            # 🎯 Nonverbal信息（AU + Audio emotion）直接作为文本嵌入prompt，格式与Subtitle一致
            nonverbal_prompt = f"The nonverbal clues (facial action units and audio emotion clues) are: <Nonverbal>{nonverbal_text}</Nonverbal>. " if nonverbal_text else ""
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + nonverbal_prompt \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        elif face_or_frame == 'multiface_audio_face_au_text': # (multi, face, audio, nonverbal, text) - 无Frame
            assert subtitle is not None
            # 🎯 Nonverbal信息（AU + Audio emotion）直接作为文本嵌入prompt，格式与Subtitle一致
            nonverbal_prompt = f"The nonverbal clues (facial action units and audio emotion clues) are: <Nonverbal>{nonverbal_text}</Nonverbal>. " if nonverbal_text else ""
            prompt = f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. " \
                    + f"The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + nonverbal_prompt \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: "
        return prompt
    
    ## 替换 <FaceHere> / <FrameHere> / <AudioHere> / <ImageHere> / <MultiToken>
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
        # 🎯 Nonverbal信息直接作为文本嵌入，不需要token替换
        return prompt


    ####################################################################################
    ## 读取一个样本 (read one sample)
    ####################################################################################
    def __getitem__(self, index):
        num_retries = 10 # skip error or too long videos
        for _ in range(num_retries):
            try:
                sample = self.annotation[index]
                cur_label_type = self.get_cur_label_type(self.label_type_candidates, self.label_type)
                # print ('cur_label_type: ', cur_label_type)

                # step1: read needed data
                video_path, image_path, audio_path, face_npy = None, None, None, None
                if hasattr(self, '_get_video_path'): video_path = self._get_video_path(sample)
                if hasattr(self, '_get_image_path'): image_path = self._get_image_path(sample)
                if hasattr(self, '_get_audio_path'): audio_path = self._get_audio_path(sample)
                if hasattr(self, '_get_face_path'):  face_npy   = self._get_face_path(sample)
                # print (video_path, image_path, audio_path, face_npy)
                sample_name = sample.get('name', None)  # 获取样本名称用于预提取特征
                sample_data = self.read_frame_face_audio_text(video_path, face_npy, audio_path, image_path, sample_name)

                # step2: read (question, answer)
                # => 如果 sample 中缺少 qa 对应内容的信息，结果是会报错的
                qa_pair = self.get_qa_pairs(self.dataset, cur_label_type, sample)
                # print (qa_pair)

                # step4: generate (text_input, label)
                if 'subtitle' not in sample: sample['subtitle'] = None
                
                # 🎯 获取Nonverbal文本（如果启用）
                nonverbal_text = None
                if self.use_nonverbal_text and sample_name:
                    nonverbal_text = self.get_nonverbal_text(sample_name)
                
                prompt = self.get_prompt_for_multimodal(self.face_or_frame, sample['subtitle'], qa_pair['question'], nonverbal_text=nonverbal_text) # get prompt
                prompt = self.replace_token_for_multimodal(prompt) # replace specific tokens
                # print (prompt)

                ## tokenizer [每部分内容不能超过 self.max_length, 且两部分内容的和也不能超过 self.max_length]
                prompt_id = self.to_token_ids(prompt, self.max_length) # => 避免 GPU OOM
                
                target = qa_pair['answer'] + '###'
                # print (target)
                target_id = self.to_token_ids(target, self.max_length)

                text_input = torch.cat([prompt_id, target_id])
                label = torch.cat([torch.ones([len(prompt_id)], dtype=text_input.dtype) * -100, target_id])
                assert len(text_input) == len(label)
                if len(text_input) > self.max_length:
                    raise RuntimeError("too long text_input")
            except Exception as error:
                print(f'Error: {error}')
                print(f"Failed to load data {self.dataset} {sample['name']}. We will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # 构建返回字典，只包含有效的模态
        result = {}
        
        # Face模态
        if 'face' in sample_data:
            result["face"] = sample_data['face']           # [c=3, frame=8, 224, 224] [这个经过了transformer变换]
            result["raw_face"] = sample_data['raw_face']   # [c=3, frame=8, 224, 224]

        # Frame模态
        if 'frame' in sample_data:
            result["frame"] = sample_data['frame']         # [c=3, frame=8, 224, 224] [这个经过了transformer变换]
            result["raw_frame"] = sample_data['raw_frame'] # [c=3, frame=8, 224, 224]

        # Audio模态
        if 'audio' in sample_data:
            result["audio"] = sample_data['audio']          # [frame=8, c=1, 128, 204]
            result["raw_audio"] = sample_data['raw_audio']  # [frame=8, c=1, 16000*2采样点]

        # Image模态
        if 'image' in sample_data:
            result["image"] = sample_data['image']
            result["raw_image"] = sample_data['raw_image']
            
        # Multi模态
        result["multi"] = sample_data.get('multi', None)
        result["raw_multi"] = sample_data.get('raw_multi', None)
        
        # 🎯 AU/Nonverbal不再作为特征模态，直接作为文本嵌入prompt
        
        # 其他必要字段
        result["label"] = label
        result["text_input"] = text_input
        result['dataset'] = self.dataset.lower()
        result['face_or_frame'] = self.face_or_frame
        
        # 传递预提取标志
        result['frame_preextracted'] = sample_data.get('frame_preextracted', False)
        result['face_preextracted'] = sample_data.get('face_preextracted', False)
        result['audio_preextracted'] = sample_data.get('audio_preextracted', False)
        result['multi_preextracted'] = sample_data.get('multi_preextracted', False)
        
        return result

        
    ####################################################################################
    ## batch 级别数据合并
    ####################################################################################
    def collater(self, instances):
        '''
        llama token ids:
            <unk>: 0
            bos|<s>: 1
            eos|pad|</s>: 2
            <ImageHere>: 32000
            <AudioHere>: 32001

        data_dict:  input_ids:[###Human: <Image> <ImageHere>*32 /<Image> xxx  ...   ###Assistant: xxx###Human: xxx###Assistant: xxx###]
                    labels:   [-100..., -100, ....,                                 ...           xxx###-100...,        ...     xxx###]

        data_dict:  input_ids:[<bos>###Human: <Image> <ImageHere>*32 /<Image> xxx  ...   ###Assistant: xxx###Human: xxx###Assistant: xxx###, <eos>,    ...]
                    labels:   [-100..., -100, ....,                                 ...                xxx###-100...,        ...     xxx###, -100, ...]
                    images:   [bs=3, c=3, 224, 224]
        '''
        labels = []
        input_ids = []
        for instance in instances:
            label = instance['label']
            input_id = instance['text_input']
            label    = torch.cat([torch.ones([1], dtype=input_id.dtype) * config.IGNORE_INDEX, label,
                                  torch.ones([1], dtype=input_id.dtype) * self.tokenizer.eos_token_id]) # (-100  xxx <eos>)
            input_id = torch.cat([torch.ones([1], dtype=input_id.dtype) * self.tokenizer.bos_token_id, input_id,
                                  torch.ones([1], dtype=input_id.dtype) * self.tokenizer.eos_token_id]) # (<bos> xxx <eos>)
            labels.append(label)
            input_ids.append(input_id)

        # pad bacth input into the same length 
        # => input_ids: <bos> xxx <eos> <pad>
        # => label    : -100  xxx <eos> -100
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, 
                                                    batch_first=True, 
                                                    padding_value=self.tokenizer.pad_token_id)
        labels    = torch.nn.utils.rnn.pad_sequence(labels,    
                                                    batch_first=True, 
                                                    padding_value=config.IGNORE_INDEX)
        batch = dict(
            labels=labels,
            input_ids=input_ids,
            attention_masks=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # 后面跟着的是 dataset 中所有数据类型
        # => 只有符合约束，才把这部分数据存储在 batch 里面，如果有问题，直接就不存储
        for sample_type in ['face', 'raw_face', 'frame', 'raw_frame', 'audio', 'raw_audio', 'image', 'raw_image', 'multi', 'raw_multi']:
            batch_type = sample_type + 's'

            if sample_type in instances[0]:
                datas = [instance[sample_type] for instance in instances]
                first = datas[0]
                # 仅对张量/ndarray做shape检查与stack，避免dict等无shape对象报错
                if isinstance(first, (torch.Tensor, np.ndarray)):
                    if all(x is not None and isinstance(x, (torch.Tensor, np.ndarray)) and (np.shape(x) == np.shape(first)) for x in datas):
                        # 将ndarray转换为Tensor，再进行stack
                        tensors = [x if isinstance(x, torch.Tensor) else torch.as_tensor(x) for x in datas]
                        batch[batch_type] = torch.stack(tensors)
        
        batch['dataset'] = instances[0]['dataset']
        batch['face_or_frame'] = instances[0]['face_or_frame']
        
        # 传递预提取标志
        batch['frame_preextracted'] = instances[0].get('frame_preextracted', False)
        batch['face_preextracted'] = instances[0].get('face_preextracted', False)
        batch['audio_preextracted'] = instances[0].get('audio_preextracted', False)
        batch['multi_preextracted'] = instances[0].get('multi_preextracted', False)
        # 🎯 AU/Nonverbal不再作为特征模态，直接作为文本嵌入prompt
        
        return batch
    

class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        
        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
