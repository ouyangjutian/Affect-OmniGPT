import os
import time
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import decord
decord.bridge.set_bridge('torch')

from my_affect_omnigpt.tasks import *
from my_affect_omnigpt.models import *
from my_affect_omnigpt.runners import *
from my_affect_omnigpt.processors import *
from my_affect_omnigpt.datasets.builders import *
from my_affect_omnigpt.common.config import Config
from my_affect_omnigpt.common.dist_utils import get_rank
from my_affect_omnigpt.common.registry import registry
from my_affect_omnigpt.conversation.conversation_video import Chat
from my_affect_omnigpt.datasets.builders.image_text_pair_builder import * # 加载所有dataset cls

import config
from toolkit.utils.read_files import *


# 采用的是这个文件下存储数量最多的 root
def search_for_ckpt_root(root_candidates):
    if len(root_candidates) == 0:
        return ''
    
    # 找到 files 最多的 root
    maxcount = 0
    targetroot = ''
    for root in root_candidates:
        count = len([path for path in os.listdir(root) if path.startswith('checkpoint_')])
        print (root, '==>', count)
        if count > maxcount:
            maxcount = count
            targetroot = root
    print ('================================================')
    print (f'Targetroot: epoch range: 0-{maxcount-1}')
    
    # 打印最后一个文件的创建时间 for targetroot
    last_file = sorted(glob.glob(targetroot + '/checkpoint*'))[-1]
    file_stat = Path(last_file).stat()
    creation_time = file_stat.st_ctime
    print("Targetroot: Last ckpt creation time:", datetime.fromtimestamp(creation_time))
    print ('================================================')
    return targetroot


# case1: 默认 => last epoch
# case2: 指定 inference_cfg.test_epoch == a; 那就只跑这个 epoch 下的结果
# case3: 指定 inference_cfg.test_epochs == a-b; 跑最后一个
def get_ckpt3_candidates(ckpt3_root, inference_cfg):
    
    if inference_cfg.test_epoch != 'xxx':
        cur_epoch = inference_cfg.test_epoch
        ckpts = glob.glob("%s/*%06d*.pth" %(ckpt3_root, int(cur_epoch)))
        assert len(ckpts) == 1, 'Error: (ckpt, epoch) combination is not exists or contain multiple candidates!'
        return [ckpts[0]]
    
    elif inference_cfg.test_epochs == 'xxx-xxx':
        last_ckpt = sorted(glob.glob("%s/*.pth" %(ckpt3_root)))[-1]
        last_epoch=  int(last_ckpt.split('_')[-3])
        assert last_epoch > 10, f'Error: too less training time to conduct automatic inference!'
        return [last_ckpt]
    
    else:
        start_epoch, end_epoch = inference_cfg.test_epochs.split('-')
        skip_epoch = int(inference_cfg.skip_epoch) 
        whole_ckpts = []
        for cur_epoch in range(int(start_epoch), int(end_epoch)+1):
            if cur_epoch % skip_epoch == 0:
                ckpts = glob.glob("%s/*%06d*.pth" %(ckpt3_root, int(cur_epoch)))
                assert len(ckpts) == 1, 'Error: (ckpt, epoch) combination is not exists or contain multiple candidates!'
                whole_ckpts.append(ckpts[0])
        return whole_ckpts


# 因为我们目前只处理 merbench，这些是 video 的，需要和原始训练数据中的 video 数据对应的 face_or_frame 一致
def get_face_or_frame(datasets_cfg, outside_face_or_frame):
    if outside_face_or_frame is not None:
        return outside_face_or_frame
    
    face_or_frame_candidates = []
    if 'mercaptionplus' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['mercaptionplus'].face_or_frame)
    if 'ovmerd' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['ovmerd'].face_or_frame)
    assert len(set(face_or_frame_candidates)) == 1, f'must has the unified face_or_frame type'
    face_or_frame = list(set(face_or_frame_candidates))[0]
    return face_or_frame


# 优先级：outside_user_message > zeroshot > use_reasoning > dataset specific
def get_user_message(dataset_cls, zeroshot, outside_user_message, use_reasoning=True):
    if outside_user_message is not None:
        user_message = outside_user_message
    elif zeroshot:
        # zeroshot优先：只要求分类，不要求reasoning
        user_message = dataset_cls.func_get_qa_ovlabel(sample=None, question_only=True)
    elif use_reasoning:
        # 使用reasoning模式：要求模型给出推理过程
        user_message = dataset_cls.func_get_qa_description(sample=None, question_only=True)
    else:
        # 默认使用reasoning
        user_message = dataset_cls.func_get_qa_description(sample=None, question_only=True)
    return user_message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AffectGPT Inference Process")
    parser.add_argument("--cfg-path", default='xxx', help="path to configuration file.")
    parser.add_argument("--options",  nargs="+", help="override some settings in the used config, format: --option xx=xx yy=yy zz=zz")
    parser.add_argument('--zeroshot', action='store_true', default=False, help='whether testing on zeroshot performance?')
    parser.add_argument('--no_reasoning', action='store_true', default=False, help='disable reasoning output, only classification')
    parser.add_argument('--outside_user_message',  default=None, help="we use the outside user message, rather than dataset dependent.")
    parser.add_argument('--outside_face_or_frame', default=None, help="we use the outside face_or_frame, rather than dataset dependent.")
    parser.add_argument('--video_path', default=None)
    parser.add_argument('--audio_path', default=None)
    parser.add_argument('--subtitle',  default=None)
    parser.add_argument('--nonverbal_text', default=None, help="Nonverbal description text to embed in prompt")
    args = parser.parse_args()
    cfg = Config(args)
    model_cfg = cfg.model_cfg
    datasets_cfg = cfg.datasets_cfg
    inference_cfg = cfg.inference_cfg
    device = 'cuda:{}'.format(inference_cfg.gpu)
    

    print ('======== Step1: cfg pre-analysis ========')
    # 支持 ckpt_root / ckpt_name 两种类型输入 => (ckpt3_root)
    # 默认情况是依据 os.path.basename(args.cfg_path) 找到 => (ckpt3_root)
    if inference_cfg.ckpt_root not in ['', 'xxx']:
        ckpt3_root = inference_cfg.ckpt_root
    elif inference_cfg.ckpt_name not in ['', 'xxx']:
        cfg_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
        ckpt3_root = os.path.join('output', cfg_name, inference_cfg.ckpt_name)
        assert inference_cfg.ckpt_name.startswith(cfg_name) # 这块和 train 部分是相互配合下的结果
    else:
        print ('strat searching for suitable ckpt_root')
        cfg_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
        root_candidates = glob.glob(os.path.join('output', cfg_name, cfg_name+'*'))
        ckpt3_root = search_for_ckpt_root(root_candidates)
    print ('processed ckpt3 root:')
    print (ckpt3_root)

    # (ckpt3_root) => processed epochs
    print ('processed ckpt3 epochs:')
    whole_ckpt3s = get_ckpt3_candidates(ckpt3_root, inference_cfg)
    for item in whole_ckpt3s: print (os.path.basename(item))

    # => (face_or_frame) (这个需要与训练数据采用的 face_or_frame 相同)
    face_or_frame = get_face_or_frame(datasets_cfg, args.outside_face_or_frame)
    print (f'Read data type: {face_or_frame}')
    print ('=======================================')


    ## main process for each ckpt3 candidates
    for ii, ckpt_3 in enumerate(whole_ckpt3s):

        ##############################################################
        print (f'======== Step2: initial model; using ckpt_3: {os.path.basename(ckpt_3)} ========')
        model_cfg.ckpt_3 = ckpt_3 # ckpt_3 has the highest priority
        if ii == 0: # first-round: initialize models
            model_cls = registry.get_model_class(model_cfg.arch) # affectgpt
            model = model_cls.from_config(model_cfg)
        if ii > 0:  # second-round: update trainable params (用新的 ckpt_3 参数覆盖)
            ckpt = torch.load(model_cfg.ckpt_3, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt['model'], strict=False)
        model = model.to(device).eval() # !! reduce randomness during the inference
        chat = Chat(model, model_cfg, device=device)
        ##############################################################


        print ('======== Step3: Inferece ========')
        ## dataset_cls 内部在 train / inference 内部的更新
        dataset_cls = MER2025OV_Dataset()
        dataset_cls.needed_data = dataset_cls.get_needed_data(face_or_frame)
        dataset_cls.vis_processor = BaseProcessor()
        dataset_cls.img_processor = BaseProcessor()
        vis_processor_cfg = inference_cfg.get("vis_processor") # read vis processor
        img_processor_cfg = inference_cfg.get("img_processor") # read img processor
        if vis_processor_cfg is not None:
            dataset_cls.vis_processor = registry.get_processor_class(vis_processor_cfg.train.name).from_config(vis_processor_cfg.train)
        if img_processor_cfg is not None:
            dataset_cls.img_processor = registry.get_processor_class(img_processor_cfg.train.name).from_config(img_processor_cfg.train)
        dataset_cls.n_frms = model_cfg.vis_processor.train.n_frms
        
        # 添加Frame采样配置支持 - 从inference_cfg中读取
        dataset_cls.frame_n_frms = getattr(inference_cfg, 'frame_n_frms', dataset_cls.n_frms)
        dataset_cls.frame_sampling = getattr(inference_cfg, 'frame_sampling', 'uniform')
        
        # 推理模式配置 - 单样本推理默认使用实时处理
        dataset_cls.use_realtime_extraction = False
        
        # 各模态预提取配置 - 单样本推理默认全部使用实时处理
        if dataset_cls.frame_sampling == 'emotion_peak':
            dataset_cls.use_preextracted_frame = True
        else:
            dataset_cls.use_preextracted_frame = False
        dataset_cls.use_preextracted_face = getattr(inference_cfg, 'use_preextracted_face', False)
        dataset_cls.use_preextracted_audio = getattr(inference_cfg, 'use_preextracted_audio', False)
        dataset_cls.use_preextracted_au = getattr(inference_cfg, 'use_preextracted_au', False)
        
        dataset_cls.preextracted_root = getattr(inference_cfg, 'preextracted_root', './preextracted_features')
        dataset_cls.visual_encoder = getattr(inference_cfg, 'visual_encoder', 'CLIP_VIT_LARGE')
        dataset_cls.acoustic_encoder = getattr(inference_cfg, 'acoustic_encoder', 'HUBERT_LARGE')
        
        # Nonverbal文本模式配置
        dataset_cls.use_nonverbal_text = False
        dataset_cls.nonverbal_json = None
        dataset_cls._nonverbal_data = None


        ## main process for one sample        
        subtitle = args.subtitle
        video_path = args.video_path
        audio_path = args.audio_path
        image_path = None
        face_npy = None
        
        # 🎯 如果需要face模态，自动推断face_npy路径
        if 'face' in face_or_frame and video_path is not None:
            # 从video_path推断face_npy路径
            # video_path: /path/to/dataset/video/sample_name.mp4
            # face_npy:   /path/to/dataset/openface_face/sample_name/sample_name.npy
            video_dir = os.path.dirname(video_path)  # /path/to/dataset/video
            dataset_root = os.path.dirname(video_dir)  # /path/to/dataset
            sample_name = os.path.splitext(os.path.basename(video_path))[0]  # sample_name
            
            # 尝试两种常见的face路径格式
            face_candidates = [
                os.path.join(dataset_root, 'openface_face', sample_name, sample_name + '.npy'),  # MER2025OV格式
                os.path.join(dataset_root, 'openface_face', sample_name + '.npy'),  # MER2023等格式
            ]
            
            for candidate in face_candidates:
                if os.path.exists(candidate):
                    face_npy = candidate
                    print(f"✅ 自动找到face特征: {face_npy}")
                    break
            
            if face_npy is None:
                print(f"⚠️ 未找到face特征文件，尝试过的路径:")
                for c in face_candidates:
                    print(f"   - {c}")
                print("   将跳过face模态")
        
        sample_data = dataset_cls.read_frame_face_audio_text(video_path, face_npy, audio_path, image_path)

        # => img_list
        audio_llms, frame_llms, face_llms, image_llms, multi_llms = None, None, None, None, None
        audio_hiddens, audio_llms = chat.postprocess_audio(sample_data)  
        frame_hiddens, frame_llms = chat.postprocess_frame(sample_data)
        face_hiddens,  face_llms  = chat.postprocess_face(sample_data)
        _,             image_llms = chat.postprocess_image(sample_data)
        if face_or_frame.startswith('multiface'):
            _, multi_llms = chat.postprocess_multi(face_hiddens, audio_hiddens)
        elif face_or_frame.startswith('multiframe'):
            _, multi_llms = chat.postprocess_multi(frame_hiddens, audio_hiddens)

        img_list = {}
        img_list['audio'] = audio_llms
        img_list['frame'] = frame_llms
        img_list['face']  = face_llms
        img_list['image'] = image_llms
        img_list['multi'] = multi_llms

        # 🎯 根据实际可用的特征动态调整face_or_frame
        # 单样本推理时，如果某个模态特征为None，需要从face_or_frame中移除
        actual_face_or_frame = face_or_frame
        modality_available = {
            'face': face_llms is not None,
            'frame': frame_llms is not None,
            'audio': audio_llms is not None,
            'multi': multi_llms is not None,
        }
        
        # 检查并警告缺失的模态
        missing_modalities = []
        for modality, available in modality_available.items():
            if modality in actual_face_or_frame and not available:
                missing_modalities.append(modality)
        
        if missing_modalities:
            print(f"⚠️ 警告: 以下模态在配置中但特征不可用: {missing_modalities}")
            # 从face_or_frame中移除不可用的模态
            for modality in missing_modalities:
                # 处理各种可能的格式: "face_frame_au", "multiface_au", etc.
                if f'_{modality}_' in actual_face_or_frame:
                    actual_face_or_frame = actual_face_or_frame.replace(f'_{modality}_', '_')
                elif actual_face_or_frame.startswith(f'{modality}_'):
                    actual_face_or_frame = actual_face_or_frame[len(f'{modality}_'):]
                elif actual_face_or_frame.endswith(f'_{modality}'):
                    actual_face_or_frame = actual_face_or_frame[:-len(f'_{modality}')]
                elif f'multi{modality}' in actual_face_or_frame:
                    actual_face_or_frame = actual_face_or_frame.replace(f'multi{modality}', 'multi')
            print(f"📝 调整后的模态配置: {actual_face_or_frame}")

        # 获取Nonverbal文本（从命令行参数）
        nonverbal_text = args.nonverbal_text

        # get prompt (use_reasoning=True => reasoning; zeroshot => ov labels; else => dataset specific)
        use_reasoning = not args.no_reasoning
        user_message = get_user_message(dataset_cls, args.zeroshot, args.outside_user_message, use_reasoning)
        prompt = dataset_cls.get_prompt_for_multimodal(actual_face_or_frame, subtitle, user_message, nonverbal_text=nonverbal_text)
        
        # => call function
        response = chat.answer_sample(prompt=prompt, img_list=img_list,
                                    num_beams=1, temperature=1, do_sample=True, top_p=0.9, 
                                    max_new_tokens=1200, max_length=2000) # llama: max_token_num=2048
        print (response)
