import copy
import einops
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import BertConfig
from my_affectgpt.common.registry import registry
from my_affectgpt.models.blip2 import Blip2Base, disabled_train
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from my_affectgpt.models.Qformer import BertConfig, BertLMHeadModel
from my_affectgpt.models.tokenizer import load_tokenizer_from_LLM
from my_affectgpt.models.encoder import * # 只有调用了，才能实现 registry 过程
import config


@registry.register_model("affectgpt")
class AffectGPT(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/affectgpt.yaml",
    }

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("models/bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(
        self,
        visual_encoder_name,
        acoustic_encoder_name,
        llama_model_name,
        frozen_video_proj,
        frozen_video_Qformer,
        frozen_audio_Qformer,
        frozen_audio_proj,
        frozen_llm,
        lora_r,
        num_video_query_token,
        num_audio_query_token,
        num_multi_query_token,
        num_image_query_token,
        num_au_query_token,        # 新增：AU query token数量
        frozen_multi_Qformer,
        frozen_multi_llama_proj,
        frozen_au_proj,            # 新增：AU投影层冻结参数
        multi_fusion_type,
        video_fusion_type,
        audio_fusion_type,
        image_fusion_type,
        au_fusion_type,            # 新增：AU融合类型
        skip_encoders=False,  # 新增参数：是否跳过编码器加载
    ):
        super().__init__()

        print('====== Loading LLM ======')
        '''
        # => 目前尚未给 LLaMA 增加 QLora 层
        llama token ids:
            <unk>: 0
            bos|<s>: 1
            eos|pad|</s>: 2
            <ImageHere>: 32000
            <AudioHere>: 32001
        '''
        self.llama_model_name = llama_model_name    
        self.llama_tokenizer = load_tokenizer_from_LLM(llama_model_name)
        DEFAULT_IMAGE_PATCH_TOKEN = config.DEFAULT_IMAGE_PATCH_TOKEN
        DEFAULT_AUDIO_PATCH_TOKEN = config.DEFAULT_AUDIO_PATCH_TOKEN
        DEFAULT_FRAME_PATCH_TOKEN = config.DEFAULT_FRAME_PATCH_TOKEN
        DEFAULT_FACE_PATCH_TOKEN  = config.DEFAULT_FACE_PATCH_TOKEN
        DEFAULT_MULTI_PATCH_TOKEN = config.DEFAULT_MULTI_PATCH_TOKEN
        # 🎯 Lexical信息直接作为文本嵌入，不需要PATCH_TOKEN
        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]
        self.AUDIO_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_AUDIO_PATCH_TOKEN]
        self.FRAME_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_FRAME_PATCH_TOKEN]
        self.FACE_PATCH_TOKEN_ID  = self.llama_tokenizer.get_vocab()[DEFAULT_FACE_PATCH_TOKEN]
        self.MULTI_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_MULTI_PATCH_TOKEN]
        # 🎯 Lexical信息直接作为文本嵌入，不需要TOKEN_ID

        if llama_model_name in ['Baichuan2']:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                config.PATH_TO_LLM[llama_model_name],
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                config.PATH_TO_LLM[llama_model_name],
                torch_dtype=torch.float16
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False


        print('====== Using LoRA on LLM ======')
        from peft import get_peft_model, LoraConfig, TaskType
        
        # freeze base model's layers
        for param in self.llama_model.parameters():
            param.requires_grad = False
        
        # LoRA 部分参数是可调节的
        layer_num = len(self.llama_model.model.layers)
        target_modules=['model.layers.'+str(i)+'.'+ k for i in range(layer_num) for k in ["self_attn.q_proj", "self_attn.k_proj", 
                                                                                        "self_attn.v_proj", "self_attn.o_proj", 
                                                                                        "mlp.gate_proj","mlp.down_proj","mlp.up_proj"]]
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                                    inference_mode=False, 
                                    r=lora_r, lora_alpha=32, 
                                    lora_dropout=0.05, 
                                    target_modules=target_modules)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        
        if frozen_llm:
            for param in self.llama_model.parameters(): # lora 部分也冻结
                param.requires_grad = False
            print('freeze: LLAMA Model')
        else:
            print('trainable: LLAMA Model') # lora 部分可训练
        self.llama_model.print_trainable_parameters()
        
        
        print('====== Loading Image Encoder ======')
        self.image_fusion_type = image_fusion_type
        self.num_image_query_token = num_image_query_token
        self.skip_encoders = skip_encoders
        
        if not skip_encoders:
            self.visual_encoder = registry.get_visual_encoder_class(visual_encoder_name)()
            visual_hidden_size = self.visual_encoder.hidden_size
        else:
            print('🎯 Skipping visual encoder loading (preextracted mode)')
            self.visual_encoder = None
            # 从配置中获取预提取特征维度，或使用默认值
            visual_hidden_size = getattr(self, 'preextracted_visual_dim', 768)
            
        self.image_llama_proj = nn.Linear(visual_hidden_size, 
                                        self.llama_model.config.hidden_size)
        
        print('====== Loading Video Q-Former ======')
        self.video_fusion_type = video_fusion_type
        self.num_video_query_token = num_video_query_token

        ## case1: qformer
        if self.video_fusion_type == 'qformer':
            self.video_frame_position_embedding = nn.Embedding(32, visual_hidden_size) # [32, featdim]
            self.video_Qformer, self.video_query_tokens = self.init_video_Qformer(num_query_token=num_video_query_token,
                                                                                vision_width=visual_hidden_size, 
                                                                                num_hidden_layers=2)
            self.video_Qformer.cls = None
            self.video_Qformer.bert.embeddings.word_embeddings = None
            self.video_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.video_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

            if frozen_video_Qformer:
                for name, param in self.video_Qformer.named_parameters():
                    param.requires_grad = False
                for name, param in self.video_frame_position_embedding.named_parameters():
                    param.requires_grad = False
                self.video_query_tokens.requires_grad = False
                print('freeze: video_Qformer')
            else:
                for name, param in self.video_Qformer.named_parameters():
                    param.requires_grad = True
                for name, param in self.video_frame_position_embedding.named_parameters():
                    param.requires_grad = True
                self.video_query_tokens.requires_grad = True
                print('trainable: video_Qformer')
            video_hidden_size = self.video_Qformer.config.hidden_size
        ## case2: mean
        elif self.video_fusion_type == 'mean':
            video_hidden_size = visual_hidden_size
        ## case3: attention
        elif self.video_fusion_type == 'attention':
            self.video_attention_mlp = nn.Linear(visual_hidden_size, 1)
            video_hidden_size = visual_hidden_size


        print(f'====== Loading Video LLAMA proj ======')
        self.affectgpt_proj = nn.Linear(video_hidden_size, self.llama_model.config.hidden_size)
        if frozen_video_proj:
            for name, param in self.affectgpt_proj.named_parameters():
                param.requires_grad = False
            print('freeze: Video Q-Former LLaMA proj')
        else:
            for name, param in self.affectgpt_proj.named_parameters():
                param.requires_grad = True
            print('trainable: Video Q-Former LLaMA proj')




        print(f'====== Loading Audio Encoder ======')
        if not skip_encoders:
            self.acoustic_encoder = registry.get_acoustic_encoder_class(acoustic_encoder_name)()
            acoustic_hidden_size = self.acoustic_encoder.hidden_size
        else:
            print('🎯 Skipping acoustic encoder loading (preextracted mode)')
            self.acoustic_encoder = None
            # 从配置中获取预提取特征维度，或使用默认值
            acoustic_hidden_size = getattr(self, 'preextracted_acoustic_dim', 1024)

        print('====== Loading Audio Q-Former: ======')
        self.audio_fusion_type = audio_fusion_type
        self.num_audio_query_token = num_audio_query_token

        if self.audio_fusion_type == 'qformer':
            self.audio_position_embedding = nn.Embedding(8, acoustic_hidden_size)
            self.audio_Qformer, self.audio_query_tokens = self.init_video_Qformer(num_query_token=self.num_audio_query_token,
                                                                                vision_width=acoustic_hidden_size, 
                                                                                num_hidden_layers=2)
            self.audio_Qformer.cls = None
            self.audio_Qformer.bert.embeddings.word_embeddings = None
            self.audio_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.audio_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            if frozen_audio_Qformer:
                for name, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = False
                self.audio_query_tokens.requires_grad = False
                for name, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = False
                print('freeze: audio_Qformer')
            else:
                for name, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = True
                self.audio_query_tokens.requires_grad = True
                for name, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = True
                print('trainable: audio_Qformer')
            audio_hidden_size = self.audio_Qformer.config.hidden_size
        elif self.audio_fusion_type == 'mean':
            audio_hidden_size = acoustic_hidden_size
        elif self.audio_fusion_type == 'attention':
            self.audio_attention_mlp = nn.Linear(acoustic_hidden_size, 1)
            audio_hidden_size = acoustic_hidden_size

        print('====== Loading audio_llama_proj: ======')
        self.audio_llama_proj = nn.Linear(audio_hidden_size, 
                                        self.llama_model.config.hidden_size)
        if frozen_audio_proj:
            for name, param in self.audio_llama_proj.named_parameters():
                param.requires_grad = False
            print('freeze: Audio Q-Former LLaMA proj')
        else:
            for name, param in self.audio_llama_proj.named_parameters():
                param.requires_grad = True
            print('trainable: Audio Q-Former LLaMA proj')

        # ====== AU模态处理 ======
        print('====== Loading AU Q-Former: ======')
        self.num_au_query_token = num_au_query_token
        self.au_fusion_type = au_fusion_type
        
        # AU特征维度保持CLIP ViT-B/32原始输出512维
        au_hidden_size = 512
        
        if self.au_fusion_type == 'mean':
            # 简单平均融合
            pass  # 不需要额外的层
        elif self.au_fusion_type == 'attention':
            # 注意力融合
            self.au_attention_mlp = nn.Linear(au_hidden_size, 1)
        elif self.au_fusion_type == 'qformer':
            # Q-Former融合 (更复杂，可选)
            encoder_config = BertConfig.from_pretrained("models/bert-base-uncased")
            self.au_position_embedding = nn.Embedding(32, au_hidden_size)  # 最多支持32帧
            self.au_Qformer, self.au_query_tokens = self.init_video_Qformer(
                num_query_token=self.num_au_query_token,
                vision_width=au_hidden_size,
                num_hidden_layers=2
            )
            self.au_Qformer.cls = None
            self.au_Qformer.bert.embeddings.word_embeddings = None
            self.au_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.au_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            au_hidden_size = self.au_Qformer.config.hidden_size
        
        print('====== Loading au_llama_proj: ======')
        self.au_llama_proj = nn.Linear(au_hidden_size, self.llama_model.config.hidden_size)
        if frozen_au_proj:
            for name, param in self.au_llama_proj.named_parameters():
                param.requires_grad = False
            print('freeze: AU LLaMA proj')
        else:
            for name, param in self.au_llama_proj.named_parameters():
                param.requires_grad = True
            print('trainable: AU LLaMA proj')

        print('====== Loading Multi Q-Former (pre-fusion: this part is put in front of LLMs): ======')
        self.num_multi_query_token = num_multi_query_token
        self.multi_fusion_type = multi_fusion_type
        self.max_hidden_size = max(acoustic_hidden_size, visual_hidden_size)
        self.multi_audio_embs = nn.Linear(acoustic_hidden_size, self.max_hidden_size)
        self.multi_video_embs = nn.Linear(visual_hidden_size, self.max_hidden_size)

        ## case1: [audio, video] + Q-Former
        if self.multi_fusion_type == 'qformer':
            encoder_config = BertConfig.from_pretrained("models/bert-base-uncased") # Q-Former 的输出维度是固定的 = 768
            self.multi_position_embedding = nn.Embedding(264, self.max_hidden_size)
            self.multi_Qformer, self.multi_query_tokens = self.init_video_Qformer(num_query_token=self.num_multi_query_token,
                                                                                vision_width=self.max_hidden_size, 
                                                                                num_hidden_layers=2)
            self.multi_Qformer.cls = None
            self.multi_Qformer.bert.embeddings.word_embeddings = None
            self.multi_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.multi_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            if frozen_multi_Qformer:
                for name, param in self.multi_Qformer.named_parameters():
                    param.requires_grad = False
                self.multi_query_tokens.requires_grad = False
                for name, param in self.multi_position_embedding.named_parameters():
                    param.requires_grad = False
                print('freeze: multi_Qformer')
            else:
                for name, param in self.multi_Qformer.named_parameters():
                    param.requires_grad = True
                self.multi_query_tokens.requires_grad = True
                for name, param in self.multi_position_embedding.named_parameters():
                    param.requires_grad = True
                print('trainable: multi_Qformer')
            multi_hidden_size = self.multi_Qformer.config.hidden_size # 设置 output_hidden_size
        elif self.multi_fusion_type == 'attention':
            self.attention_mlp = nn.Linear(self.max_hidden_size * 2, self.max_hidden_size)
            self.fc_att = nn.Linear(self.max_hidden_size, 2)
            multi_hidden_size = self.max_hidden_size # 设置 output_hidden_size

        print('====== Loading multi_llama_proj: ======')
        self.multi_llama_proj = nn.Linear(multi_hidden_size, self.llama_model.config.hidden_size)
        if frozen_multi_llama_proj:
            for name, param in self.multi_llama_proj.named_parameters():
                param.requires_grad = False
            print('freeze: Multi Q-Former LLaMA proj')
        else:
            for name, param in self.multi_llama_proj.named_parameters():
                param.requires_grad = True
            print('trainable: Multi Q-Former LLaMA proj')

        
    # ===================================================== #
    # ===================================================== #
    # 为 EVA_CLIP 保留每个图片的 32 tokens
    # 其他 visual encoder 默认是 1 tokens
    def encode_image_token(self, image, raw_image):
        device = image.device
        with self.maybe_autocast():

            # VIT: w/ Q-Former or w/o Q-Former: [b c t h w] -> [b, t, q=32, h=768]
            frame_hidden_state = self.visual_encoder(image, raw_image).to(device)
            batch_size, time_length = frame_hidden_state.size()[:2]
            
            ## + Position Embedding [支持两种类型输入格式]
            # case1: 输入维度为 [b, t, 32, 768]
            if len(frame_hidden_state.size()) == 4:
                frame_hidden_state = einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h', b=batch_size, t=time_length) # [b, (t, 32), 768]
            # case2: 输入维度为 [b, t, 768]
            elif len(frame_hidden_state.size()) == 3:
                frame_hidden_state = frame_hidden_state.expand(-1, self.num_image_query_token, -1) # [b, t=img_token_num, featdim]

            # + llama_proj: 将输入映射到 LLM dimensional
            image_hidden = frame_hidden_state # [b, t=32, featdim]
            inputs_llama = self.image_llama_proj(image_hidden) # [b, t=32, llmdim]

        return image_hidden, inputs_llama
    
    # 将所有 image 压缩到 1 tokens
    def encode_image_mean(self, image, raw_image):
        device = image.device
        with self.maybe_autocast():

            # [b, t=1, q=32, h=768] / [b, t=1, h=768]
            hidden_state = self.visual_encoder(image, raw_image).to(device)
            
            ## fusion process -> [b, h]
            # case1: 输入维度为 [b, t=1, q=32, h=768]
            if len(hidden_state.size()) == 4:
                hidden_state = hidden_state.squeeze(axis=1) # [b, q, h]
                hidden_state = torch.mean(hidden_state, axis=1) # [b, h]
            # case2: 输入维度为 [b, t=1, 768]
            elif len(hidden_state.size()) == 3:
                hidden_state = hidden_state.squeeze(axis=1) # [b, h]
            
            ## map to LLM dimensions -> [b, token_num, llmdim]
            inputs_llama = self.image_llama_proj(hidden_state) # [b, llmdim]
            inputs_llama = torch.unsqueeze(inputs_llama, 1).expand(-1, self.num_image_query_token, -1) # [b, 16, llmdim]

        return None, inputs_llama

    def encode_image_merge(self, image, raw_image):
        if self.image_fusion_type == 'token':
            image_hiddens, image_llms = self.encode_image_token(image, raw_image) # image: [b c 1 h w] -> [b, 1,  4096]; raw_images: b * [Image.Image]
        elif self.image_fusion_type == 'mean':
            image_hiddens, image_llms = self.encode_image_mean(image, raw_image) 
        return image_hiddens, image_llms
        

    # ===================================================== #
    # ===================================================== #
    # 将视频的时间维度压缩到 32 tokens
    def encode_video_qformer(self, video, raw_video):
        device = video.device
        with self.maybe_autocast():

            # [b, t, q=32, h=768] / [b, t, h]
            frame_hidden_state = self.visual_encoder(video, raw_video).to(device)
            batch_size, time_length = frame_hidden_state.size()[:2]

            '''
            修订：获取 hidden state，用于后续的 multimodal fusion process
            store_hidden_state => [b, t, h]
            '''
            store_hidden_state = ""
            if len(frame_hidden_state.size()) == 4:
                store_hidden_state = einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h', b=batch_size, t=time_length)
            elif len(frame_hidden_state.size()) == 3:
                store_hidden_state = frame_hidden_state

            ## + Position Embedding [支持两种类型输入格式] => 两者都是在时间维度增加位置编码信息
            # case1: 输入维度为 [b, t, 32, 768]
            if len(frame_hidden_state.size()) == 4:
                position_ids = torch.arange(time_length, dtype=torch.long, device=device) 
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1) # [b, t]
                frame_position_embeddings = self.video_frame_position_embedding(position_ids) # [b, t, featdim]
                frame_position_embeddings = frame_position_embeddings.unsqueeze(-2) # [b, t, 1, featdim]
                frame_hidden_state = frame_position_embeddings + frame_hidden_state # [b, t, 32, 768]
                frame_hidden_state = einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h', b=batch_size, t=time_length) # [b, (t, 32), 768]
            # case2: 输入维度为 [b, t, 768]
            elif len(frame_hidden_state.size()) == 3:
                position_ids = torch.arange(time_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                frame_position_embeddings = self.video_frame_position_embedding(position_ids) # [b, t, 768]
                frame_hidden_state = frame_hidden_state + frame_position_embeddings # [b, t, 768]

            # + Video Q-Former: => 时间维度压缩到 32 tokens => [b, (t, 32), 768] 压缩到 [b, 32, 768]
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)
            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            # + llama_proj: 将输入映射到 LLM dimensional
            inputs_llama = self.affectgpt_proj(video_hidden)

        return store_hidden_state, inputs_llama
    
    # 将视频的时间维度压缩到 1 tokens
    def encode_video_mean(self, video, raw_video):
        device = video.device
        with self.maybe_autocast():

            # -> [b, t, q=32, h=768] / [b, t, h]
            hidden_state = self.visual_encoder(video, raw_video).to(device)
            batch_size, time_length = hidden_state.size()[:2]

            '''
            修订：获取 hidden state，用于后续的 multimodal fusion process
            store_hidden_state => [b, t, h]
            '''
            store_hidden_state = ""
            if len(hidden_state.size()) == 4:
                store_hidden_state = einops.rearrange(hidden_state, 'b t q h -> b (t q) h', b=batch_size, t=time_length)
            elif len(hidden_state.size()) == 3:
                store_hidden_state = hidden_state

            ## fusion process => [b, h]
            # case1: 输入维度为 [b, t, 32, 768]
            if len(hidden_state.size()) == 4:
                hidden_state = torch.mean(hidden_state, axis=2) # [b, t, h]
                hidden_state = torch.mean(hidden_state, axis=1) # [b, h]
            # case2: 输入维度为 [b, t, 768]
            elif len(hidden_state.size()) == 3:
                hidden_state = torch.mean(hidden_state, axis=1) # [b, h]

            # convert to llm inputs => [b, token, llmdim]
            inputs_llama = self.affectgpt_proj(hidden_state) # [b, llmdim]
            inputs_llama = torch.unsqueeze(inputs_llama, 1).expand(-1, self.num_video_query_token, -1) # [b, 16, llmdim]

        return store_hidden_state, inputs_llama
    
    # 将视频的时间维度压缩到 1 tokens
    def encode_video_attention(self, video, raw_video):
        device = video.device
        with self.maybe_autocast():

            # -> [b, t, q=32, h=768] / [b, t, h]
            hidden_state = self.visual_encoder(video, raw_video).to(device)
            batch_size, time_length = hidden_state.size()[:2]

            '''
            修订：获取 hidden state，用于后续的 multimodal fusion process
            store_hidden_state => [b, t, h]
            '''
            store_hidden_state = ""
            if len(hidden_state.size()) == 4:
                store_hidden_state = einops.rearrange(hidden_state, 'b t q h -> b (t q) h', b=batch_size, t=time_length)
            elif len(hidden_state.size()) == 3:
                store_hidden_state = hidden_state

            ## fusion process => [b, h]
            if len(hidden_state.size()) == 4:
                hidden_state = torch.mean(hidden_state, axis=2) # -> [b, t, h]
            attention = self.video_attention_mlp(hidden_state) # [b, t, h] -> [b, t, 1]
            hidden_state = einops.rearrange(hidden_state, 'b t h -> b h t', b=batch_size, t=time_length) # [b, h, t]
            fused_feat = torch.matmul(hidden_state, attention) # [b, h, 1]
            fused_feat  = fused_feat.squeeze(axis=2) # [b, h]

            # convert to llm inputs => [b, token, llmdim]
            inputs_llama = self.affectgpt_proj(fused_feat) # [b, llmdim]
            inputs_llama = torch.unsqueeze(inputs_llama, 1).expand(-1, self.num_video_query_token, -1) # [b, 16, llmdim]

        return store_hidden_state, inputs_llama
    
    def encode_video_merge(self, video, raw_video, is_preextracted=False):
        if is_preextracted:
            # 预提取特征模式 - 直接使用预提取的特征，跳过视觉编码器
            device = video.device
            
            # 🔧 使用maybe_autocast统一处理dtype转换（与实时模式保持一致）
            with self.maybe_autocast():
                batch_size = video.shape[0]
                
                # 检查特征维度以确定处理方式
                if len(video.shape) == 3:
                    # 多时间步特征 [b, t, d] - 需要时序融合 (Face: [b, 8, 768])
                    time_length, hidden_dim = video.shape[1], video.shape[2]
                    store_hidden_state = video  # [b, t, d]
                    
                    # 只有多时间步特征才需要融合处理
                elif len(video.shape) == 2:
                    # 单一特征向量 [b, d] - 无需时序融合 (Frame: [b, 768])
                    # 为了保持代码一致性，扩展为 [b, 1, d]
                    video = video.unsqueeze(1)  # [b, d] -> [b, 1, d]
                    time_length, hidden_dim = 1, video.shape[2]
                    store_hidden_state = video  # [b, 1, d]
                else:
                    raise ValueError(f"Unexpected video shape: {video.shape}")
                
                # 🎯 修复：统一处理流程，单帧特征也通过Q-Former处理以保持与实时模式一致
                if self.video_fusion_type == 'qformer':
                    # 添加位置编码（无论单帧还是多帧）- 使用与实时模式相同的位置编码
                    position_ids = torch.arange(time_length, dtype=torch.long, device=device)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                    video_position_embeddings = self.video_frame_position_embedding(position_ids)
                    video_hidden_state = store_hidden_state + video_position_embeddings
                    
                    # Q-Former处理（统一流程）
                    query_tokens = self.video_query_tokens.expand(batch_size, -1, -1)
                    attention_mask = torch.ones(video_hidden_state.size()[:-1], dtype=torch.long, device=device)
                    
                    query_output = self.video_Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=video_hidden_state,
                        encoder_attention_mask=attention_mask,
                        return_dict=True,
                    )
                    
                    # 投影到LLM空间
                    inputs_llama = self.affectgpt_proj(query_output.last_hidden_state)
                    frame_hiddens, frame_llms = store_hidden_state, inputs_llama
                
                elif self.video_fusion_type == 'attention':
                    # 注意力融合处理（统一流程）
                    if time_length == 1:
                        # 单帧特征直接处理
                        fused_feat = store_hidden_state.squeeze(1)  # [b, 1, 768] -> [b, 768]
                    else:
                        # 多帧特征注意力融合 - 与实时模式保持一致
                        attention = self.video_attention_mlp(store_hidden_state)  # [b, t, 1]
                        store_hidden_rearranged = einops.rearrange(store_hidden_state, 'b t h -> b h t', b=batch_size, t=time_length)  # [b, h, t]
                        fused_feat = torch.matmul(store_hidden_rearranged, attention)  # [b, h, 1]
                        fused_feat = fused_feat.squeeze(axis=2)  # [b, h]
                    
                    inputs_llama = self.affectgpt_proj(fused_feat)  # [b, llmdim]
                    inputs_llama = torch.unsqueeze(inputs_llama, 1).expand(-1, self.num_video_query_token, -1)  # [b, num_video_query_token, llmdim]
                    frame_hiddens, frame_llms = store_hidden_state, inputs_llama
                    
                elif self.video_fusion_type == 'mean':
                    # 均值融合处理预提取特征
                    mean_features = torch.mean(store_hidden_state, dim=1)  # [batch, hidden_dim]
                    
                    # 投影到LLM空间
                    inputs_llama = self.affectgpt_proj(mean_features)  # [b, llmdim]
                    inputs_llama = torch.unsqueeze(inputs_llama, 1).expand(-1, self.num_video_query_token, -1)  # [b, num_video_query_token, llmdim]
                    frame_hiddens, frame_llms = store_hidden_state, inputs_llama
                
            return frame_hiddens, frame_llms
                
        else:
            # 实时处理模式 - 原有逻辑
            if self.visual_encoder is None:
                raise RuntimeError("Visual encoder is None but trying to use real-time mode. This indicates feature extraction service failed. Please check the service status.")
            else:
                # 传统实时模式：训练进程有编码器
                if self.video_fusion_type == 'qformer':
                    frame_hiddens, frame_llms = self.encode_video_qformer(video, raw_video) # frame: [b c t h w] -> [b, 32, 4096]
                elif self.video_fusion_type == 'attention':
                    frame_hiddens, frame_llms = self.encode_video_attention(video, raw_video)
                elif self.video_fusion_type == 'mean':
                    frame_hiddens, frame_llms = self.encode_video_mean(video, raw_video)
        return frame_hiddens, frame_llms

    # ===================================================== #
    # ===================================================== #
    def encode_audio_qformer(self, audio, raw_audio):
        device = audio.device
        with self.maybe_autocast():

            # Audio encoder: [b, t, 1024]
            if self.acoustic_encoder is None:
                raise RuntimeError("Acoustic encoder is None but trying to use real-time mode. This indicates feature extraction service failed. Please check the service status.")
            audio_hidden_state = self.acoustic_encoder(audio, raw_audio).to(device)
            batch_size, time_length = audio_hidden_state.size()[:2]

            '''
            修订：获取 hidden state，用于后续的 multimodal fusion process
            store_hidden_state => [b, t, h]
            '''
            store_hidden_state = audio_hidden_state

            # + Position Embeddings:
            position_ids = torch.arange(time_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            audio_position_embeddings = self.audio_position_embedding(position_ids) # [b, t, 1024]
            audio_hidden_state = audio_hidden_state + audio_position_embeddings # [b, t, 1024]

            # + Audio Q-Former: [b, t, 1024] -> [b, t, 786]
            frame_atts = torch.ones(audio_hidden_state.size()[:-1], dtype=torch.long).to(device)
            audio_query_tokens = self.audio_query_tokens.expand(audio_hidden_state.shape[0], -1, -1)
            audio_query_output = self.audio_Qformer.bert(
                query_embeds=audio_query_tokens,
                encoder_hidden_states=audio_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
            )
            audio_hidden = audio_query_output.last_hidden_state

            # + audio_llama_proj
            inputs_llama = self.audio_llama_proj(audio_hidden)
    
        return store_hidden_state, inputs_llama

    def encode_audio_mean(self, audio, raw_audio):
        device = audio.device
        with self.maybe_autocast():

            # audio encoder: [b, t, 1024]
            if self.acoustic_encoder is None:
                raise RuntimeError("Acoustic encoder is None but trying to use real-time mode. This indicates feature extraction service failed. Please check the service status.")
            hidden_state = self.acoustic_encoder(audio, raw_audio).to(device)

            '''
            修订：获取 hidden state，用于后续的 multimodal fusion process
            store_hidden_state => [b, t, h]
            '''
            store_hidden_state = hidden_state

            # fusion process => [b, h]
            hidden_state = torch.mean(hidden_state, axis=1) 

            # convert to llm inputs => [b, token, llmdim]
            inputs_llama = self.audio_llama_proj(hidden_state) # [b, llmdim]
            inputs_llama = torch.unsqueeze(inputs_llama, 1).expand(-1, self.num_audio_query_token, -1) # [b, 16, llmdim]
    
        return store_hidden_state, inputs_llama
    
    def encode_audio_attention(self, audio, raw_audio):
        device = audio.device
        with self.maybe_autocast():

            # Audio encoder: [b, t, 1024]
            if self.acoustic_encoder is None:
                raise RuntimeError("Acoustic encoder is None but trying to use real-time mode. This indicates feature extraction service failed. Please check the service status.")
            hidden_state = self.acoustic_encoder(audio, raw_audio).to(device)
            batch_size, time_length = hidden_state.size()[:2]

            '''
            修订：获取 hidden state，用于后续的 multimodal fusion process
            store_hidden_state => [b, t, h]
            '''
            store_hidden_state = hidden_state

            # fusion process => [b, h]
            attention = self.audio_attention_mlp(hidden_state) # [b, t, 1024] -> [b, t, 1]
            hidden_state = einops.rearrange(hidden_state, 'b t h -> b h t', b=batch_size, t=time_length) # [b, h, t]
            fused_feat = torch.matmul(hidden_state, attention) # [b, h, 1]
            fused_feat  = fused_feat.squeeze(axis=2) # [b, h]

            # convert to llm inputs => [b, token, llmdim]
            inputs_llama = self.audio_llama_proj(fused_feat) # [b, llmdim]
            inputs_llama = torch.unsqueeze(inputs_llama, 1).expand(-1, self.num_audio_query_token, -1) # [b, 16, llmdim]
    
        return store_hidden_state, inputs_llama

    def encode_audio_merge(self, audio, raw_audio, is_preextracted=False):
        if is_preextracted:
            # 预提取特征模式 - 直接使用预提取的特征，跳过声学编码器
            device = audio.device
            batch_size, time_length, hidden_dim = audio.size()  # [b, t, d]
            
            # 直接使用预提取特征作为hidden_state
            store_hidden_state = audio  # [b, t, d]
            
            # 🎯 修复：与实时模式保持完全一致的处理流程
            if self.audio_fusion_type == 'qformer':
                # 总是添加位置编码，与实时模式保持一致
                position_ids = torch.arange(time_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                audio_position_embeddings = self.audio_position_embedding(position_ids)
                audio_hidden_state = store_hidden_state + audio_position_embeddings
                
                # Q-Former处理
                query_tokens = self.audio_query_tokens.expand(batch_size, -1, -1)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=device)
                attention_mask = torch.ones(audio_hidden_state.size()[:-1], dtype=torch.long, device=device)
                
                query_output = self.audio_Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=audio_hidden_state,
                    encoder_attention_mask=attention_mask,
                    return_dict=True,
                )
                
                # 投影到LLM空间
                inputs_llama = self.audio_llama_proj(query_output.last_hidden_state)
                audio_hiddens, audio_llms = store_hidden_state, inputs_llama
                
            elif self.audio_fusion_type == 'attention':
                # 注意力融合处理预提取特征 - 使用与实时模式相同的attention机制
                batch_size, time_length = store_hidden_state.shape[0], store_hidden_state.shape[1]
                
                # 使用与实时模式相同的attention MLP
                attention = self.audio_attention_mlp(store_hidden_state)  # [b, t, h] -> [b, t, 1]
                audio_rearranged = einops.rearrange(store_hidden_state, 'b t h -> b h t', b=batch_size, t=time_length)  # [b, h, t]
                fused_feat = torch.matmul(audio_rearranged, attention)  # [b, h, 1]
                fused_feat = fused_feat.squeeze(axis=2)  # [b, h]
                
                # 投影到LLM空间
                inputs_llama = self.audio_llama_proj(fused_feat)  # [b, llmdim]
                inputs_llama = torch.unsqueeze(inputs_llama, 1).expand(-1, self.num_audio_query_token, -1)  # [b, num_audio_query_token, llmdim]
                audio_hiddens, audio_llms = store_hidden_state, inputs_llama
                
            elif self.audio_fusion_type == 'mean':
                # 均值融合处理预提取特征 [batch, clips, hidden_dim]
                mean_features = torch.mean(store_hidden_state, dim=1)  # [batch, hidden_dim]
                
                # 投影到LLM空间
                inputs_llama = self.audio_llama_proj(mean_features)  # [b, llmdim]
                inputs_llama = torch.unsqueeze(inputs_llama, 1).expand(-1, self.num_audio_query_token, -1)  # [b, num_audio_query_token, llmdim]
                audio_hiddens, audio_llms = store_hidden_state, inputs_llama
        else:
            # 实时处理模式 - 原有逻辑
            if self.audio_fusion_type == 'qformer':
                audio_hiddens, audio_llms = self.encode_audio_qformer(audio, raw_audio) # audio: [b t c h w] -> [b, 8,  4096]
            elif self.audio_fusion_type == 'attention':
                audio_hiddens, audio_llms = self.encode_audio_attention(audio, raw_audio)
            elif self.audio_fusion_type == 'mean':
                audio_hiddens, audio_llms = self.encode_audio_mean(audio, raw_audio)
        return audio_hiddens, audio_llms

    def encode_au_merge(self, au_features, is_preextracted=True):
        """处理AU特征 - 仅支持预提取模式，因为AU特征来自MER-Factory输出"""
        if is_preextracted:
            device = au_features.device
            
            with self.maybe_autocast():
                batch_size, time_length, hidden_dim = au_features.size()  # [b, t, 512]
                
                # AU特征保持CLIP ViT-B/32原始512维特征
                store_hidden_state = au_features  # [b, t, 512]
                
                # 根据融合类型处理
                if self.au_fusion_type == 'qformer':
                    # Q-Former处理
                    position_ids = torch.arange(time_length, dtype=torch.long, device=device)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                    au_position_embeddings = self.au_position_embedding(position_ids)
                    au_hidden_state = store_hidden_state + au_position_embeddings
                    
                    query_tokens = self.au_query_tokens.expand(batch_size, -1, -1)
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=device)
                    attention_mask = torch.ones(au_hidden_state.size()[:-1], dtype=torch.long, device=device)
                    
                    query_output = self.au_Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=au_hidden_state,
                        encoder_attention_mask=attention_mask,
                        return_dict=True,
                    )
                    
                    # 投影到LLM空间
                    inputs_llama = self.au_llama_proj(query_output.last_hidden_state)
                    au_hiddens, au_llms = store_hidden_state, inputs_llama
                    
                elif self.au_fusion_type == 'attention':
                    # 注意力融合
                    batch_size, time_length = store_hidden_state.shape[0], store_hidden_state.shape[1]
                    
                    attention = self.au_attention_mlp(store_hidden_state)  # [b, t, h] -> [b, t, 1]
                    au_rearranged = einops.rearrange(store_hidden_state, 'b t h -> b h t', b=batch_size, t=time_length)  # [b, h, t]
                    fused_feat = torch.matmul(au_rearranged, attention)  # [b, h, 1]
                    fused_feat = fused_feat.squeeze(axis=2)  # [b, h]
                    
                    # 投影到LLM空间
                    inputs_llama = self.au_llama_proj(fused_feat)  # [b, llmdim]
                    inputs_llama = torch.unsqueeze(inputs_llama, 1).expand(-1, self.num_au_query_token, -1)  # [b, num_au_query_token, llmdim]
                    au_hiddens, au_llms = store_hidden_state, inputs_llama
                    
                elif self.au_fusion_type == 'mean':
                    # 均值融合 - 默认方式
                    mean_features = torch.mean(store_hidden_state, dim=1)  # [batch, hidden_dim]
                    
                    # 投影到LLM空间
                    inputs_llama = self.au_llama_proj(mean_features)  # [b, llmdim]
                    inputs_llama = torch.unsqueeze(inputs_llama, 1).expand(-1, self.num_au_query_token, -1)  # [b, num_au_query_token, llmdim]
                    au_hiddens, au_llms = store_hidden_state, inputs_llama
            
            return au_hiddens, au_llms
        else:
            # AU特征只支持预提取模式
            raise RuntimeError("AU features only support preextracted mode")

    # ===================================================== #
    # ===================================================== #
    def encode_multi_qformer(self, video_hidden_state, audio_hidden_state):
        # print ('fusion type: qformer')
        device = video_hidden_state.device
        with self.maybe_autocast():

            # print (video_hidden_state.size()) # [b=3, t=8*32/8, featdim1]
            # print (audio_hidden_state.size()) # [b=3, t=8,      featdim2]
            video_hidden_state = self.multi_video_embs(video_hidden_state) # [b, t=8*32/8, maxdim]
            audio_hidden_state = self.multi_audio_embs(audio_hidden_state) # [b, t=8,      maxdim]
            multi_hidden_state = torch.concat([video_hidden_state, audio_hidden_state], axis=1) # [b, ta+tv, maxdim]
            # print (multi_hidden_state.size())
            batch_size, time_length = multi_hidden_state.size()[:2]

            # + Position Embeddings:
            position_ids = torch.arange(time_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            multi_position_embeddings = self.multi_position_embedding(position_ids) # [b, ta+tv, maxdim]
            multi_hidden_state = multi_hidden_state + multi_position_embeddings     # [b, ta+tv, maxdim]

            # + Multi Q-Former: [b, t, 1024] -> [b, t, 786]
            frame_atts = torch.ones(multi_hidden_state.size()[:-1], dtype=torch.long).to(device)
            multi_query_tokens = self.multi_query_tokens.expand(multi_hidden_state.shape[0], -1, -1)
            multi_query_output = self.multi_Qformer.bert(
                query_embeds=multi_query_tokens,
                encoder_hidden_states=multi_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
            )
            multi_hidden = multi_query_output.last_hidden_state

            # + multi_llama_proj
            inputs_llama = self.multi_llama_proj(multi_hidden)
    
        return multi_hidden, inputs_llama

    def encode_multi_attention(self, video_hidden_state, audio_hidden_state):
        # print ('fusion type: attention')
        device = video_hidden_state.device
        with self.maybe_autocast():
            
            # print (video_hidden_state.size()) # [3, 16*32/16, featdim2]
            # print (audio_hidden_state.size()) # [3, t=8,      featdim1]
            video_hidden_state = torch.mean(video_hidden_state, axis=1) # [b, featdim1]
            audio_hidden_state = torch.mean(audio_hidden_state, axis=1) # [b, featdim2]
            video_hidden_state = self.multi_video_embs(video_hidden_state) # [b, maxdim]
            audio_hidden_state = self.multi_audio_embs(audio_hidden_state) # [b, maxdim]

            multi_hidden_state = torch.concat([video_hidden_state, audio_hidden_state], axis=1) # [b, maxdim * 2]
            attention = self.attention_mlp(multi_hidden_state) # [b, maxdim]
            attention = self.fc_att(attention)  # [b, 2]
            attention = torch.unsqueeze(attention, 2) # [b, 2, 1]

            multi_hidden2 = torch.stack([video_hidden_state, audio_hidden_state], dim=2) # [b, maxdim, 2]
            fused_feat = torch.matmul(multi_hidden2, attention)  # [b, maxdim, 1]
            multi_hidden  = fused_feat.squeeze(axis=2) # [b, maxdim]

            # + multi_llama_proj
            inputs_llama = self.multi_llama_proj(multi_hidden) # [b, llmdim]
            inputs_llama = torch.unsqueeze(inputs_llama, 1).expand(-1, self.num_multi_query_token, -1) # [b, 16, llmdim]
    
        return multi_hidden, inputs_llama
    
    def encode_multi_merge(self, video_hidden_state, audio_hidden_state):
        if self.multi_fusion_type == 'qformer':
            multi_hiddens, multi_llms = self.encode_multi_qformer(video_hidden_state, audio_hidden_state)
        elif self.multi_fusion_type == 'attention':
            multi_hiddens, multi_llms = self.encode_multi_attention(video_hidden_state, audio_hidden_state)
        return multi_hiddens, multi_llms

    '''
    inference prompt:
    <s>###Human: Close your eyes, open your ears and you imagine only based on the sound that <Audio><AudioHere></Audio>. \
    Close your ears, open your eyes and you see that <Video><ImageHere></Video>.  \
    The subtitle of this video is <Subtitle>{subtitle}</Subtitle>. \
    Now answer my question based on what you have seen, heard, and subtitles. {user_message} ###Assistant:
    '''
    def forward(self, samples):

        self.face_or_frame = samples['face_or_frame'] # 把这个参数传出来
        frame_llms, face_llms, audio_llms, image_llms, multi_llms, au_llms = None, None, None, None, None, None
        if 'frames' in samples: 
            frame_hiddens, frame_llms = self.encode_video_merge(samples['frames'],  samples['raw_frames'], is_preextracted=samples.get('frame_preextracted', False)) # frame: [b c t h w] -> [b, 32, 4096]
        if 'faces'  in samples: 
            # print (samples['faces'].shape)
            face_hiddens,  face_llms  = self.encode_video_merge(samples['faces'],  samples['raw_faces'], is_preextracted=samples.get('face_preextracted', False)) # face:  [b c t h w] -> [b, 32, 4096]
        if 'audios' in samples: 
            audio_hiddens, audio_llms = self.encode_audio_merge(samples['audios'],  samples['raw_audios'], is_preextracted=samples.get('audio_preextracted', False))
        if 'aus' in samples:  # 新增：AU特征处理
            au_hiddens, au_llms = self.encode_au_merge(samples['aus'], is_preextracted=samples.get('au_preextracted', True))  # AU仅支持预提取模式
        if 'images' in samples: 
            image_hiddens, image_llms = self.encode_image_merge(samples['images'],  samples['raw_images'])
        if (samples['input_ids'][0] == self.MULTI_PATCH_TOKEN_ID).sum() != 0: # 这是时候才需要 multi
            # 🎯 修复：强制使用实时Multi融合，避免预提取Multi特征的近似误差
            # 即使有预提取的Multi特征，也使用实时融合以保持端到端梯度流
            if self.face_or_frame.startswith('multiface'):
                # multiface模式需要face和audio
                if 'faces' in samples and 'audios' in samples and face_hiddens is not None and audio_hiddens is not None:
                    multi_hiddens, multi_llms = self.encode_multi_merge(face_hiddens, audio_hiddens)
                else:
                    print("⚠️ Warning: multiface fusion requires both face and audio features")
                    multi_hiddens, multi_llms = None, None
            elif self.face_or_frame.startswith('multiframe'):
                # multiframe模式需要frame和audio
                if 'frames' in samples and 'audios' in samples and frame_hiddens is not None and audio_hiddens is not None:
                    multi_hiddens, multi_llms = self.encode_multi_merge(frame_hiddens, audio_hiddens)
                else:
                    print("⚠️ Warning: multiframe fusion requires both frame and audio features")
                    multi_hiddens, multi_llms = None, None
            else:
                print(f"⚠️ Warning: Unknown multi fusion type: {self.face_or_frame}")
                multi_hiddens, multi_llms = None, None

        # temp_input_ids: <ImageHere> -> [0]   
        input_ids = samples['input_ids']
        temp_input_ids = copy.deepcopy(input_ids)
        temp_input_ids[temp_input_ids == self.FRAME_PATCH_TOKEN_ID] = 0
        temp_input_ids[temp_input_ids == self.FACE_PATCH_TOKEN_ID]  = 0
        temp_input_ids[temp_input_ids == self.AUDIO_PATCH_TOKEN_ID] = 0
        temp_input_ids[temp_input_ids == self.MULTI_PATCH_TOKEN_ID] = 0
        temp_input_ids[temp_input_ids == self.IMAGE_PATCH_TOKEN_ID] = 0
        # 🎯 Lexical信息直接作为文本嵌入，不需要特殊处理
        temp_input_embedding = self.llama_model.model.model.embed_tokens(temp_input_ids) # 嵌套 LoRA 之后，会在 model 外面再包一层

        ## replace <ImageHere>; <MultiHere>; <FrameHere>; <FaceHere>; <AudioHere>
        cur_idx = 0
        new_input_embeds = []
        for cur_input_ids, cur_input_embeds in zip(input_ids, temp_input_embedding):
            # 🎯 Lexical信息直接作为文本嵌入，不需要特征替换
            for (patch_token_id, query_token_number, embeds) in [(self.FRAME_PATCH_TOKEN_ID, self.num_video_query_token, frame_llms),
                                                                (self.FACE_PATCH_TOKEN_ID,  self.num_video_query_token, face_llms),
                                                                (self.AUDIO_PATCH_TOKEN_ID, self.num_audio_query_token, audio_llms),
                                                                (self.MULTI_PATCH_TOKEN_ID, self.num_multi_query_token, multi_llms),
                                                                (self.IMAGE_PATCH_TOKEN_ID, self.num_image_query_token, image_llms),
                                                                ]:
                if (cur_input_ids == patch_token_id).sum() != 0:
                    if embeds is None:
                        # 详细调试信息
                        token_names = {
                            self.FRAME_PATCH_TOKEN_ID: "FRAME",
                            self.FACE_PATCH_TOKEN_ID: "FACE", 
                            self.AUDIO_PATCH_TOKEN_ID: "AUDIO",
                            self.MULTI_PATCH_TOKEN_ID: "MULTI",
                            self.IMAGE_PATCH_TOKEN_ID: "IMAGE",
                        }
                        token_name = token_names.get(patch_token_id, f"UNKNOWN({patch_token_id})")
                        print(f"❌ {token_name} embeds is None for sample {cur_idx}")
                        print(f"   frame_llms: {frame_llms is not None if 'frame_llms' in locals() else 'undefined'}")
                        print(f"   face_llms: {face_llms is not None if 'face_llms' in locals() else 'undefined'}")
                        print(f"   audio_llms: {audio_llms is not None if 'audio_llms' in locals() else 'undefined'}")
                        print(f"   multi_llms: {multi_llms is not None if 'multi_llms' in locals() else 'undefined'}")
                        print(f"   image_llms: {image_llms is not None if 'image_llms' in locals() else 'undefined'}")
                    assert embeds is not None, f'Some input info is missing: {token_names.get(patch_token_id, f"UNKNOWN({patch_token_id})")} embeds is None.'
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
            
            new_input_embeds.append(cur_input_embeds)
            cur_idx += 1
        inputs_embeds = torch.stack(new_input_embeds, dim=0)

        '''
        Notation：比如 ChatGLM 这种模型是不支持 inputs_embeds 输入的，所以无法采用这种方式去计算loss

        inputs_embeds 前面加个 <bos> 才能实现自回归的预测
        inputs_embeds:  [<bos>###Human: <Video> <ImageHere>*32 </Video> xxx###Assistant: {target}<eos=pad><pad><pad><pad><pad><pad><pad>]
        attention_mask: [1, 1, ...                                                             1,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
        targets:        [-100......,                               .............., -100, {target}<eos=pad>, -100,..                 -100]
        '''
        targets = samples['labels']
        attention_mask = samples['attention_masks']
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets)
        loss = outputs.loss
        return {"loss": loss}


    @classmethod
    def from_config(cls, cfg):
        
        # 这几个参数必须设置，默认情况下为：
        # => llama_model: 'Vicuna' acoustic_encoder: "IMAGEBIND" visual_encoder: "EVA_CLIP_G"
        visual_encoder_name   = cfg.get("visual_encoder", "xxx") # Image Encoder
        acoustic_encoder_name = cfg.get("acoustic_encoder", "xxx") # Audio Encoder
        llama_model_name      = cfg.get("llama_model", "xxx") # LLaMA
        multi_fusion_type = cfg.get("multi_fusion_type", "attention")
        video_fusion_type = cfg.get("video_fusion_type", "qformer")
        audio_fusion_type = cfg.get("audio_fusion_type", "qformer")
        image_fusion_type = cfg.get("image_fusion_type", "token")
        au_fusion_type = cfg.get("au_fusion_type", "mean")  # AU默认使用mean融合

        # Audio/Video Q-Former
        frozen_video_Qformer    = cfg.get("frozen_video_Qformer", False)
        frozen_video_proj = cfg.get("frozen_video_proj", False)
        frozen_audio_Qformer    = cfg.get("frozen_audio_Qformer", False)
        frozen_audio_proj = cfg.get("frozen_audio_proj", False)
        frozen_multi_Qformer    = cfg.get("frozen_multi_Qformer", False)
        frozen_multi_llama_proj = cfg.get("frozen_multi_llama_proj", False)
        frozen_au_proj = cfg.get("frozen_au_proj", False)  # AU投影层冻结参数
        frozen_llm = cfg.get("frozen_llm", False)
        lora_r = cfg.get("lora_r", 16)

        # 这几个参数是默认的
        num_audio_query_token = cfg.get("num_audio_query_token", 'xxx') # 8
        num_video_query_token = cfg.get("num_video_query_token", 'xxx') # 32
        num_multi_query_token = cfg.get("num_multi_query_token", 'xxx') # 16
        num_image_query_token = cfg.get("num_image_query_token", 'xxx') # 32
        num_au_query_token = cfg.get("num_au_query_token", 8)  # AU query token数量，默认8个
        
        # 预提取模式下跳过编码器加载
        skip_encoders = cfg.get("skip_encoders", False)
        preextracted_visual_dim = cfg.get("preextracted_visual_dim", 768)
        preextracted_acoustic_dim = cfg.get("preextracted_acoustic_dim", 1024)

        model = cls(
            visual_encoder_name=visual_encoder_name,
            acoustic_encoder_name=acoustic_encoder_name,
            llama_model_name=llama_model_name,
            frozen_video_proj=frozen_video_proj,
            frozen_audio_proj=frozen_audio_proj,
            frozen_multi_llama_proj=frozen_multi_llama_proj,
            frozen_video_Qformer=frozen_video_Qformer,
            frozen_audio_Qformer=frozen_audio_Qformer,
            frozen_multi_Qformer=frozen_multi_Qformer,
            frozen_llm=frozen_llm,
            lora_r=lora_r,
            num_video_query_token=num_video_query_token,
            num_audio_query_token=num_audio_query_token,
            num_multi_query_token=num_multi_query_token,
            num_image_query_token=num_image_query_token,
            num_au_query_token=num_au_query_token,
            frozen_au_proj=frozen_au_proj,
            multi_fusion_type=multi_fusion_type,
            video_fusion_type=video_fusion_type,
            audio_fusion_type=audio_fusion_type,
            image_fusion_type=image_fusion_type,
            au_fusion_type=au_fusion_type,
            skip_encoders=skip_encoders,
        )
        
        # 设置预提取特征维度
        if skip_encoders:
            model.preextracted_visual_dim = preextracted_visual_dim
            model.preextracted_acoustic_dim = preextracted_acoustic_dim

        # priority: ckpt < ckpt_2 < ckpt_3 
        # => 后面的预训练权重会覆盖前面的预训练权重，所有模型加载的顺序是有讲究的
        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            print("Load first Checkpoint: {}".format(ckpt_path))
            # ckpt = torch.load(ckpt_path, map_location="cpu")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt['model'], strict=False)
            
        ckpt_path_2 = cfg.get("ckpt_2", "")
        if ckpt_path_2:
            print("Load second Checkpoint: {}".format(ckpt_path_2))
            # ckpt = torch.load(ckpt_path_2, map_location="cpu")
            ckpt = torch.load(ckpt_path_2, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt['model'], strict=False)

        ckpt_path_3 = cfg.get("ckpt_3", "")
        if ckpt_path_3:
            print("Load third Checkpoint: {}".format(ckpt_path_3))
            # ckpt = torch.load(ckpt_path_3, map_location="cpu")
            ckpt = torch.load(ckpt_path_3, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt['model'], strict=False)
        
        return model

    