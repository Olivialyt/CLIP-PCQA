import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
import copy
import math
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
clip_vis = 'ViT-B/16'

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]).to(x.device), tokenized_prompts.argmax(dim=-1).to(x.device)] @ self.text_projection.to(x.device)
        
        return x


class PromptLearner(nn.Module):
    def __init__(self, device, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.n_ctx
        ctx_init = args.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.device = device

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if args.csc:
                # print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                # print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).to(self.device)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = args.class_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CLIPPCQA_Net(nn.Module):
    def __init__(self, device, args, score_list, quality_classes):
        super(CLIPPCQA_Net, self).__init__()
        self.device = device
        self.score_list = score_list
        self.num_quality_classes = len(quality_classes)

        self.clip_texture, _ = clip.load(clip_vis)
        self.clip_texture = self.clip_texture.to(torch.float32)

        self.clip_depth, _ = clip.load(clip_vis)
        self.clip_depth = self.clip_depth.to(torch.float32)

        self.dtype = self.clip_texture.dtype

        self.quality_prompt_learner = PromptLearner(self.device, args, quality_classes, self.clip_texture)
        self.quality_tokenized_prompts = self.quality_prompt_learner.tokenized_prompts
        self.quality_text_encoder = TextEncoder(self.clip_texture)
        self.quality_logit_scale = self.clip_texture.logit_scale

    def forward(self, texture_imgs, depth_imgs):
        
        # preprocess
        batch_size, num_views, channels, image_height, image_width = texture_imgs.shape
        texture_imgs = texture_imgs.reshape(-1, channels, image_height, image_width).type(self.dtype)
        depth_imgs = depth_imgs.reshape(-1, channels, image_height, image_width).type(self.dtype)

        # feature extraction
        texture_f = self.clip_texture.encode_image(texture_imgs)  # B*num_views, patches, C;
        texture_f = texture_f / texture_f.norm(dim=-1, keepdim=True)
        _, patches_num, C = texture_f.shape

        depth_f = self.clip_depth.encode_image(depth_imgs)
        depth_f = depth_f / depth_f.norm(dim=-1, keepdim=True) 

        # normalize
        image_f = (texture_f + depth_f) / 2
        image_f = (image_f / image_f.norm(dim=-1, keepdim=True)).mean(dim=1, keepdim=False).reshape(batch_size, num_views, C).mean(dim=1, keepdim=False)

        # prompts
        quality_prompts = self.quality_prompt_learner()
        quality_tokenized_prompts = self.quality_tokenized_prompts
        quality_text_f = self.quality_text_encoder(quality_prompts, quality_tokenized_prompts)
        quality_text_f = quality_text_f / quality_text_f.norm(dim=-1, keepdim=True)
        quality_logit_scale = self.quality_logit_scale.exp()
        quality_logits = quality_logit_scale * image_f @ quality_text_f.t() # B, quality_classes

        # quality_cdf and quality_score
        pred_distribution = F.softmax(quality_logits, dim=1).to(self.device)
        pred_CDF = torch.cumsum(pred_distribution, dim=1)
        bin_tensor = torch.tensor(self.score_list).to(self.device).reshape(1, self.num_quality_classes)
        quality_score = (pred_distribution * bin_tensor).sum(1, keepdim=True).to(self.device)

        texture_f = texture_f.flatten(start_dim=1)
        depth_f = depth_f.flatten(start_dim=1)
        return texture_f, depth_f, quality_score, pred_distribution, pred_CDF
