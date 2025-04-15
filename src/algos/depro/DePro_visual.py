import torch
from clip.model import CLIP, VisionTransformer
import math
import torch.nn as nn
from PIL import Image
from functools import reduce
from operator import mul
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
_tokenizer = _Tokenizer()


class VisualGenerator(nn.Module):
    def __init__(self, config, model, device, dict_doms, dict_clss):
        super(VisualGenerator, self).__init__()
        self.config = config
        self.no_domain_specific_trick = not self.config.domain_specific_trick
        self.dict_doms = dict_doms
        self.dict_clss = dict_clss
        self.dom_num_tokens = len(self.dict_doms)
        self.cls_num_tokens = len(self.dict_clss)
        self.device = device
        self.init_method = config.first_init
        self.disable_pre_ln = self.config.disable_first_pre_ln

        self.conv1 = model.visual.conv1
        self.class_embedding = model.visual.class_embedding  # self.feature_template = clip.visual.class_embedding
        self.positional_embedding = model.visual.positional_embedding
        self.ln_pre = model.visual.ln_pre

        if self.config.generator_layer > 0:
            self.vit = VisionTransformer(224, 32, 768, self.config.generator_layer, 12, 512)
            width = self.vit.conv1.out_channels
            scale = width ** -0.5
            self.proj = nn.Parameter(scale * torch.randn(768, 768))

            sp_cls_prompt_dim = width
            sp_dom_prompt_dim = width
            patch_size = self.conv1.kernel_size
            self.dom_num_tokens = len(self.dict_doms)
            self.cls_num_tokens = len(self.dict_clss)
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + sp_dom_prompt_dim))  # noqa
            self.specific_domain_prompts = nn.Parameter(
                torch.zeros(self.dom_num_tokens, sp_dom_prompt_dim))  # layer, num_token, prompt_dim
            nn.init.uniform_(self.specific_domain_prompts.data, -val, val)

            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + sp_cls_prompt_dim))  # noqa
            self.specific_class_prompts = nn.Parameter(
                torch.zeros(self.cls_num_tokens, sp_cls_prompt_dim))  # layer, num_token, prompt_dim
            nn.init.uniform_(self.specific_class_prompts.data, -val, val)

        self.num_tokens = self.config.vptNumTokens  # "10"  # number of prompted tokens

    def incorporate_prompt(self, x, prompt_embeddings, dom_id=None, cls_id=None, sdp=None, scp=None):
        B = x.shape[0]
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # 65 768 49
        x = x.permute(0, 2, 1)
        x = x + self.positional_embedding[1:].to(x.dtype)  # [65, 50, 678] + [50 ,768]
        if dom_id is None or self.no_domain_specific_trick:
            x = torch.cat((  # no need for class token
                prompt_embeddings.expand(B, -1, -1),
                x
            ), dim=1)
        else:
            sdp = self.specific_domain_prompts
            scp = self.specific_class_prompts
            if self.training:
                dom_mask = torch.zeros(B, sdp.shape[0], sdp.shape[1]).type(torch.bool).to(
                    self.device)
                cls_mask = torch.zeros(B, scp.shape[0], scp.shape[1]).type(torch.bool).to(
                    self.device)
                dom_mask[range(B), dom_id, :] = 1  # batchsize, dom_num, 768
                cls_mask[range(B), cls_id, :] = 1  # batchsize, dom_num, 768
                sdp = sdp.expand(B, -1, -1).masked_fill(dom_mask, 0)  #
                scp = scp.expand(B, -1, -1).masked_fill(cls_mask, 0)  #
            else:
                sdp = sdp.expand(B, -1, -1)
                scp = scp.expand(B, -1, -1)

            x = torch.cat((  # no need for class token
                prompt_embeddings.expand(B, -1, -1),
                sdp, scp, x
            ), dim=1)
        return x

    def forward(self, x, prompt_embeddings, dom_id=None, cls_id=None, sdp=None, scp=None):
        B = x.shape[0]
        if self.config.generator_layer == 0:
            return prompt_embeddings.expand(B, -1, -1)
        x = self.incorporate_prompt(x, prompt_embeddings, dom_id, cls_id, sdp, scp)
        if not self.disable_pre_ln:
            x = self.ln_pre(x)  # should exist?
        x = x.permute(1, 0, 2)
        x = self.vit.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.vit.ln_post(x[:, :self.num_tokens, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


class VisualUDPGenerator(nn.Module):
    def __init__(self, config, promptLeaner, device):
        super(VisualUDPGenerator, self).__init__()
        self.VL_independent = config.VL_independent
        self.config = config
        self.promptLearner = promptLeaner
        self.device = device

        width = 768
        self.num_tokens = self.config.vptNumTokens  # "16"  # number of prompted tokens

        scale = width ** -0.5

        if not self.VL_independent:
            # whether to build connection between the language and image modality
            scale_2 = 512 ** -0.5
            self.prompt_proj = nn.Parameter(scale_2 * torch.randn(512, 768))
            self.text_embeddings = self.promptLearner.ctx  # 16, 512
            self.proj = nn.Parameter(scale * torch.randn(768, 768))  #
        else:
            self.prompt_embeddings = nn.Parameter(scale * torch.randn(1, self.num_tokens, width))

    def forward(self, x):
        B = x.shape[0]  # batch size
        if not self.VL_independent:
            return (self.text_embeddings @ self.prompt_proj).expand(B, -1, -1)
        else:
            return self.prompt_embeddings.expand(B, -1, -1)


class PromptedViT(nn.Module):
    def __init__(self, config, model: CLIP, device, dict_doms, dict_clss):
        super(PromptedViT, self).__init__()
        self.config = config
        self.init_method = config.second_init
        self.out_order = 0 if config.ln_trick else 1
        self.dict_doms = dict_doms
        self.dict_clss = dict_clss
        self.conv1 = model.visual.conv1
        width = self.conv1.out_channels
        self.class_embedding = model.visual.class_embedding  # self.feature_template = clip.visual.class_embedding
        self.positional_embedding = model.visual.positional_embedding  # self.clip_positional_embedding = clip.visual.positional_embedding  # 768
        self.ln_pre = model.visual.ln_pre
        self.transformer = model.visual.transformer
        self.ln_post = model.visual.ln_post
        self.proj = model.visual.proj  # self.feature_proj = clip.visual.proj

        patch_size = self.conv1.kernel_size
        self.num_tokens = self.config.vptNumTokens  # "10"  # number of prompted tokens

        prompt_dim = width
        self.prompt_proj = nn.Identity()

        # xavier_uniform initialization
        if self.init_method == "xavier":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, self.num_tokens, prompt_dim))  # layer, num_token, prompt_dim
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        else:
            scale = width ** -0.5
            self.prompt_embeddings = nn.Parameter(scale * torch.randn(1, self.num_tokens, width))

        self.meta_net = VisualGenerator(self.config, model, device, dict_doms, dict_clss)

    def no_incorporate(self, x):
        B = x.shape[0]  # batch size
        # after CLS token, all before image patches
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # 65 768 49
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)  # [65, 50, 678] + [50 ,768]
        return x

    def incorporate_visual_prompt(self, x, visual_prompts):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]  # batch size
        # after CLS token, all before image patches
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # 65 768 49
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)  # [65, 50, 678] + [50 ,768]

        x = torch.cat((
            x[:, :1, :],  # CLS token
            visual_prompts,
            x[:, 1:, :]
        ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        return x

    def incorporate_visual_text_prompt(self, x, visual_prompts, text_prompts):
        if visual_prompts is None and text_prompts is None:
            return self.no_incorporate(x)
        if text_prompts is None:
            return self.incorporate_visual_prompt(x, visual_prompts)
        if visual_prompts is None:
            return self.incorporate_visual_prompt(x, text_prompts)

        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]  # batch size
        # after CLS token, all before image patches
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # 65 768 49
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)  # [65, 50, 678] + [50 ,768]

        x = torch.cat((
            x[:, :1, :],  # CLS token
            text_prompts,
            visual_prompts,
            x[:, 1:, :]
        ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        return x

    def vit(self, x):
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, self.out_order, :])  # b, 2, dim
        if self.proj is not None:
            x = x @ self.proj
        return x

    def forward(self, x, udps=None, dom_id=0, cls_id=0):
        cps = self.meta_net(x, self.prompt_embeddings, dom_id, cls_id)
        with torch.no_grad():
            x0 = self.no_incorporate(x)
        x1 = self.incorporate_visual_text_prompt(x, cps, None)
        x2 = self.incorporate_visual_text_prompt(x, cps, udps)

        with torch.no_grad():
            x0 = self.vit(x0)
        x1 = self.vit(x1)
        x2 = self.vit(x2)

        return [x0, x1, x2]
