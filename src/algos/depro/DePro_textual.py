import sys
import os

code_path = '/home/user/Code/DePro_SIGIR'  # e.g. '/home/username/ProS'
sys.path.append(code_path)
sys.path.append(os.path.join(code_path, "../.."))
import torch
import torch.nn as nn
from clip import clip
from clip.model import CLIP, VisionTransformer
import math
import torch.nn as nn
from PIL import Image
from functools import reduce
from operator import mul
from src.utils import utils
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import copy

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)  # 300, 77, 512
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class FixedEmbeddings():
    def __init__(self, cfg, classnames, clip_model, device):
        self.device = device
        self.clip_model = clip_model
        clip_imsize = clip_model.visual.input_resolution  # 输入图片大小 # 224
        cfg_imsize = cfg.image_size  # 设定的输入图片大小
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        prompt_prefix = "a photo of a"
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        self.text_features = self.clip_model.encode_text(self.tokenized_prompts).clone().detach()

    def return_fixed_embeddings(self):
        return self.text_features


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        n_cls = len(classnames)  # 400
        n_ctx = cfg.textNumTokens  # number of context tokens 16
        dtype = clip_model.dtype  # float32
        ctx_dim = clip_model.ln_final.weight.shape[0]  # LayerNorm第0维，前一层的输出维度 512
        clip_imsize = clip_model.visual.input_resolution  # 输入图片大小 # 224
        cfg_imsize = cfg.image_size  # 设定的输入图片大小
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # random initialization
        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).to(device)  # 16, 512
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)  # 生成n_ctx个 X，eg. X X X X X

        print(f'Initial context: "{prompt_prefix}"')  # 'X X X X X X X X X X X X X X X X'
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized 16,512

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]  # 400，类名的长度
        prompts = [prompt_prefix + " " + name + "." for name in classnames]  # xxxxxxxxx classname .

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(
            device)  # 将prompt中的句子的每个单词转换成字典中的数字，长度固定为77，多的用0补齐, [400,77]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # 400, 77, 512

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # 400,16,512

        prefix = self.token_prefix  # 400,1,512
        suffix = self.token_suffix  # 400,60,512

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts  # 400,77,512


class DomainPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model: CLIP, device):
        super().__init__()
        n_cls = len(classnames)  # 300
        n_ctx = cfg.textNumTokens  # number of context tokens 16
        dtype = clip_model.dtype  # float32
        self.ctx_dim = clip_model.ln_final.weight.shape[0]  # LayerNorm第0维，前一层的输出维度 512
        clip_imsize = clip_model.visual.input_resolution  # 输入图片大小 # 224
        cfg_imsize = cfg.image_size  # 设定的输入图片大小
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # random initialization
        # print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=dtype).to(device)  # 16, 512
        nn.init.normal_(ctx_vectors, std=0.02)
        # prompt_prefix = " ".join(["X"] * n_ctx) # 生成n_ctx个 X，eg. X X X X X

        # print(f'Initial context: "{prompt_prefix}"') # 'X X X X X X X X X X X X X X X X'
        # print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized 16,512 

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]  # 300，类名的长度, 有的長度是2，比如aircraft carrier
        prompts = ["a photo of " + name + " from " + "X " * n_ctx + "domain." for name in
                   classnames]  # xxxxxxxxx classname .
        self.prefix_index = [length + 5 for length in name_lens]  # SOS a photo of classname from
        print("Text Prompt Example:" + prompts[0])
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(
            device)  # 将prompt中的句子的每个单词转换成字典中的数字，长度固定为77，多的用0补齐, [300,77]

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # 300, 77, 512
        self.n_cls = n_cls
        self.n_ctx = n_ctx

        self.register_buffer("origin_text_embedding", embedding)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # 300,16,512

        prompts = [torch.cat([self.origin_text_embedding[i, :self.prefix_index[i]], ctx[i],
                              self.origin_text_embedding[i, self.prefix_index[i] + self.n_ctx:]], dim=0).view(1, -1,
                                                                                                              self.ctx_dim)
                   for i in range(self.n_cls)]
        prompts = torch.cat(prompts, dim=0)
        return prompts  # 300,77,512



