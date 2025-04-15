
from src.algos.depro.DePro_textual import FixedEmbeddings, PromptLearner, DomainPromptLearner, \
    TextEncoder
from src.algos.depro.DePro_visual import PromptedViT, VisualUDPGenerator


import torch
from clip import clip
from clip.model import CLIP
import torch.nn as nn
from PIL import Image
from src.utils import utils
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
_tokenizer = _Tokenizer()


class depro(nn.Module):
    def __init__(self, cfg, dict_clss, dict_doms, device):
        super().__init__()
        self.cfg = cfg
        self.VL_independent = self.cfg.VL_independent
        self.dict_clss = dict_clss
        self.dict_doms = dict_doms
        self.dom_num_tokens = len(self.dict_doms)  # 5
        self.cls_num_tokens = len(self.dict_clss)  # 300
        self.device = device

        clip: CLIP = self.load_clip()

        # FOR TEXT
        self.text_encoder = None
        self.tokenized_prompts = None
        self.text_prompt_learner = None
        self.fixed_text_encoder = FixedEmbeddings(self.cfg, self.dict_clss.keys(), clip, device=self.device)

        if self.cfg.text == 'None' or self.cfg.textNumTokens == 0:
            self.text_encoder = FixedEmbeddings(self.cfg, self.dict_clss.keys(), clip, device=self.device)
        else:
            if self.cfg.text == "CoOp":
                self.text_prompt_learner = PromptLearner(self.cfg, self.dict_clss.keys(), clip, device)
            else:  # self.cfg.text == "DCoOp"
                self.text_prompt_learner = DomainPromptLearner(self.cfg, self.dict_clss.keys(), clip,
                                                               device)  # look here
            self.text_encoder = TextEncoder(clip)
            self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts

        self.visual_udp_generator = VisualUDPGenerator(self.cfg, self.text_prompt_learner, device)
        # for visual
        self.visual_encoder = PromptedViT(self.cfg, clip, device, dict_doms, dict_clss)

    def forward(self, image, domain_name=0, class_name=0, stage=0):  # bt 3, 244, 244
        cls_id = utils.numeric_classes(class_name, self.dict_clss)
        dom_id = utils.numeric_classes(domain_name, self.dict_doms)

        # text_forward
        text_features_fixed = self.fixed_text_encoder.return_fixed_embeddings()
        if self.cfg.text != 'None' and self.cfg.textNumTokens != 0:
            text_prompts = self.text_prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(text_prompts, tokenized_prompts)
        else:
            text_features = self.text_encoder.return_fixed_embeddings()

        text_features = [text_features_fixed, text_features]

        text_prompts = self.visual_udp_generator(image)

        # visual_forward
        image_features = self.visual_encoder(image, text_prompts, dom_id, cls_id)  # batch, 512

        image_features = [image_feature / image_feature.norm(dim=-1, keepdim=True) for image_feature in image_features]
        text_features = [text_feature / text_feature.norm(dim=-1, keepdim=True) for text_feature in text_features]
        return image_features, text_features

    def load_clip(self):
        backbone_name = self.cfg.clip_backbone
        print(f"=======load CLIP:{backbone_name}=========")
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
        # print(model_path) /home/user/.cache/clip/ViT-B-32.pt
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location=self.device).eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location=self.device)
        # print(self.device)
        # print(type(state_dict))
        model = clip.build_model(state_dict or model.state_dict())
        # print(model)
        return model.float().to(self.device)
