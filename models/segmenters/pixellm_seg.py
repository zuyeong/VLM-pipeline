import sys
import os
import torch
import numpy as np
import cv2
from PIL import Image

PIXELLM_DIR = "/home/cjy/workspace/PixelLM"
if PIXELLM_DIR not in sys.path:
    sys.path.append(PIXELLM_DIR)

from chat import parse_args, preprocess
from model.PixelLM import PixelLMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from core.base_models import BaseSegmentationModel

class PixelLMSegmentationModel(BaseSegmentationModel):
    def __init__(self, version="./PixelLM-13B/hf_model"):
        self.version = version
        self.args = None
        self.tokenizer = None
        self.model = None
        self.clip_image_processor = None
        self.transform = None
        self.transform_clip = None

    def load_model(self, **kwargs):
        print(f"[PixelLM] Loading Model: {self.version}")
        old_cwd = os.getcwd()
        os.chdir(PIXELLM_DIR) # required for config loads
        
        try:
            args_list = [
                "--version", self.version,
                "--precision", "bf16",
                "--seg_token_num", "3",
                "--pad_train_clip_images",
                "--preprocessor_config", "./configs/preprocessor_448.json",
                "--resize_vision_tower",
                "--resize_vision_tower_size", "448",
                "--vision-tower", "openai/clip-vit-large-patch14-336",
                "--vision_tower_for_mask",
                "--image_feature_scale_num", "2",
                "--separate_mm_projector"
            ]
            self.args = parse_args(args_list)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.version, cache_dir=None, model_max_length=self.args.model_max_length,
                padding_side="right", use_fast=False
            )
            self.tokenizer.pad_token = self.tokenizer.unk_token
            if self.args.seg_token_num * self.args.image_feature_scale_num == 1:
                num_added_tokens = self.tokenizer.add_tokens("[SEG]")
                self.args.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
            else:
                new_tokens = ["[SEG{}]".format(i) for i in range(self.args.seg_token_num * self.args.image_feature_scale_num)]
                num_added_tokens = self.tokenizer.add_tokens(new_tokens)
                self.args.seg_token_idx = [self.tokenizer(token, add_special_tokens=False).input_ids[0] for token in new_tokens]

            torch_dtype = torch.bfloat16
            kwargs_model = {
                "torch_dtype": torch_dtype, "device_map": "auto", "seg_token_num": self.args.seg_token_num,
                "image_feature_scale_num": self.args.image_feature_scale_num, "pad_train_clip_images": self.args.pad_train_clip_images,
                "resize_vision_tower": self.args.resize_vision_tower, "resize_vision_tower_size": self.args.resize_vision_tower_size,
                "vision_tower_for_mask": self.args.vision_tower_for_mask, "separate_mm_projector": self.args.separate_mm_projector,
            }

            self.model = PixelLMForCausalLM.from_pretrained(
                self.args.version, low_cpu_mem_usage=True, vision_tower=self.args.vision_tower, seg_token_idx=self.args.seg_token_idx, **kwargs_model
            )
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
            self.model.config.bos_token_id = self.tokenizer.bos_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

            self.model.get_model().initialize_vision_modules(self.model.get_model().config)
            vision_tower = self.model.get_model().get_vision_tower()
            vision_tower.to(dtype=torch_dtype)
            self.model = self.model.bfloat16()
            vision_tower.to(device=self.args.local_rank)

            self.clip_image_processor = CLIPImageProcessor.from_pretrained(self.model.config.vision_tower) if self.args.preprocessor_config == '' else CLIPImageProcessor.from_pretrained(self.args.preprocessor_config)
            self.transform = ResizeLongestSide(self.args.image_size)
            if self.args.pad_train_clip_images:
                self.transform_clip = ResizeLongestSide(self.clip_image_processor.size['shortest_edge'])
            self.model.eval()
            print("[PixelLM] Model loaded successfully.")
        finally:
            os.chdir(old_cwd)

    def predict(self, image: Image.Image, prompt: str) -> Image.Image:
        print(f"[PixelLM] User prompt: {prompt}")
        
        conv = conversation_lib.conv_templates[self.args.conv_type].copy()
        conv.messages = []
        prompt_txt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if self.args.use_mm_start_end:
            replace_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
            prompt_txt = prompt_txt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt_txt)
        conv.append_message(conv.roles[1], "")
        prompt_txt = conv.get_prompt()

        image_np = np.array(image.convert("RGB"))
        original_size_list = [image_np.shape[:2]]
        
        if self.args.pad_train_clip_images:
            image_clip = self.transform_clip.apply_image(image_np)
            clip_resize = image_clip.shape[:2]
            image_clip = preprocess(torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(), img_size=self.clip_image_processor.size['shortest_edge'])
            image_clip = image_clip.unsqueeze(0).cuda()
        else:
            image_clip = (self.clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda())
            clip_resize = image_clip.shape[-2:]
            
        image_clip = image_clip.bfloat16()

        image_transformed = self.transform.apply_image(image_np)
        resize_list = [image_transformed.shape[:2]]
        clip_resize = [clip_resize]

        image_t = (preprocess(torch.from_numpy(image_transformed).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda())
        image_t = image_t.bfloat16()

        input_ids = tokenizer_image_token(prompt_txt, self.tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids, pred_masks, _, _ = self.model.evaluate(
                image_clip, image_t, input_ids, resize_list, clip_resize_list=clip_resize,
                original_size_list=original_size_list, max_new_tokens=512, tokenizer=self.tokenizer
            )

        final_mask = np.zeros((original_size_list[0][0], original_size_list[0][1]), dtype=bool)
        for _pred_mask in pred_masks:
            if _pred_mask.shape[0] == 0:
                continue
            for pred_mask in _pred_mask:
                pm = pred_mask.float().detach().cpu().numpy()
                final_mask = final_mask | (pm > 0)
                
        final_mask_img = (final_mask.astype(np.uint8) * 255)
        return Image.fromarray(final_mask_img, mode="L")
