import sys
import os
import torch
import numpy as np
import cv2
from PIL import Image

GLAMM_DIR = "/home/cjy/workspace/groundingLMM"
if GLAMM_DIR not in sys.path:
    sys.path.append(GLAMM_DIR)

from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from app import parse_args, setup_tokenizer_and_special_tokens, initialize_model, prepare_model_for_inference, grounding_enc_processor
from core.base_models import BaseSegmentationModel

class GlammSegmentationModel(BaseSegmentationModel):
    def __init__(self, version="/home/cjy/workspace/GLaMM-FullScope", image_size=1024):
        self.version = version
        self.image_size = image_size
        self.args = None
        self.tokenizer = None
        self.model = None
        self.global_enc_processor = None
        self.transform = None

    def load_model(self, **kwargs):
        print(f"[GLaMM] Loading Model: {self.version}")
        self.args = parse_args(["--version", self.version, "--image_size", str(self.image_size)])
        self.tokenizer = setup_tokenizer_and_special_tokens(self.args)
        self.model = initialize_model(self.args, self.tokenizer)
        self.model = prepare_model_for_inference(self.model, self.args)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(self.model.config.vision_tower)
        self.transform = ResizeLongestSide(self.args.image_size)
        self.model.eval()
        print("[GLaMM] Model loaded successfully.")

    def predict(self, image: Image.Image, prompt: str) -> Image.Image:
        print(f"[GLaMM] User prompt: {prompt}")
        
        conv = conversation_lib.conv_templates[self.args.conv_type].copy()
        conv.messages = []
        
        prompt_txt = f"The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture." + "\n" + prompt
        if self.args.use_mm_start_end:
            replace_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
            prompt_txt = prompt_txt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt_txt)
        conv.append_message(conv.roles[1], "")
        prompt_txt = conv.get_prompt()

        image_np = np.array(image.convert("RGB"))
        original_size_list = [image_np.shape[:2]]

        global_enc_image = self.global_enc_processor.preprocess(
            image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).to(self.model.device)
        global_enc_image = global_enc_image.bfloat16()

        image_transformed = self.transform.apply_image(image_np)
        resize_list = [image_transformed.shape[:2]]
        grounding_enc_image = (grounding_enc_processor(torch.from_numpy(image_transformed).permute(2, 0, 1).
                                                       contiguous()).unsqueeze(0).to(self.model.device))
        grounding_enc_image = grounding_enc_image.bfloat16()

        input_ids = tokenizer_image_token(prompt_txt, self.tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(self.model.device)

        with torch.inference_mode():
            output_ids, pred_masks = self.model.evaluate(
                global_enc_image, grounding_enc_image, input_ids, resize_list, original_size_list, max_tokens_new=512,
                bboxes=None)

        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        text_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)

        if "[SEG]" in text_output and len(pred_masks) > 0 and pred_masks[0].shape[0] > 0:
            pred_mask = pred_masks[0].detach().cpu().numpy()
            mask_list = [pred_mask[i] for i in range(pred_mask.shape[0])]
            
            seg_count = text_output.count("[SEG]")
            mask_list = mask_list[-seg_count:]
            
            final_mask = np.zeros(mask_list[0].shape, dtype=bool)
            for curr_mask in mask_list:
                final_mask = final_mask | (curr_mask > 0)
                
            final_mask_img = (final_mask.astype(np.uint8) * 255)
        else:
            final_mask_img = np.zeros((original_size_list[0][0], original_size_list[0][1]), dtype=np.uint8)
            
        mask_image = Image.fromarray(final_mask_img, mode="L")
        return mask_image
