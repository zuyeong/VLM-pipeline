import sys
import os
import torch
import numpy as np
from PIL import Image

READ_DIR = "/home/cjy/workspace/READ"
if READ_DIR not in sys.path:
    sys.path.append(READ_DIR)

from model.READ import load_pretrained_model_READ
from model.llava import conversation as conversation_lib
from utils import prepare_input
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from dataloaders.base_dataset import ImageProcessor
from dataloaders.utils import replace_image_tokens, tokenize_and_pad
from core.base_models import BaseSegmentationModel

class ReadSegmentationModel(BaseSegmentationModel):
    def __init__(self, pretrained_model_path="rui-qian/READ-LLaVA-v1.5-13B-for-ReasonSeg-testset", vision_tower="openai/clip-vit-large-patch14-336", model_max_length=2048, image_size=1024):
        self.pretrained_model_path = pretrained_model_path
        self.vision_tower_path = vision_tower
        self.model_max_length = model_max_length
        self.image_size = image_size
        
        self.tokenizer = None
        self.segmentation_lmm = None
        self.vision_tower = None
        self.img_processor = None

    def load_model(self, **kwargs):
        print(f"[ReadSeg] Loading Model: {self.pretrained_model_path}")
        (
            self.tokenizer,
            self.segmentation_lmm,
            self.vision_tower,
            context_len,
        ) = load_pretrained_model_READ(
            model_path=self.pretrained_model_path,
            vision_tower=self.vision_tower_path,
            model_max_length=self.model_max_length
        )
        self.tokenizer.padding_side = "left"
        self.img_processor = ImageProcessor(self.vision_tower.image_processor, self.image_size)
        print("[ReadSeg] Model loaded successfully.")

    def predict(self, image: Image.Image, prompt: str) -> Image.Image:
        import tempfile
        print(f"[ReadSeg] User prompt: {prompt}")
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_in:
            image.convert("RGB").save(temp_in.name)
            image_path = temp_in.name
            
        try:
            image_t, image_clip, sam_mask_shape = self.img_processor.load_and_preprocess_image(image_path)
            
            conv = conversation_lib.default_conversation.copy()
            question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            conversation_list = [conv.get_prompt()]
            
            mm_use_im_start_end = getattr(self.segmentation_lmm.config, "mm_use_im_start_end", False)
            if mm_use_im_start_end:
                conversation_list = replace_image_tokens(conversation_list)
                
            input_ids, _ = tokenize_and_pad(conversation_list, self.tokenizer, padding='left')
            
            input_dict = {
                "image_path": image_path,
                "images_clip": torch.stack([image_clip], dim=0),
                "images": torch.stack([image_t], dim=0),
                "input_ids": input_ids,
                "sam_mask_shape_list": [sam_mask_shape]
            }
            input_dict = prepare_input(input_dict, "bf16", is_cuda=True)
            
            with torch.inference_mode():
                output_ids, pred_masks, object_presence = self.segmentation_lmm.evaluate(
                    input_dict["images_clip"],
                    input_dict["images"],
                    input_dict["input_ids"],
                    input_dict["sam_mask_shape_list"],
                    max_new_tokens=self.model_max_length,
                )
                
            pred_mask = pred_masks[0].detach().cpu().numpy()
            pred_mask = (pred_mask > 0)
            
            mask_image = Image.fromarray((pred_mask * 255).astype(np.uint8), mode="L")
            return mask_image
            
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)
