import sys
import os
import torch
import numpy as np
import cv2
from PIL import Image

INSTRUCTSEG_DIR = "/home/cjy/workspace/InstructSeg"
if INSTRUCTSEG_DIR not in sys.path:
    sys.path.append(INSTRUCTSEG_DIR)

from instructseg.utils.builder import load_pretrained_model
from instructseg.utils import conversation as conversation_lib
from instructseg.datasets.InstructSegDatasets import DataCollatorForCOCODatasetV2
from transformers import SiglipImageProcessor
from instructseg.utils.constants import REFER_TOKEN_INDEX
from batch_inference_instructseg import DataArguments, preprocess_llama2, preprocess_reasoning_instruction, parse_outputs
from core.base_models import BaseSegmentationModel

class InstructSegModel(BaseSegmentationModel):
    def __init__(self, model_path="model/InstructSeg"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.data_collator = None
        self.data_args = None

    def load_model(self, **kwargs):
        print(f"[InstructSeg] Loading Model: {self.model_path}")
        self.data_args = DataArguments()
        model_p = os.path.expanduser(self.model_path)
        if not os.path.isabs(model_p):
             model_p = os.path.join(INSTRUCTSEG_DIR, model_p)
             
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
            model_p, model_args=self.data_args, mask_config=self.data_args.mask_config, device='cuda'
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(dtype=torch.float32, device=self.device)
        self.model.eval()

        self.data_args.image_processor = self.image_processor
        self.data_args.is_multimodal = True
        conversation_lib.default_conversation = conversation_lib.conv_templates[self.data_args.version]
        clip_image_processor = SiglipImageProcessor.from_pretrained(self.data_args.vision_tower)
        self.data_collator = DataCollatorForCOCODatasetV2(tokenizer=self.tokenizer, clip_image_processor=clip_image_processor)
        print("[InstructSeg] Model loaded successfully.")

    def predict(self, image: Image.Image, prompt: str) -> Image.Image:
        import tempfile
        print(f"[InstructSeg] User prompt: {prompt}")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_in:
            image.convert("RGB").save(temp_in.name)
            image_path = temp_in.name
            
        try:
            data_dict = {}
            data_dict['file_name'] = image_path
            data_dict = self.image_processor.preprocess(data_dict, mask_format='polygon')
            
            token_refer_id, gt_ref_id = preprocess_reasoning_instruction(prompt, self.tokenizer, sum_ref_gt=None)

            prefix_inst = 'This is an image <temporal>\n<image>\n, please doing Reasoning Segmentation according to the following instruction:'
            sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <gt><seg>'}]]
            
            text_dict = preprocess_llama2(sources, self.tokenizer, ref_id_gt=gt_ref_id)
            input_ids = text_dict['input_ids'][0]
            
            refer_embedding_indices = torch.zeros_like(input_ids)
            refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1

            data_dict['input_ids'] = input_ids
            data_dict['labels'] = text_dict['labels'][0]
            data_dict['dataset_type'] = 'reason_seg'
            data_dict['token_refer_id'] = token_refer_id
            data_dict['refer_embedding_indices'] = refer_embedding_indices
            
            batch = self.data_collator([data_dict])
            
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items() if k != 'seg_info'}
            inputs['seg_info'] = batch['seg_info']
            inputs['token_refer_id'] = [ids.to(self.device) for ids in inputs['token_refer_id']]
            
            with torch.no_grad():
                outputs = self.model.eval_seg(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    images=inputs['images'].float(),
                    images_clip=inputs['images_clip'].float(),
                    seg_info=inputs['seg_info'],
                    token_refer_id=inputs['token_refer_id'],
                    refer_embedding_indices=inputs['refer_embedding_indices'],
                    labels=inputs['labels']
                )
            
            cur_res = parse_outputs(outputs)
            
            scores = cur_res[0]['scores']
            preds = cur_res[0]['pred']
            topk_scores, idx_top = torch.topk(scores, 1)
            topk_preds = preds[idx_top, :]
            pred_mask = topk_preds[0]
            
            pred_mask_np = pred_mask.detach().cpu().numpy()
            pred_mask_np = (pred_mask_np > 0)
            
            orig_h = batch['seg_info'][0]['height']
            orig_w = batch['seg_info'][0]['width']
            pred_mask_np = cv2.resize(pred_mask_np.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            mask_image = Image.fromarray((pred_mask_np * 255).astype(np.uint8), mode="L")
            return mask_image
            
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)
