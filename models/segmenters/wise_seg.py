import sys
import os
import torch
import json
import re
import numpy as np
from PIL import Image

# WISE 경로를 시스템 패스에 추가하여 WISE 내부 모듈을 import할 수 있도록 함
WISE_DIR = "/home/cjy/workspace/WISE"
if WISE_DIR not in sys.path:
    sys.path.append(WISE_DIR)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sam2.sam2_image_predictor import SAM2ImagePredictor
from core.base_models import BaseSegmentationModel


# VR-7B 가중치 기본 경로 (로컬 디스크). HF Hub ID 도 허용한다.
DEFAULT_REASONING_MODEL_PATH = (
    "/home/cjy/workspace/WISE/VisionReasoner/pretrained_models/VisionReasoner-7B"
)


def parse_reasoning_output(output_text, x_factor, y_factor):
    """VR-7B (다중) 와 Seg-Zero-7B (단일) 출력을 모두 받아들이는 통합 파서.

    VR-7B 출력 (정상 경로):
        <answer>[{"bbox_2d":[x1,y1,x2,y2], "point_2d":[x,y]}, ...]</answer>
    Seg-Zero-7B 출력 (하위 호환):
        <answer>{"bbox":[x1,y1,x2,y2], "points_1":[x,y], "points_2":[x,y]}</answer>

    반환: (bboxes, points, think_text)
        - bboxes: list[[x1,y1,x2,y2]]  (원본 해상도)
        - points: list[[x,y]] (객체별 1점; 없으면 빈 list)
        - think_text: <think>...</think> 우선, 없으면 <answer> 이전 텍스트
    """
    bboxes = []
    points = []
    think_text = ""

    # 1) <answer>...</answer> 블록을 먼저 찾는다 (VR 정규 출력)
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", output_text, re.DOTALL)
    if answer_match:
        payload = answer_match.group(1).strip()
        try:
            data = json.loads(payload)
            # (a) VR 다중 객체: list of dict
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    bb = item.get("bbox_2d") or item.get("bbox")
                    if not bb or len(bb) != 4:
                        continue
                    bboxes.append([
                        int(round(bb[0] * x_factor)),
                        int(round(bb[1] * y_factor)),
                        int(round(bb[2] * x_factor)),
                        int(round(bb[3] * y_factor)),
                    ])
                    p = item.get("point_2d") or item.get("point")
                    if p and len(p) == 2:
                        points.append([
                            int(round(p[0] * x_factor)),
                            int(round(p[1] * y_factor)),
                        ])
            # (b) Seg-Zero 단일 객체 dict (하위 호환)
            elif isinstance(data, dict):
                bbox_key = next((k for k in data.keys() if "bbox" in k.lower()), None)
                if bbox_key and len(data[bbox_key]) == 4:
                    bb = data[bbox_key]
                    bboxes.append([
                        int(round(bb[0] * x_factor)),
                        int(round(bb[1] * y_factor)),
                        int(round(bb[2] * x_factor)),
                        int(round(bb[3] * y_factor)),
                    ])
                # 단일 객체에 points_1/points_2 가 있으면 첫 점 하나만 채택
                points_keys = [k for k in data.keys() if "point" in k.lower()]
                if points_keys:
                    p = data[points_keys[0]]
                    if isinstance(p, list) and len(p) == 2:
                        points.append([
                            int(round(p[0] * x_factor)),
                            int(round(p[1] * y_factor)),
                        ])
        except Exception as e:
            print(f"[WiseSeg] <answer> JSON 파싱 실패: {e}; payload={payload[:200]!r}")

    # 2) <answer> 가 없거나 비어있으면 free-form list/dict 한 번 더 시도
    if not bboxes:
        free_match = re.search(r"\[[^\[\]]*\]|\{[^{}]*\}", output_text, re.DOTALL)
        if free_match:
            try:
                data = json.loads(free_match.group(0))
                if isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        bb = item.get("bbox_2d") or item.get("bbox")
                        if not bb or len(bb) != 4:
                            continue
                        bboxes.append([
                            int(round(bb[0] * x_factor)),
                            int(round(bb[1] * y_factor)),
                            int(round(bb[2] * x_factor)),
                            int(round(bb[3] * y_factor)),
                        ])
                        p = item.get("point_2d") or item.get("point")
                        if p and len(p) == 2:
                            points.append([
                                int(round(p[0] * x_factor)),
                                int(round(p[1] * y_factor)),
                            ])
            except Exception:
                pass

    # think 추출: <think>...</think> 우선, 없으면 <answer> 앞 텍스트
    think_match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
    if think_match:
        think_text = think_match.group(1).strip()
    elif answer_match:
        think_text = output_text[:answer_match.start()].strip()

    return bboxes, points, think_text


class WiseSegmentationModel(BaseSegmentationModel):
    """
    WISE 코드 + VisionReasoner-7B 가중치 조합으로 다중 객체 reasoning 세그를 수행한다.

    설계 요지:
        - reasoning 모델은 VR-7B (Qwen2.5-VL 7B 기반, 다중 객체 GRPO 튜닝됨) 를 in-process 로드.
        - 프롬프트/few-shot 은 VR 의 DETECTION_TEMPLATE 형식 (list[{bbox_2d, point_2d}]).
        - SAM2 호출은 객체별로 반복, np.logical_or 로 단일 마스크 합성 → 하류 인페인터는 무수정.

    ⚠ 메모 B-9 (GPU 자원 영구 점유):
        load_model() 로 한 번 올린 VR-7B (~16GB) + SAM2-hiera-large (~0.5GB) 는 워커
        lifetime 내내 GPU 메모리에 상주한다. predict() 끝의 torch.cuda.empty_cache()
        는 PyTorch 캐시(미사용 메모리)만 OS 로 돌려주고 weights 참조는 풀지 않는다
        — 의도된 동작 (재로딩 비용 회피).

        부작용: GPU 를 다른 사용자/프로세스와 공유한다면 16GB 가 영구히 빠짐.
        해결책 후보:
          - 전용 GPU 분리 (CUDA_VISIBLE_DEVICES 로 격리)
          - MIG (A100 등)
          - 워커가 task 처리 안 할 때 model.cpu() 로 호스트 메모리로 swap-out
            (다음 task 호출 시 model.cuda() 로 swap-in — 호출당 ~1-2초 추가)
    """

    def __init__(self,
                 reasoning_model_path=DEFAULT_REASONING_MODEL_PATH,
                 segmentation_model_path="facebook/sam2-hiera-large",
                 device="cuda"):
        self.reasoning_model_path = reasoning_model_path
        self.segmentation_model_path = segmentation_model_path
        # 멀티 GPU 배치용. 'cuda' = 기본 GPU, 'cuda:1' = 2번째 GPU pin.
        # (prompt model 을 cuda:0 에 두고 VR-7B 를 cuda:1 로 분리하는 데 사용)
        self.device = device

        self.reasoning_model = None
        self.processor = None
        self.segmentation_model = None

        # VR 의 DETECTION_TEMPLATE 을 그대로 채용 (다중 객체 지향)
        self.QUESTION_TEMPLATE = (
            'Please find "{Question}" with bboxs and points.'
            "Compare the difference between object(s) and find the most closely matched object(s)."
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
            "Output the bbox(es) and point(s) inside the interested object(s) in JSON format."
            "i.e., <think> thinking process here </think>"
            "<answer>{Answer}</answer>"
        )
        # VR 의 few-shot 예시 (2개 객체) 그대로
        self.ANSWER_EXAMPLE = (
            '[{"bbox_2d": [10,100,200,210], "point_2d": [30,110]}, '
            '{"bbox_2d": [225,296,706,786], "point_2d": [302,410]}]'
        )

    def load_model(self, **kwargs):
        print(f"[WiseSeg] Loading Reasoning Model: {self.reasoning_model_path} (device={self.device})")
        self.reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.reasoning_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=self.device,   # 'auto' 대신 지정 device 에 고정 (멀티 GPU 분리)
        )
        self.reasoning_model.eval()

        print("[WiseSeg] Loading Processor...")
        self.processor = AutoProcessor.from_pretrained(self.reasoning_model_path, padding_side="left")

        print(f"[WiseSeg] Loading Segmentation Model: {self.segmentation_model_path} (device={self.device})")
        # SAM2 도 같은 device 로 (구버전 호환: device 인자 미지원 시 fallback).
        try:
            self.segmentation_model = SAM2ImagePredictor.from_pretrained(
                self.segmentation_model_path, device=self.device)
        except TypeError:
            self.segmentation_model = SAM2ImagePredictor.from_pretrained(self.segmentation_model_path)
        print("[WiseSeg] Models loaded successfully.")

    def predict(self, image: Image.Image, prompt: str) -> Image.Image:
        # GPU 메모리 회수 안전망: predict() 가 끝날 때 (정상/예외 무관) 캐시된 미사용
        # CUDA 메모리를 OS 로 돌려준다. 다음 단계(예: lama_inpaint 서브프로세스)가
        # 같은 GPU 에서 weights 를 새로 올려야 하므로 OOM 위험을 줄여준다.
        # empty_cache() 는 실제 사용 중인 텐서/모델은 건드리지 않으므로 안전.
        try:
            original_width, original_height = image.size
            resize_size = 840
            x_factor, y_factor = original_width/resize_size, original_height/resize_size

            print(f"[WiseSeg] User prompt: {prompt}")

            messages = [[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image.resize((resize_size, resize_size), Image.BILINEAR)
                    },
                    {
                        "type": "text",
                        "text": self.QUESTION_TEMPLATE.format(
                            Question=prompt.lower().strip(".\"?!"),
                            Answer=self.ANSWER_EXAMPLE,
                        )
                    }
                ]
            }]]

            text = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            print("[WiseSeg] Generating reasoning output...")
            generated_ids = self.reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            bboxes, points, think = parse_reasoning_output(output_text[0], x_factor, y_factor)
            print(f"[WiseSeg] Thinking process: {think}")
            print(f"[WiseSeg] Parsed objects: {len(bboxes)} bbox(es), {len(points)} point(s)")

            # 객체 0 → 빈 마스크 (하류 인페인터가 'nothing-to-do' 케이스로 처리)
            if not bboxes:
                print("[WiseSeg] Warning: no bbox parsed from reasoning output. Returning empty mask.")
                return Image.new("L", (original_width, original_height), 0)

            # points 개수가 bboxes 와 안 맞으면 box-only 모드로 안전 전환 (VR 도 동일 정책)
            use_points = bool(points) and len(points) == len(bboxes)

            print(f"[WiseSeg] Running SAM2 on {len(bboxes)} object(s) "
                  f"({'box+point' if use_points else 'box-only'})...")
            mask_union = np.zeros((original_height, original_width), dtype=bool)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.segmentation_model.set_image(image)
                for i, bbox in enumerate(bboxes):
                    if use_points:
                        masks, scores, _ = self.segmentation_model.predict(
                            point_coords=[points[i]],
                            point_labels=[1],
                            box=bbox,
                        )
                    else:
                        masks, scores, _ = self.segmentation_model.predict(box=bbox)
                    # 점수 최상위 마스크 채택 후 합집합
                    best = masks[int(np.argmax(scores))].astype(bool)
                    mask_union = np.logical_or(mask_union, best)

            # 단일 2D 합집합 마스크를 PIL "L"(흑백, 0/255) 로 반환 — 하류 contract 동일
            mask_image = Image.fromarray((mask_union * 255).astype(np.uint8), mode="L")
            return mask_image
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
