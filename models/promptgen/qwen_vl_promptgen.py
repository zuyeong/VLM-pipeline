# pipeline_test 에 있던 코드

"""Qwen2.5-VL 기반 프롬프트 생성기 (in-process, 4bit, 단일 GPU pin).

user 의도(한글 가능) + 원본 이미지 → 영어 VR 세그멘테이션 프롬프트(str).
검증된 패턴(메모리 wise-vr-prompt-patterns)을 시스템 프롬프트로 주입.

상주 패턴 (base_models 설계메모 기준): (a) in-process 즉시 로드.
  - 4bit(NF4) 로 device(기본 cuda:0)에 ~6GB 상주. SDXL 서브프로세스와 GPU0 동거 가능.
  - base + 옵션 LoRA 어댑터. 어댑터가 없으면 base + few-shot 으로 동작(= 하네스의 S0).
"""
from __future__ import annotations

import json
import os
import re

import torch
from PIL import Image

from core.base_models import BasePromptGenModel


# 검증된 패턴 규칙 (prompt_model/teacher/generate.py SYSTEM_PROMPT 와 동기화)
SYSTEM_PROMPT = (
    "You generate ENGLISH segmentation prompts for the VR-7B (VisionReasoner) model.\n"
    "The user wants to KEEP a main subject and REMOVE other people from a photo.\n"
    "Rules:\n"
    "- Default (removal targets at edges/background): "
    '"Background figures and people at the edges of the frame, excluding the central foreground subject."\n'
    '- If a statue/sculpture is present, append ", ignoring statues and sculptures".\n'
    "- If the removal target is close and visually distinct, name it directly "
    '("The person in brown clothing behind the woman in white").\n'
    "- If countable, use the counted form "
    '("Two people walking, excluding the person making a peace sign.").\n'
    '- NEVER use the negative form "Erase X without Y".\n'
    "- The prompt MUST be English even if the user's intent is Korean.\n"
    'Output strict JSON: {"pattern": "...", "needs_statue_clause": bool, "prompt": "..."}'
)

# 어댑터 부재 시 출력 형식/패턴을 유도하는 few-shot (텍스트 전용)
FEWSHOT = [
    {"role": "user",
     "content": "중앙 전경 인물만 남기고 가장자리·원거리 사람 제거"},
    {"role": "assistant",
     "content": '{"pattern":"g09","needs_statue_clause":false,"prompt":"Background figures and people at the edges of the frame, excluding the central foreground subject."}'},
    {"role": "user",
     "content": "가운데 여성만 남기고 테두리에 잘린 사람 제거, 동상은 보존"},
    {"role": "assistant",
     "content": '{"pattern":"g09","needs_statue_clause":true,"prompt":"Background figures and people at the edges of the frame, excluding the central foreground subject, ignoring statues and sculptures."}'},
]


class QwenVLPromptGenModel(BasePromptGenModel):
    def __init__(
        self,
        base: str,
        adapter: str | None = None,
        device: str = "cuda:0",
        quantization: str = "4bit",   # "4bit" | "none"
        max_new_tokens: int = 128,
    ):
        self.base = base
        self.adapter = adapter
        self.device = device
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens

        self.model = None
        self.processor = None
        self.use_fewshot = True   # 어댑터 로드되면 False

    def load_model(self, **kwargs):
        from transformers import (
            Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig,
        )
        quant = None
        if self.quantization == "4bit":
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        print(f"[PromptGen] base={self.base} device={self.device} quant={self.quantization}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base,
            quantization_config=quant,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        if self.adapter and os.path.exists(self.adapter):
            from peft import PeftModel
            print(f"[PromptGen] LoRA 어댑터 로드: {self.adapter}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter)
            self.use_fewshot = False
        else:
            if self.adapter:
                print(f"[PromptGen] 어댑터 경로 없음({self.adapter}) → base + few-shot(S0) 폴백")
            else:
                print("[PromptGen] 어댑터 미지정 → base + few-shot(S0) 폴백")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.base)
        print("[PromptGen] 로드 완료.")

    # 의도 이해엔 풀 해상도 불필요 — 긴 변을 캡해 vision token/VRAM 폭증 방지
    # (고해상도 입력 + 멀티샘플 시 OOM 회피. seg 단계는 별도로 자체 해상도 사용).
    MAX_SIDE = 1024

    def _build_inputs(self, user_text: str, image: Image.Image):
        from qwen_vl_utils import process_vision_info
        if max(image.size) > self.MAX_SIDE:
            s = self.MAX_SIDE / max(image.size)
            image = image.resize((int(image.width * s), int(image.height * s)),
                                  Image.LANCZOS)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self.use_fewshot:
            messages += FEWSHOT
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        })
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        return self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)

    def predict(self, user_text: str, image: Image.Image) -> str:
        inputs = self._build_inputs(user_text, image)
        with torch.inference_mode():
            gen = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        out = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        prompt = self._extract_prompt(out)
        print(f"[PromptGen] user={user_text!r} → vr_prompt={prompt!r}")
        return prompt

    def predict_n(self, user_text: str, image: Image.Image, n: int = 3,
                  temperature: float = 0.8) -> list:
        """멀티-후보: greedy 1개(가장 신뢰) + 샘플 (n-1)개를 합쳐 중복 제거.

        VR 의 표현 민감성을 역이용 — 어순/단어가 다른 후보들을 만들어
        하류에서 마스크가 갈리게 한 뒤 필터/소비자선택으로 좁힌다.
        """
        inputs = self._build_inputs(user_text, image)
        ilen = inputs.input_ids.shape[1]
        prompts, seen = [], set()
        # 순차 생성 (배치 num_return_sequences 는 KV 캐시 폭증 → OOM). 피크=1 시퀀스.
        for i in range(n):
            with torch.inference_mode():
                if i == 0:
                    g = self.model.generate(  # greedy (결정적, 베스트 추정)
                        **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
                else:
                    g = self.model.generate(  # 샘플 (다양성)
                        **inputs, max_new_tokens=self.max_new_tokens, do_sample=True,
                        temperature=temperature, top_p=0.95)
            txt = self.processor.batch_decode([g[0][ilen:]], skip_special_tokens=True)[0]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            p = self._extract_prompt(txt)
            if p and p not in seen:
                seen.add(p)
                prompts.append(p)
        print(f"[PromptGen] user={user_text!r} → {len(prompts)} 후보: {prompts}")
        return prompts

    @staticmethod
    def _extract_prompt(text: str) -> str:
        """모델 출력에서 영어 prompt 추출. JSON 우선, 실패 시 정규식, 최후엔 원문."""
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and obj.get("prompt"):
                    return str(obj["prompt"]).strip()
            except Exception:
                pass
        m2 = re.search(r'"prompt"\s*:\s*"([^"]+)"', text)
        if m2:
            return m2.group(1).strip()
        return text.strip()
