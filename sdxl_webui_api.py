import os
import cv2
import base64
import requests
import json
from glob import glob
from tqdm import tqdm

# Web UI API 주소
URL = "http://127.0.0.1:7860/sdapi/v1/img2img"

def encode_base64(img_path):
    """이미지를 읽어서 base64 문자열로 변환합니다."""
    with open(img_path, "rb") as b_file:
        return base64.b64encode(b_file.read()).decode('utf-8')

def decode_base64(base64_str, output_path):
    """base64 문자열을 이미지로 디코딩하여 저장합니다."""
    img_data = base64.b64decode(base64_str)
    with open(output_path, "wb") as f:
        f.write(img_data)

def process_batch(image_dir, mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 목록 가져오기
    img_paths = glob(os.path.join(image_dir, "*.png")) + glob(os.path.join(image_dir, "*.jpg"))
    
    for img_path in tqdm(img_paths, desc="Processing Images"):
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        
        mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")
        out_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(mask_path):
            print(f"Skipping {filename} - 마스크 파일이 없습니다: {mask_path}")
            continue

        # 이미지 및 마스크를 base64로 변환
        encoded_image = encode_base64(img_path)
        encoded_mask = encode_base64(mask_path)

        # 파라미터 적용 (API 페이로드)
        payload = {
            "prompt": "background",
            "negative_prompt": "person, figure, woman, man, character",
            "init_images": [encoded_image],              # 원본 이미지
            "mask": encoded_mask,                        # 흑백 마스크
            
            "width": 1080,                               # 가로
            "height": 1920,                              # 세로
            "inpaint_full_res": True,                    # Inpaint area: Only masked
            "inpaint_full_res_padding": 208,             # Only masked padding, pixels
            "mask_blur": 9,                              # Mask blur
            "inpainting_fill": 0,                        # Masked content: 0(fill), 1(original), 2(latent noise)
            "denoising_strength": 0.97,                  # Denoising strength
            "sampler_name": "DPM++ 2M",                  # Sampling method
            "steps": 20,                                 # Sampling steps
            "seed": -1,                                  # 랜덤 시드
            
            # --- Soft Inpainting 적용 ---
            "alwayson_scripts": {
                "soft inpainting": {
                    "args": [True] # 체크박스 켬
                }
            }
        }

        # Web UI로 API 요청 보내기
        response = requests.post(url=URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            # 생성된 첫 번째 이미지를 가져와서 저장
            generated_img_b64 = result['images'][0]
            decode_base64(generated_img_b64, out_path)
        else:
            print(f"Error on {filename}: {response.text}")

if __name__ == "__main__":
    print("Web UI API Batch Processing 시작...")
    
    # 사용할 폴더 경로를 지정하세요!
    IMAGE_FOLDER = "/home/cjy/workspace/segment_evaluation/dataset/images"
    MASK_FOLDER = "/home/cjy/workspace/segment_evaluation/dataset/mask_10px"
    OUTPUT_FOLDER = "/home/cjy/workspace/stable-diffusion-webui/outputs"
    
    # API를 실행하기 전에 터미널에서 ./webui.sh --api 가 실행되어 있어야 함
    process_batch(IMAGE_FOLDER, MASK_FOLDER, OUTPUT_FOLDER)

