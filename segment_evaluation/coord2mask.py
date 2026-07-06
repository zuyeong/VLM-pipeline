import json
import numpy as np
import cv2
import os
import glob

def generate_masks_from_json(json_path, output_base_dir):
    # 1. 파일 이름 추출
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    
    # 2. JSON 파일 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 이미지 크기 추출
    img_h = data['imageHeight']
    img_w = data['imageWidth']

    # 3. 라벨과 그룹별로 폴리곤 좌표 모으기
    shape_dict = {}
    for shape in data['shapes']:
        label = shape['label']
        group_id = shape.get('group_id', 0) # 그룹 ID가 없으면 기본값 0
        points = np.array(shape['points'], dtype=np.int32)

        key = (label, group_id)
        if key not in shape_dict:
            shape_dict[key] = []
        shape_dict[key].append(points)

    # 4. 저장할 하위 폴더 경로 설정
    person_dir = os.path.join(output_base_dir, "person")
    shadow_dir = os.path.join(output_base_dir, "shadow")
    
    # 5. 마스크 그리고 저장하기
    mask_count = 0
    for (label, group_id), polygons in shape_dict.items():
        # 빈 캔버스 생성
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        # 다각형 채우기
        cv2.fillPoly(mask, polygons, 255)

        # 파일명 생성 및 저장
        filename = f"{base_name}_{label}_{group_id}.png"
        save_path = os.path.join(person_dir if label == "person" else shadow_dir, filename)

        cv2.imwrite(save_path, mask)
        mask_count += 1
        
    print(f"처리 완료: {base_name} (총 {mask_count}개의 개별 마스크 생성)")

def process_all_jsons(input_dir, output_dir):
    # 1. 마스크를 저장할 최종 폴더 미리 생성
    os.makedirs(os.path.join(output_dir, "person"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "shadow"), exist_ok=True)

    # 2. 지정된 폴더 안의 모든 .json 파일 목록 가져오기
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if len(json_files) == 0:
        print(f"에러: '{input_dir}' 경로에 JSON 파일이 존재하지 않습니다.")
        return

    print(f"총 {len(json_files)}개의 JSON 파일 변환...\n")

    # 3. 파일 목록을 돌면서 하나씩 변환 함수 실행
    for json_path in json_files:
        generate_masks_from_json(json_path, output_dir)
        
    print("\n모든 정답지(GT) 마스크 생성 완료")


# ==========================================


# JSON 파일
INPUT_JSON_DIR = "/home/cjy/workspace/segment_evaluation/dataset/jy/json"

# 마스크 파일들이 저장될 폴더
OUTPUT_MASK_DIR = "/home/cjy/workspace/segment_evaluation/dataset/jy"

# 실행
process_all_jsons(INPUT_JSON_DIR, OUTPUT_MASK_DIR)