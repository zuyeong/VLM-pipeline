import cv2
import os
import glob
from tqdm import tqdm

def extract_frames(video_path, output_dir, frame_interval=30):
    # 저장할 폴더가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 'case1.MOV' -> 'case1'
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[{video_name}] 에러: 영상을 열 수 없습니다.")
        return

    # 영상의 전체 프레임 수 가져오기
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    count = 0
    saved_count = 0

    # tqdm 적용
    with tqdm(total=total_frames, desc=f"{video_name} 처리 중") as pbar:
        while True:
            ret, frame = cap.read()
            
            # 영상이 끝나면 루프 종료
            if not ret:
                break
                
            # 설정한 간격마다 프레임 저장
            if count % frame_interval == 0:
                # 파일명 지정: case1_0000.jpg, case1_0030.jpg 형태
                output_filename = f"{video_name}_{count:04d}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                # 저장
                cv2.imwrite(output_path, frame)
                saved_count += 1
                
            count += 1
            pbar.update(1)

    cap.release()
    print(f"[{video_name}] 추출 완료: 총 {saved_count}장의 이미지 저장\n")

# 경로
video_files = [
    "/home/cjy/workspace/segment_eval/raw_video/case1.MOV",
    "/home/cjy/workspace/segment_eval/raw_video/case2.MOV",
    "/home/cjy/workspace/segment_eval/raw_video/case3.MOV",
    "/home/cjy/workspace/segment_eval/raw_video/case4.MOV"
    ]
save_folder = "/home/cjy/workspace/segment_eval/raw_frames"

# 30프레임당 1장 추출
for video in video_files:    
    extract_frames(video, save_folder, frame_interval=30)