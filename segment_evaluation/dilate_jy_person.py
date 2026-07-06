import cv2
import numpy as np
import os
import glob

input_dir = "/home/cjy/workspace/segment_evaluation/dataset/jy/shadow"
output_dir = "/home/cjy/workspace/segment_evaluation/dataset/jy/person_dilated_10px"

os.makedirs(output_dir, exist_ok=True)

dilate_px = 10
kernel_size = 2 * dilate_px + 1
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

for img_path in glob.glob(os.path.join(input_dir, "*.png")):
    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Failed to read: {img_path}")
        continue
        
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    filename = os.path.basename(img_path)
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, dilated)
    print(f"Saved: {out_path}")

print("Done!")
