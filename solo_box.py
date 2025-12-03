import os
import cv2
import json

# 讀取 YOLO 偵測結果 JSON
yolo_json_path = "annotations/yolo_bboxes.json"
with open(yolo_json_path, 'r') as f:
    yolo_bboxes = json.load(f)

# 設定裁切後的圖片資料夾
output_dir = "dataset/single_person_images"
os.makedirs(output_dir, exist_ok=True)

image_dir = "dataset/images"

# 裁切每張圖片中的每個人物
for img_name, bboxes in yolo_bboxes.items():
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)

    # 讀取圖片並裁切人物
    for i, box in enumerate(bboxes):
        # box = [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box)
        
        # 裁切人物區域
        person_img = img[y1:y2, x1:x2]
        
        # 儲存裁切後的圖片
        person_img_name = f"{img_name.split('.')[0]}_person_{i}.jpg"
        cv2.imwrite(os.path.join(output_dir, person_img_name), person_img)

print(f"裁切完成，共 {len(yolo_bboxes)} 張圖片，已儲存為單人圖片。")