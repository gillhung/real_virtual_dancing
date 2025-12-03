from ultralytics import YOLO
import cv2
import json
import os

# 載入 YOLOv8-pose 預訓練模型
model = YOLO("yolov8n-pose.pt")  # 這是預訓練的 YOLOv8-Pose 模型

# 圖片資料夾
image_dir = "dataset/single_person_images"
output_json = "annotations/yolov8_keypoints.json"
os.makedirs("annotations", exist_ok=True)

results_dict = {}

# 開始偵測每張圖片
for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    # 讀取圖片
    img = cv2.imread(img_path)
    
    # 使用 YOLOv8-Pose 進行推斷
    results = model.predict(source=img_path, save=False, verbose=False)
    
    keypoints = []
    for r in results:
        # 每個偵測結果 (每個人物) 的 keypoints
        for keypoint in r.keypoints:  # 假設每個人物有 17 或 33 個關鍵點
            keypoints.append(keypoint.tolist())  # 將每個關鍵點轉為列表
    # 儲存該圖片的關鍵點
    results_dict[img_name] = keypoints

# 儲存為 JSON 格式
with open(output_json, "w") as f:
    json.dump(results_dict, f)

print(f"YOLOv8-pose 偵測完成，關鍵點儲存為 {output_json}")
