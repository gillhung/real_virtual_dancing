from ultralytics import YOLO
import os
import json

# 加載 YOLOv8 Pose 模型
model = YOLO("yolov8n-pose.pt")

image_dir = "dataset/images"
output_json = "annotations/yolo_bboxes.json"
os.makedirs("annotations", exist_ok=True)

results_dict = {}

for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    results = model.predict(source=img_path, save=False, verbose=False)

    bboxes = []
    for r in results:
        for box in r.boxes.xyxy:  # xyxy = [x1,y1,x2,y2]
            bboxes.append(box.tolist())
    results_dict[img_name] = bboxes

# 存成 JSON
with open(output_json, "w") as f:
    json.dump(results_dict, f)

print(f"YOLO 偵測完成，結果存到 {output_json}")
