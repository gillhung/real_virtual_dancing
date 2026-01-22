import os
import json
import cv2

IMAGE_DIR = "dataset/images"
KP_DIR = "annotations/pose_clean_coco17"
OUT_JSON = "data/coco_dataset/train.json"

os.makedirs("data/coco_dataset", exist_ok=True)

images = []
annotations = []

img_id = 1
ann_id = 1

for fn in os.listdir(KP_DIR):
    if not fn.endswith(".json"):
        continue

    img_path = os.path.join(IMAGE_DIR, fn.replace(".json", ".jpg"))
    if not os.path.exists(img_path):
        continue

    # 讀 keypoints
    with open(os.path.join(KP_DIR, fn)) as f:
        kps = json.load(f)   # [[x,y,v] × 17]

    # 攤平成 COCO 格式
    coco_kps = []
    xs, ys = [], []
    visible = 0

    for x, y, v in kps:
        coco_kps.extend([x, y, v])
        if v > 0:
            xs.append(x)
            ys.append(y)
            visible += 1

    if visible < 8:
        continue  # 人太不完整，丟掉

    # 算 bbox
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bbox = [
        x_min,
        y_min,
        x_max - x_min,
        y_max - y_min
    ]

    # image info
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    images.append({
        "id": img_id,
        "file_name": fn.replace(".json", ".jpg"),
        "width": w,
        "height": h
    })

    annotations.append({
        "id": ann_id,
        "image_id": img_id,
        "category_id": 1,
        "keypoints": coco_kps,
        "num_keypoints": visible,
        "bbox": bbox,
        "iscrowd": 0
    })

    img_id += 1
    ann_id += 1

# categories（COCO 規定）
categories = [{
    "id": 1,
    "name": "person",
    "keypoints": [
        "nose",
        "left_eye", "right_eye",
        "left_ear", "right_ear",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    ],
    "skeleton": []
}]

coco = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(OUT_JSON, "w") as f:
    json.dump(coco, f)

print(f"COCO dataset built: {OUT_JSON}")
print(f"Images: {len(images)}, Annotations: {len(annotations)}")
