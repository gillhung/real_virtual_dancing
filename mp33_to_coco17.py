import os
import json

IN_DIR = "annotations/pose_clean"      # 33 keypoints
OUT_DIR = "annotations/pose_coco17"    # 17 keypoints
os.makedirs(OUT_DIR, exist_ok=True)

# MediaPipe index -> COCO index
MP_TO_COCO = {
    0: 0,    # nose
    11: 5,   # left_shoulder
    12: 6,   # right_shoulder
    13: 7,   # left_elbow
    14: 8,   # right_elbow
    15: 9,   # left_wrist
    16: 10,  # right_wrist
    23: 11,  # left_hip
    24: 12,  # right_hip
    25: 13,  # left_knee
    26: 14,  # right_knee
    27: 15,  # left_ankle
    28: 16   # right_ankle
}

for fn in os.listdir(IN_DIR):
    if not fn.endswith(".json"):
        continue

    with open(os.path.join(IN_DIR, fn)) as f:
        mp_kps = json.load(f)   # [[x,y,score] × 33]

    # 初始化 COCO 17（全部不可見）
    coco_kps = [[0.0, 0.0, 0] for _ in range(17)]

    for mp_idx, coco_idx in MP_TO_COCO.items():
        x, y, score = mp_kps[mp_idx]

        if score > 0.3:
            coco_kps[coco_idx] = [x, y, 2]   # visible
        else:
            coco_kps[coco_idx] = [x, y, 1]   # labeled but not visible

    with open(os.path.join(OUT_DIR, fn), "w") as f:
        json.dump(coco_kps, f)

    print(f"Converted: {fn}")
