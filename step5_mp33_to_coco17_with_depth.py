import os, json

IN_DIR = "annotations/pose_with_depth_33"
OUT_DIR = "annotations/pose_coco17_with_depth"
os.makedirs(OUT_DIR, exist_ok=True)

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
        mp_kps = json.load(f)   # [[x,y,z,score] x 33]

    # 初始化 COCO 17（含 depth）
    coco17 = [[0.0, 0.0, 0.0, 0.0] for _ in range(17)]

    for mp_idx, coco_idx in MP_TO_COCO.items():
        x, y, z, s = mp_kps[mp_idx]
        coco17[coco_idx] = [x, y, z, s]

    with open(os.path.join(OUT_DIR, fn), "w") as f:
        json.dump(coco17, f)

    print(f"33 -> COCO17: {fn}")
