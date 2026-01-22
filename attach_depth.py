import os, json, numpy as np

KP_DIR = "annotations/mediapipe_pose"   # 33 keypoints
DEPTH_DIR = "dataset/depth"
OUT_DIR = "annotations/pose_with_depth_33"
os.makedirs(OUT_DIR, exist_ok=True)

for fn in os.listdir(KP_DIR):
    if not fn.endswith(".json"):
        continue

    with open(os.path.join(KP_DIR, fn)) as f:
        kps = json.load(f)   # [[x,y,score] x 33]

    depth = np.load(
        os.path.join(DEPTH_DIR, fn.replace(".json", ".npy"))
    )
    h, w = depth.shape

    kps_d = []
    for x, y, s in kps:
        xi = int(np.clip(x, 0, w - 1))
        yi = int(np.clip(y, 0, h - 1))
        z = float(depth[yi, xi])
        kps_d.append([x, y, z, s])

    with open(os.path.join(OUT_DIR, fn), "w") as f:
        json.dump(kps_d, f)

    print(f"Attach depth (keep 33): {fn}")
