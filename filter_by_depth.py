import os, json, numpy as np

IN_DIR = "annotations/pose_with_depth"
OUT_DIR = "annotations/pose_clean"
os.makedirs(OUT_DIR, exist_ok=True)

for fn in os.listdir(IN_DIR):
    with open(os.path.join(IN_DIR, fn)) as f:
        kps = json.load(f)

    zs = [z for _, _, z, _ in kps]
    z_med = np.median(zs)

    clean = [
        [x, y, s]
        for x, y, z, s in kps
        if abs(z - z_med) < 0.15 and s > 0.3
    ]

    if len(clean) < 8:
        continue  # 人太不完整，丟掉

    with open(os.path.join(OUT_DIR, fn), "w") as f:
        json.dump(clean, f)

    print(f"Cleaned: {fn}")
