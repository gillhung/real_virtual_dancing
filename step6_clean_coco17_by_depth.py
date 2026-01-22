import os, json, numpy as np

IN_DIR = "annotations/pose_coco17_with_depth"
OUT_DIR = "annotations/pose_clean_coco17"
os.makedirs(OUT_DIR, exist_ok=True)

for fn in os.listdir(IN_DIR):
    if not fn.endswith(".json"):
        continue

    with open(os.path.join(IN_DIR, fn)) as f:
        kps = json.load(f)   # [[x,y,z,score] x 17]

    zs = [z for _, _, z, s in kps if s > 0]
    if len(zs) < 5:
        continue

    z_med = np.median(zs)

    coco_clean = []
    visible_cnt = 0

    for x, y, z, s in kps:
        if s > 0.3 and abs(z - z_med) < 0.15:
            coco_clean.append([x, y, 2])  # visible
            visible_cnt += 1
        else:
            coco_clean.append([x, y, 0])  # not visible

    if visible_cnt < 8:
        continue  # 整個人太不完整，丟掉

    with open(os.path.join(OUT_DIR, fn), "w") as f:
        json.dump(coco_clean, f)

    print(f"Clean COCO17: {fn}")
