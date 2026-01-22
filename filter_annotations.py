import os, json

IN_DIR = "data/pose_coco17"
OUT_DIR = "data/pose_clean"
os.makedirs(OUT_DIR, exist_ok=True)

for jf in os.listdir(IN_DIR):
    with open(f"{IN_DIR}/{jf}") as f:
        kps = json.load(f)

    if sum(1 for _,_,s in kps if s > 0.3) < 8:
        continue  # 人太不完整，丟掉

    with open(f"{OUT_DIR}/{jf}", "w") as f:
        json.dump(kps, f)
