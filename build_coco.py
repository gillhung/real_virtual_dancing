import os, json

IMG_DIR = "data/images"
ANN_DIR = "data/pose_clean"
OUT = "data/dataset/train.json"
os.makedirs("data/dataset", exist_ok=True)

images = []
annotations = []
img_id = ann_id = 1

for jf in os.listdir(ANN_DIR):
    images.append({
        "id": img_id,
        "file_name": jf.replace(".json",".jpg")
    })

    with open(f"{ANN_DIR}/{jf}") as f:
        kps = json.load(f)

    keypoints = []
    for x,y,s in kps:
        v = 2 if s > 0.3 else 1
        keypoints += [x,y,v]

    annotations.append({
        "id": ann_id,
        "image_id": img_id,
        "keypoints": keypoints,
        "num_keypoints": len(keypoints)//3
    })

    img_id += 1
    ann_id += 1

with open(OUT,"w") as f:
    json.dump({
        "images": images,
        "annotations": annotations,
        "categories":[{"id":1,"name":"person"}]
    }, f)
