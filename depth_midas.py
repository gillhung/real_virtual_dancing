import os
import cv2
import torch
import numpy as np

IMAGE_DIR = "dataset/images"
DEPTH_DIR = "dataset/depth"
os.makedirs(DEPTH_DIR, exist_ok=True)

# 載入 MiDaS
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.dpt_transform

# 對每張影像跑深度
for fn in os.listdir(IMAGE_DIR):
    if not fn.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(IMAGE_DIR, fn)
    img = cv2.imread(img_path)
    if img is None:
        continue

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = transform(rgb)

    with torch.no_grad():
        depth = midas(inp).squeeze().cpu().numpy()

    # normalize to 0–1
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    np.save(
        os.path.join(DEPTH_DIR, fn.replace(".jpg", ".npy")),
        depth
    )

    print(f"Depth done: {fn}")
