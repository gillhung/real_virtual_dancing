import cv2
import os
import sys

import tkinter as tk
from tkinter import filedialog 

def select_video_file():
    """打開檔案總管，讓使用者選擇影片檔案"""

    root = tk.Tk()
    root.withdraw() 
    
    print("正在開啟檔案選擇視窗：請選擇目標影片...")

    video_path = filedialog.askopenfilename(
        title="請選擇您要拆幀的影片檔案",
        filetypes=[
            ("影片檔案", "*.mp4 *.avi *.mov *.mkv"), 
            ("所有檔案", "*.*")
        ]
    )
    
    root.destroy() 
    
    if not video_path:
        print("❌ 警告：使用者取消了選擇。程式將結束。")
        sys.exit(0)
        
    return video_path

if __name__ == "__main__":
    
    video_path = select_video_file()
    
    print(f"✅ 已選擇目標影片: {video_path}")

    output_dir = "dataset/images" 
    # 每隔多少幀存一張圖片 
    save_every_n_frame = 1
    
    # 確保輸出資料夾存在
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"致命錯誤: 無法開啟影片檔案 -> {video_path}")
        sys.exit(1) # 結束程式並回傳錯誤碼

    frame_count = 0
    saved_count = 0

    print(f"\n⏳ 開始處理，將每隔 {save_every_n_frame} 幀儲存一張圖片到 {output_dir}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break # 影片結束或讀取失敗

        if frame_count % save_every_n_frame == 0:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            
            img_name = f"{base_name}_frame_{frame_count:05d}.jpg"
            
            cv2.imwrite(os.path.join(output_dir, img_name), frame)
            saved_count += 1

        frame_count += 1
    cap.release()
    print("\n--- 處理完成 ---")
    print(f"🎥 總幀數: {frame_count}")
    print(f"🖼️ 已儲存圖片數量: {saved_count}")
    print(f"📂 圖片存放路徑: {os.path.abspath(output_dir)}")