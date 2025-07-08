import os
import argparse
import cv2
from ultralytics import YOLO
import numpy as np

def detect_players(video_path, output_dir, model_path):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading model from {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame)[0]  # First result only

        # Draw boxes
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{model.names[cls_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Save frame
        out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"[INFO] Saved: {out_path}")
        frame_idx += 1

    cap.release()
    print("[INFO] Detection complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    detect_players(args.video_path, args.output_dir, args.model_path)
