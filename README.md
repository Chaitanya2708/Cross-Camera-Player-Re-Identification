# 🏃‍♂️ Cross-Camera Player Re-Identification

## 📌 Overview
This project addresses the task of identifying and matching players across multiple camera views (broadcast and tacticam) using a two-stage pipeline:
1. **Detection Stage:** A fine-tuned YOLO model detects player crops from videos.
2. **Re-Identification Stage:** A ResNet-based embedding extractor compares and matches players between the two camera views.

---

## 📁 Directory Structure

player-reid-cross-camera/
├── src/
│ ├── detect.py # Player detection script
│ ├── reid.py # Re-identification script
├── model/
│ └── yolo_player.pt # Fine-tuned YOLOv8 model
├── data/
│ ├── broadcast.mp4 # Broadcast camera video
│ └── tacticam.mp4 # Tacticam camera video
├── detections/
│ ├── broadcast/ # Cropped players from broadcast video
│ └── tacticam/ # Cropped players from tacticam video
├── reid_output/
│ └── matched_players.csv # Final CSV of matched player pairs
├── requirements.txt # Dependencies
├── README.md # This file
└── report.md or report.pdf # Project methodology + reflection

---

## 🚀 How to Run

##⚙️ 1. Set up virtual environment (Windows)
```bash
python -m venv cleanenv
.\cleanenv\Scripts\activate
##📦 2. Install Dependencies
pip install -r requirements.txt
Alternatively, manually:
pip install torch torchvision opencv-python==4.8.0.74 ultralytics==8.0.192 numpy<2
##🎯 3. Run Player Detection
python src/detect.py --video_path data/broadcast.mp4 --output_dir detections/broadcast --model_path model/yolo_player.pt

python src/detect.py --video_path data/tacticam.mp4 --output_dir detections/tacticam --model_path model/yolo_player.pt
##🔁 4. Run Re-Identification
python src/reid.py --det1 detections/broadcast --det2 detections/tacticam --output_dir reid_output
Output: reid_output/matched_players.csv containing matched player ID pairs across cameras.

📄 Output Example (matched_players.csv)
broadcast_id	tacticam_id
001.png	087.png
034.png	120.png
...	...

##🧠 Approach & Methodology
###🔍 Detection
Fine-tuned YOLOv8 model (yolo_player.pt) used to detect player bounding boxes and crop frames.

###👥 Re-Identification
Used ResNet18 from torchvision.models to extract embeddings.

Compared all-vs-all feature vectors using cosine similarity.

Selected best matches based on similarity threshold.

###🧪 Techniques Attempted
Tried YOLOv8 (ultralytics) and fallback to PyTorch torch.load.

Used pretrained ResNet18 for visual embedding extraction.

Explored both GPU and CPU inference.

###⚠️ Challenges
YOLOv8 checkpoints are often tied to Ultralytics framework—required reworking to raw PyTorch loading.

Compatibility issues with torch.load() due to pickling.

Matching accuracy sensitive to resolution, lighting, occlusion, and viewpoint variation.

###🛠️ Improvements (With More Time)
Train a dedicated ReID model with triplet loss or contrastive learning.

Use temporal consistency or video tracking.

Integrate a UI to visualize matching results interactively.

###✅ Dependencies
torch
torchvision
opencv-python==4.8.0.74
ultralytics==8.0.192
numpy<2
Install using:

pip install -r requirements.txt

###📬 Submission Checklist
✅ All source code (src/*.py)

✅ YOLO model file

✅ Input and output sample videos/images

✅ README.md (this file)

✅ requirements.txt

✅ report.md or report.pdf explaining methodology

###📌 Notes
Tested on Python 3.9 (Windows).

Compatible with CPU-only machines (slower).

Ensure the model yolo_player.pt is valid and corresponds to YOLOv8 format.

###🤝 Acknowledgments
This project was completed as part of the Liat.ai Internship Assignment Task 1 – Cross-Camera Player Mapping.
