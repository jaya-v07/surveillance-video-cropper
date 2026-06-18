# Automated Surveillance Video Cropper (YOLOv8 + OpenCV)

A lightweight Python utility that processes a video stream, automatically detects people using a pre-trained object detection model, and extracts individual cropped images of them. 

This is perfect for automating data collection for training surveillance models, auditing security footage, or parsing video feeds efficiently.

## 🚀 Features
* **Efficient Sampling:** Samples 1 frame per second (configurable) to avoid saving thousands of nearly identical images, optimizing storage and processing time.
* **Intelligent Filtering:** Specifically targets the `person` class with a customizable confidence threshold (>50%) to minimize false positives.
* **Automated Bounding-Box Cropping:** Leverages OpenCV to dynamically crop and isolate targets into separate high-quality `.jpg` files.

## 🛠️ Tech Stack
* **Python** (Core Logic)
* **Ultralytics YOLOv8** (Object Detection Model)
* **OpenCV** (Video Parsing and Image Manipulation)

## 📁 Project Structure
```text
surveillance-video-cropper/
├── src/
│   ├── models/
│   │   └── yolov8n.pt       # Downloaded YOLO weight file
│   ├── __init__.py
│   └── main.py              # Core execution script
├── venv/                    # Local virtual environment (ignored by git)
├── .gitignore               # Keeps video data and cache out of production
├── LICENSE                  # Project license
├── README.md                # Documentation
└── requirements.txt         # Project dependencies

## 📦 Setup & Installation
Clone the repository:
Bash
git clone https://github.com/jaya-v07/surveillance-video-cropper.git
cd surveillance-video-cropper
Install the required dependencies:
Bash
pip install -r requirements.txt
Prepare your data folders:
Create a data directory in the root of the project and place your input video file there:
Plaintext
data/input_vid/sample.mp4
💡 Note: The data/ folder is pre-configured in .gitignore so your raw videos and outputs won't accidentally be pushed to GitHub.
💻 How to Run
Execute the script from the root directory using the following command:
Bash
python src/main.py
To quit the execution loop early, press q while focusing on the preview window.
Extracted crops will be automatically saved to data/output/.
⚙️ Configuration
You can easily tweak the detection parameters inside src/main.py:
Frame Rate Sampling: Modify if frame_count % 30 != 0: to change how frequently frames are evaluated.
Confidence Threshold: Change confidence > 0.5 to increase or decrease detection sensitivity.
Target Classes: Change class_id == 0 to target different COCO dataset classes (e.g., 2 for cars, 16 for dogs).