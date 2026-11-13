# 🚦 VisionPort – Intelligent Traffic Object Detection

**GPU-Accelerated Indian Traffic Object Detection using YOLOv8**

VisionPort is a computer vision application designed to detect and identify traffic objects in Indian road scenes using YOLOv8 and Python. It provides an interactive Streamlit interface for uploading traffic images, adjusting detection parameters, and visualizing detection results.

## ✨ Features

- **YOLOv8 Object Detection:** Detect multiple objects in traffic images.
- **Image Upload:** Upload traffic images for detection.
- **Confidence Threshold:** Adjust the minimum detection confidence.
- **IoU Threshold:** Configure Intersection over Union for detection.
- **Image Size Control:** Select the inference image size.
- **Detection Visualization:** Compare original images with detected images.
- **Detection Summary:** View total detected objects, object counts, and average confidence.
- **Environment Check:** Validate installed dependencies and hardware support.
- **API Gateway:** Includes a FastAPI gateway component.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLOv8 | Object detection |
| PyTorch | Deep learning framework |
| Streamlit | Interactive web interface |
| FastAPI | API gateway |
| OpenCV | Image processing |
| Pillow | Image handling |
| NumPy | Numerical operations |

## 📁 Project Structure

```text
VisionPort/
│
├── data/
├── docs/
├── eval/
├── logs/
├── models/
│   └── yolov8n.pt
├── optimise/
├── serve/
│   ├── __init__.py
│   ├── gateway.py
│   └── gateway_trace.py
├── train/
│
├── main.py
├── env_check.py
├── env_report.md
├── requirements.txt
├── images.png
└── .gitignore
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/SanthoshYadala/VisionPort.git
```

### 2. Navigate to the project

```bash
cd VisionPort
```

### 3. Create a virtual environment

```bash
python3 -m venv .venv
```

### 4. Activate the environment

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure the YOLO model

Place your YOLOv8 model weights at:

```text
models/yolov8n.pt
```

Ensure the model file is available before running the application.

## 🚀 Run the Application

Start VisionPort using:

```bash
streamlit run main.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## 🖥️ Application Workflow

1. Launch the VisionPort application.
2. Upload an Indian traffic image.
3. Adjust confidence and IoU thresholds.
4. Select the image size.
5. Run object detection.
6. View the original image and detection results.
7. Analyze the detection summary.

## 🔍 Environment Validation

Run the environment checker:

```bash
python env_check.py
```

This checks installed dependencies, PyTorch configuration, and available hardware acceleration.

## 🔮 Future Enhancements

- Real-time traffic video detection.
- GPU acceleration optimization.
- Vehicle classification and traffic analytics.
- Traffic density estimation.
- Object tracking.
- Advanced performance evaluation.

## 👨‍💻 Author

**Santhosh Yadala**

GitHub: [SanthoshYadala](https://github.com/SanthoshYadala)

Project: [VisionPort](https://github.com/SanthoshYadala/VisionPort)

---

⭐ If you find this project interesting, consider giving the repository a star!

**Version:** v0.1
