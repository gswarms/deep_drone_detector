# Aerial Target Detection and Tracking 🎯📷

This repository contains Python code for real-time **detection** and **tracking** 
of aerial targets (currently rc planes) 
using camera-based computer vision. 
The system is built using **YOLOv8** for object detection 
and **OpenCV** for tracking and visualization.

<img title="a title" alt="Alt text" src="./doc/rc_plane_detection.png">


## 🚀 Features

- ✅ Real-time object detection using YOLOv8 (Ultralytics)
- 🎯 Multi-object tracking with OpenCV or DeepSORT
- 🧠 Easy integration with custom-trained detection models
- 💾 Option to save tracking logs and video output

## 📁 Project Structure

```text
aerial-tracking/
├── detection/           # Code related to YOLOv8 detection
│   └── detector.py
├── tracking/            # Tracking algorithms (e.g., OpenCV, DeepSORT)
│   └── tracker.py
├── dataset_utils/            # Utility functions for handling dataset for training
│   └── helpers.py
├── test_utils/               # Utility functions (grabbing, drawing, logging, etc.)
│   └── helpers.py
├── detector_tracker_record_test.py    # Main entry point for detection + tracking
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 🛠️ Installation


### 📦 Dependencies

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- PyYAML for config loading

All dependencies are listed in \`requirements.txt\`.


### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/aerial-tracking.git
cd aerial-tracking
```

2. Set up a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

### 🧠 Neural Network Model Setup

By default, the system uses [YOLOv8n](https://github.com/ultralytics/ultralytics) for detection. You can also plug in your own trained model.

```python
# Inside detector.py
model = YOLO('yolov8n.pt')  # Replace with path to your custom model if needed
```


---
## usage

### ▶️ Running the System

Use \`main.py\` to start the detection and tracking pipeline:

```bash
python detector_tracker_record_test.py
```

You can configure options in \`config.yaml\`, including:

- Camera source (e.g., webcam, RTSP stream)
- Detection confidence threshold
- Output save path

### 📊 Output

- Real-time video display with bounding boxes and tracking IDs
- Optionally saves:
  - Processed video output (\`output.avi\`)
  - Tracking log (\`track_log.csv\`)


---

## 📌 Use Cases

- Drone detection in restricted airspace
- Wildlife monitoring
- Surveillance and perimeter defense
- Air traffic visualization in simulations

---

## 🧩 Future Enhancements

- Integration with radar or GPS fusion
- Non-convex object filtering (using Shapely)
- Web dashboard for remote monitoring
- Support for 3D tracking with stereo or depth cameras

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## 🤝 Contributing

Contributions, ideas, and feedback are welcome! Please open issues or pull requests if you'd like to help improve this project.

## 📬 Contact

For questions or support, feel free to reach out via [GitHub Issues](https://github.com/roee-lulav/deep_drone_detector/issues).
