# YOLO_Kalman

A **YOLOv11-based inference** pipeline integrated with **Kalman filtering** for object tracking, along with **yaw and roll calculations**. This implementation supports real-time processing by replacing video input with **GStreamer output from a camera**.

## Features
- **YOLOv11 Inference** – Object detection with real-time performance.
- **Kalman Tracking** – Smooth object tracking with predictive filtering.
- **Yaw & Roll Estimation** – Orientation calculations for detected objects.
- **Custom NMS Implementations** – Greedy, Soft-NMS (Linear & Gaussian).
- **Morphological Operations** – Enhancing object contours for better detection.
- **Navigation Line Generation** – Curve fitting for optimal path estimation.

## Installation
Clone the repository and install dependencies:

```sh
git clone https://github.com/Ritabrata-Chakraborty/YOLO_Kalman.git
cd YOLO_Kalman
pip install -r requirements.txt
```

## Usage
Run inference on a video or GStreamer stream:

### Run on a video file
```sh
python main.py --source video.mp4
```

### Run on a live camera feed (GStreamer)
```sh
python main.py --source gstreamer
```

## Post-Processing
This project includes various post-processing techniques to enhance object detection and tracking:

- **Non-Maximum Suppression (NMS):**
  - Greedy NMS
  - Soft-NMS (Linear & Gaussian)
- **Morphological Operations:** Used to refine object contours and improve segmentation.
- **Navigation Line Generation:** Curve fitting techniques are applied to generate optimal paths.

## To-Do
- [ ] Improve Kalman filter tuning for smoother tracking.
- [ ] Add support for multi-object tracking.
- [ ] Optimize inference speed using TensorRT.
