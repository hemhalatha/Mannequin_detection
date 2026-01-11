# Mannequin Detection Model for Disaster-Response Drones

A deep learning model trained to detect mannequin-based victim dummies from live drone video streams. Designed for aerial disaster-response simulations, it helps identify human-like targets in varied environments and perspectives.


## Overview

This project focuses on detecting mannequins from drone footage to simulate search-and-rescue operations in disaster zones. The model processes images, recorded videos, or live drone streams and outputs bounding boxes with confidence scores.


## Model Details

- Architecture: **YOLOv8s** (small, fast, accurate)
- Dataset: Custom drone-captured mannequin dataset with YOLO-format annotations
- Training: Fine-tuned for accuracy on diverse conditions (different lighting, angles, and distances)
- Inference: Supports image, video, and live drone stream inputs


## Features

- Real-time mannequin detection from aerial footage
- High accuracy across varied lighting and angles
- Outputs bounding boxes with confidence scores
- Lightweight and optimized for fast inference


## Technologies Used

- **Python**
- **Ultralytics YOLOv8**
- **OpenCV**
- **NumPy**


## Usage

1. Load the trained YOLOv8 model.
2. Provide image, video, or live drone feed as input.
3. Run inference to detect mannequins and visualize bounding boxes.
