# 🚗 Real-Time Vehicle Tracking System with YOLOv8

This project implements a **real-time Vehicle Tracking System** using **YOLOv8** for object detection and **ByteTrack** for tracking. Designed for efficiency, the system runs on **edge devices** like the **NVIDIA Jetson Nano** and features a **GUI** for easy interaction. It supports multiple video sources, including **webcams, IP cameras, and video files**.

## ✨ Key Features

- **Custom YOLOv8n Model** for real-time vehicle detection.
- **Multi-Object Tracking** with ByteTrack.
- **Optimized for Edge Devices** (NVIDIA Jetson Nano, TensorRT support).
- **Graphical User Interface** (PyQt5) for interactive operation.
- **Performance Monitoring** (FPS, CPU, Memory, GPU usage).
- **Supports Video Sources** (Webcam, IP Camera, Video Files).

---

## 🔥 Task Breakdown

### 📌 Task 1: Model Development & Optimization

- Used **YOLOv8n** for vehicle tracking.
- Created a dataset of **398 images across 11 vehicle classes**:
  - 3-Wheeler, 4-Wheeler, Bus, Heavy-Truck, Jeep, Medium-Truck, Micro-Bus, Mini-Bus, Mini-Truck, Motorcycle, Sedan-Car.
- **Annotated using Roboflow**.
- **Fine-tuning Configuration**:
  - **Epochs**: 25, **Image Size**: 800x800, **Batch Size**: 8, **Patience**: 10.
  - **Optimizer**: AdamW (lr=0.000625, momentum=0.9).
  - Enabled **plot visualization** to monitor training progress.

### 📌 Task 2: Real-Time Object Tracking

- Implemented **Supervision’s ByteTrack** for **multi-object tracking**.
- Enhanced **re-identification** for **accuracy and object continuity**.
- **Optimized Parameters**:
  - `track_thresh=0.5`: Ensures **high-confidence** detections are tracked.
  - `track_buffer=30`: Allows tracking objects even when **briefly occluded**.

### 📌 Task 3: Edge Device Deployment

- **Converted YOLOv8 model to ONNX** for efficient inference.
- **Optimized inference using NVIDIA TensorRT**:
  ```bash
  trtexec --onnx=final.onnx --saveEngine=final.trt --fp16
  ```
- **Deployed on NVIDIA Jetson Nano** for real-world applications.

### 📌 Task 4: Graphical User Interface (GUI)

- Built an **interactive GUI using PyQt5**.
- Displays **real-time vehicle tracking** with annotations.
- Features **video source selection (file, webcam, IP camera)**.
- Includes a **settings panel** for real-time parameter tuning:
  - **Tracking threshold, confidence score, IOU threshold, tracking buffer**.

### 📌 Task 5: Performance Monitoring & Reporting

- **Tracks FPS, CPU usage, Memory, and GPU utilization**.
- **Real-time monitoring** with updates every second.
- **Exportable performance reports** for further analysis.

---

## ⚙️ Setup & Installation Guide

### 🛠 Prerequisites

- **OS**: Windows / Linux / macOS
- **Python**: 3.8+
- **GPU Optimization**: CUDA, PyTorch, NVIDIA TensorRT
- **Dependencies**: Listed in `requirements.txt`

### 🚀 Installation Steps

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Mukaffi28/Vehicle-Tracking-System.git
   cd vehicle-tracking-yolov8
   ```

2. **Create a Virtual Environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/macOS
   venv\Scripts\activate     # For Windows
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Model**  
   Open `yolo_train.ipynb` in **Jupyter Notebook** and run the training script.

5. **Convert Model to ONNX Format**  
   ```bash
   python Convert_to_Onnx.py
   ```

6. **Optimize for NVIDIA TensorRT**  
   ```bash
   trtexec --onnx=final.onnx --saveEngine=vehicles.trt --fp16
   ```

7. **Run the Application**  
   ```bash
   python Gui.py
   ```

---

## 🎯 Conclusion

This **real-time vehicle tracking system** is a **highly efficient solution** for **automated traffic monitoring, fleet management, and surveillance applications**. With **YOLOv8's accuracy, ByteTrack's efficiency, and a GUI-based control system**, it delivers **high performance** on **edge devices**.
