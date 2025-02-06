import sys
import cv2
import time
import psutil
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, 
                            QPushButton, QFileDialog, QHBoxLayout, QInputDialog, QGroupBox,
                            QDialog, QFormLayout, QDoubleSpinBox, QSpinBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
import supervision as sv
import torch
import GPUtil
from collections import deque

class PerformanceMonitor:
    def __init__(self, fps_buffer_size=30):
        self.fps_buffer = deque(maxlen=fps_buffer_size)
        self.frame_times = deque(maxlen=fps_buffer_size)
        self.last_time = time.time()
    
        
    def update_fps(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        
        if len(self.frame_times) > 1:
            self.fps_buffer.append(1.0 / np.mean(self.frame_times))
    
    def get_fps(self):
        if len(self.fps_buffer) > 0:
            return np.mean(self.fps_buffer)
        return 0
    
    def get_resource_usage(self):
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        gpu_info = {}
        if torch.cuda.is_available():
            try:
                gpu = GPUtil.getGPUs()[0]
                gpu_info = {
                    'gpu_load': gpu.load * 100,
                    'gpu_memory_used': gpu.memoryUsed,
                    'gpu_memory_total': gpu.memoryTotal
                }
            except Exception as e:
                print(f"Error getting GPU info: {e}")
                gpu_info = {'gpu_load': 0, 'gpu_memory_used': 0, 'gpu_memory_total': 0}
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used': memory.used / (1024 * 1024 * 1024),  
            'memory_total': memory.total / (1024 * 1024 * 1024),  
            'gpu_info': gpu_info
        }


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Parameters")
        self.setModal(True)
        
        # Get current values from parent
        self.parent = parent
        
        layout = QFormLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # Create spin boxes for each parameter
        self.track_thresh = QDoubleSpinBox()
        self.track_thresh.setRange(0.1, 1.0)
        self.track_thresh.setSingleStep(0.1)
        self.track_thresh.setValue(parent.track_thresh)
        
        self.track_buffer = QSpinBox()
        self.track_buffer.setRange(1, 300)
        self.track_buffer.setValue(parent.track_buffer)
        
        self.conf_thresh = QDoubleSpinBox()
        self.conf_thresh.setRange(0.1, 1.0)
        self.conf_thresh.setSingleStep(0.1)
        self.conf_thresh.setValue(parent.conf_thresh)
        
        self.iou_thresh = QDoubleSpinBox()
        self.iou_thresh.setRange(0.1, 1.0)
        self.iou_thresh.setSingleStep(0.1)
        self.iou_thresh.setValue(parent.iou_thresh)

        # Add parameters to layout with labels
        layout.addRow("Tracking Threshold:", self.track_thresh)
        layout.addRow("Tracking Buffer:", self.track_buffer)
        layout.addRow("Confidence Threshold:", self.conf_thresh)
        layout.addRow("IOU Threshold:", self.iou_thresh)

        # Add Apply and Cancel buttons
        button_layout = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_settings)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.close)
        
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.cancel_button)
        
        # Style the buttons
        button_style = """
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """
        self.apply_button.setStyleSheet(button_style.replace("#2196F3", "#4CAF50"))
        self.cancel_button.setStyleSheet(button_style.replace("#2196F3", "#F44336"))

        # Add button layout to main layout
        layout.addRow("", button_layout)
        self.setLayout(layout)

    def apply_settings(self):
        # Update parent's parameters
        self.parent.track_thresh = self.track_thresh.value()
        self.parent.track_buffer = self.track_buffer.value()
        self.parent.conf_thresh = self.conf_thresh.value()
        self.parent.iou_thresh = self.iou_thresh.value()
        
        # Update parent's tracker
        self.parent.tracker = sv.ByteTrack(
            track_thresh=self.parent.track_thresh,
            track_buffer=self.parent.track_buffer
        )
        
        # Update status
        self.parent.resource_label.setText(
            f"Parameters updated:\nTrack Thresh: {self.parent.track_thresh:.2f}, "
            f"Buffer: {self.parent.track_buffer}, Conf: {self.parent.conf_thresh:.2f}, "
            f"IOU: {self.parent.iou_thresh:.2f}"
        )
        
        # Close the dialog
        self.close()

class YOLOv8GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vehicle Tracking System")
        self.setGeometry(100, 100, 800, 600)
        
        # Initialize variables
        self.video_path = None
        self.cap = None
        self.source_type = None
        
        # Initialize model with TensorRT optimization
        self.setup_model()
        
        # Initialize supervision components
        self.tracker = sv.ByteTrack(track_thresh=0.5, track_buffer=30)  # Reduced buffer for memory
        self.box_annotator = sv.BoundingBoxAnnotator(
            thickness=2,
            color=sv.ColorPalette.default()
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.default(),
            text_thickness=1,  # Reduced thickness for performance
            text_scale=0.5
        )
        
        # Setup timer with reduced FPS for Jetson
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor()
        
        # Add performance display timer
        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self.update_performance_stats)
        self.perf_timer.start(1000)  # Update stats every second
        
        # Add default parameter values
        self.track_thresh = 0.5
        self.track_buffer = 30
        self.conf_thresh = 0.5
        self.iou_thresh = 0.5
        
        # Add dialog reference
        self.settings_dialog = None
        
        self.initUI()

    def setup_model(self):
        # Check for CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        try:
            # Load model with TensorRT optimization
            self.model = YOLO("final.pt")
            self.model.to(self.device)
            
            # Set inference parameters 
            self.model.conf = 0.5  # Confidence threshold
            self.model.iou = 0.5   # NMS IoU threshold
            self.model.agnostic = False  # Class-agnostic NMS
            self.model.max_det = 100  # Maximum detections per image
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)  # Add spacing between widgets
        main_layout.setContentsMargins(20, 20, 20, 20)  # Add margins around the window

        # Video Display Label with frame
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)  # Set minimum size
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #cccccc;
                background-color: #f0f0f0;
                border-radius: 5px;
            }
        """)
        main_layout.addWidget(self.video_label)

        # Source Selection Group
        source_group = QGroupBox("Source Selection")
        source_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        """)
        source_layout = QHBoxLayout()
        source_layout.setSpacing(10)

        # Style for all buttons
        button_style = """
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """

        # Video File Button
        self.video_button = QPushButton("Select Video File")
        self.video_button.setStyleSheet(button_style)
        source_layout.addWidget(self.video_button)

        # Webcam Button
        self.webcam_button = QPushButton("Use Webcam")
        self.webcam_button.setStyleSheet(button_style)
        source_layout.addWidget(self.webcam_button)

        # IP Camera Button
        self.ipcam_button = QPushButton("Use IP Camera")
        self.ipcam_button.setStyleSheet(button_style)
        source_layout.addWidget(self.ipcam_button)

        source_group.setLayout(source_layout)
        main_layout.addWidget(source_group)

        # Control Group
        control_group = QGroupBox("Detection Controls")
        control_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        """)
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)

        # Start Detection Button
        self.start_button = QPushButton("Start Detection")
        self.start_button.setStyleSheet(button_style.replace("#2196F3", "#4CAF50"))  # Green
        control_layout.addWidget(self.start_button)

        # Stop Detection Button
        self.stop_button = QPushButton("Stop Detection")
        self.stop_button.setStyleSheet(button_style.replace("#2196F3", "#F44336"))  # Red
        control_layout.addWidget(self.stop_button)

        # Add Settings Button to control group
        self.settings_button = QPushButton("Settings")
        self.settings_button.setStyleSheet(button_style.replace("#2196F3", "#FF9800"))  # Orange
        self.settings_button.clicked.connect(self.show_settings)
        control_layout.addWidget(self.settings_button)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        # Performance Metrics Group
        metrics_group = QGroupBox("Performance Metrics")
        metrics_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QLabel {
                font-size: 12px;
                padding: 5px;
            }
        """)
        metrics_layout = QVBoxLayout()

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("font-family: monospace;")
        metrics_layout.addWidget(self.fps_label)

        self.resource_label = QLabel("Resource Usage: --")
        self.resource_label.setStyleSheet("font-family: monospace;")
        metrics_layout.addWidget(self.resource_label)

        # Export Performance Report Button
        self.export_button = QPushButton("Export Performance Report")
        self.export_button.setStyleSheet(button_style.replace("#2196F3", "#9C27B0"))  # Purple
        metrics_layout.addWidget(self.export_button)

        metrics_group.setLayout(metrics_layout)
        main_layout.addWidget(metrics_group)

        self.setLayout(main_layout)
        
        # Connect button signals
        self.video_button.clicked.connect(self.select_video_file)
        self.webcam_button.clicked.connect(self.use_webcam)
        self.ipcam_button.clicked.connect(self.use_ip_camera)
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        self.export_button.clicked.connect(self.export_performance_report)
        self.settings_button.clicked.connect(self.show_settings)

        # Set window style
        self.setStyleSheet("""
            QWidget {
                font-family: Arial;
                font-size: 14px;
                background-color: white;
            }
        """)

    def select_video_file(self):
        self.stop_detection()  # Stop any running detection
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", 
                                                 "Video Files (*.mp4 *.avi *.mkv);;All Files (*)")
        if file_name:
            self.video_path = file_name
            self.source_type = "video"
            self.video_label.setText("Video file selected: " + file_name)

    def use_webcam(self):
        self.stop_detection()  # Stop any running detection
        self.video_path = 0  # Use default webcam
        self.source_type = "webcam"
        self.video_label.setText("Webcam selected")

    def use_ip_camera(self):
        self.stop_detection()  # Stop any running detection
        ip_address, ok = QInputDialog.getText(self, 'IP Camera', 
                                            'Enter IP Camera URL (e.g., rtsp://username:password@ip_address:554/):')
        if ok and ip_address:
            self.video_path = ip_address
            self.source_type = "ipcam"
            self.video_label.setText("IP Camera selected: " + ip_address)

    def start_detection(self):
        if self.video_path is None:
            self.video_label.setText("Please select a video source first!")
            return
        
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.video_label.setText("Error: Could not open video source!")
                self.cap = None
                return
        
        # Adjust FPS based on device capability (lower for Jetson)
        self.timer.start(50)  # 20 FPS (adjust as needed)

    def stop_detection(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def update_frame(self):
        if self.cap is None:
            return
        
        try:
            start_time = time.time()
            
            success, frame = self.cap.read()
            if not success:
                if self.source_type == "video":
                    self.stop_detection()
                else:
                    self.cap = cv2.VideoCapture(self.video_path)
                return

            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))
            
            # Run YOLO detection and tracking with updated parameters
            with torch.no_grad():
                results = self.model.track(
                    frame, 
                    conf=self.conf_thresh,  # Use updated confidence threshold
                    iou=self.iou_thresh,    # Use updated IOU threshold
                    persist=True,
                    verbose=False
                )[0]

            # Update performance metrics
            self.performance_monitor.update_fps()
            
            detections = sv.Detections.from_ultralytics(results)
            
            if len(detections) > 0:
                detections = self.tracker.update_with_detections(detections)
                
                # Create labels with class names, tracker IDs, and confidence scores
                labels = [
                    f"{results.names[class_id]} {tracker_id} ({conf:.2f})"  # Added confidence score
                    for class_id, tracker_id, conf in zip(
                        detections.class_id, 
                        detections.tracker_id, 
                        detections.confidence  # Include confidence in label
                    )
                ]
                
                annotated_frame = frame.copy()
                annotated_frame = self.box_annotator.annotate(
                    scene=annotated_frame, 
                    detections=detections
                )
                annotated_frame = self.label_annotator.annotate(
                    scene=annotated_frame, 
                    detections=detections,
                    labels=labels
                )
            else:
                annotated_frame = frame.copy()

            # Convert frame to QImage for PyQt5
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            height, width, channels = annotated_frame.shape
            bytes_per_line = channels * width
            q_image = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Convert QImage to QPixmap and update QLabel
            pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))

        except Exception as e:
            print(f"Error in frame processing: {e}")

    def update_performance_stats(self):
        # Get current FPS
        current_fps = self.performance_monitor.get_fps()
        
        # Get resource usage
        resources = self.performance_monitor.get_resource_usage()
        
        # Update performance display
        self.fps_label.setText(f"FPS: {current_fps:.2f}")
        
        # Format resource usage string
        resource_text = (
            f"CPU: {resources['cpu_percent']:.1f}% | "
            f"Memory: {resources['memory_used']:.1f}/{resources['memory_total']:.1f} GB "
            f"({resources['memory_percent']:.1f}%)"
        )
        
        if torch.cuda.is_available():
            gpu_info = resources['gpu_info']
            resource_text += (
                f"\nGPU Load: {gpu_info['gpu_load']:.1f}% | "
                f"GPU Memory: {gpu_info['gpu_memory_used']}/{gpu_info['gpu_memory_total']} MB"
            )
        
        self.resource_label.setText(resource_text)

    def export_performance_report(self):
        """Export performance metrics to a file"""
        report = {
            'fps': self.performance_monitor.get_fps(),
            'resources': self.performance_monitor.get_resource_usage(),
            'model_info': {
                'device': str(self.device),
                'confidence_threshold': self.model.conf,
                'iou_threshold': self.model.iou,
                'max_detections': self.model.max_det
            }
        }
        
        # Save report to file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        with open(f'performance_report_{timestamp}.txt', 'w') as f:
            f.write("YOLOv8 Performance Report\n")
            f.write("========================\n\n")
            f.write(f"Average FPS: {report['fps']:.2f}\n\n")
            f.write("Resource Usage:\n")
            f.write(f"CPU Usage: {report['resources']['cpu_percent']:.1f}%\n")
            f.write(f"Memory Usage: {report['resources']['memory_used']:.1f}/{report['resources']['memory_total']:.1f} GB\n")
            if torch.cuda.is_available():
                f.write(f"GPU Load: {report['resources']['gpu_info']['gpu_load']:.1f}%\n")
                f.write(f"GPU Memory: {report['resources']['gpu_info']['gpu_memory_used']}/{report['resources']['gpu_info']['gpu_memory_total']} MB\n")
            
            f.write("\nModel Configuration:\n")
            f.write(f"Device: {report['model_info']['device']}\n")
            f.write(f"Confidence Threshold: {report['model_info']['confidence_threshold']}\n")
            f.write(f"IoU Threshold: {report['model_info']['iou_threshold']}\n")
            f.write(f"Max Detections: {report['model_info']['max_detections']}\n")
            

    def show_settings(self):
        if self.settings_dialog is None:
            self.settings_dialog = SettingsDialog(self)
            self.settings_dialog.finished.connect(self.on_settings_closed)
            self.settings_dialog.show()

    def on_settings_closed(self):
        self.settings_dialog = None

    def closeEvent(self, event):
        self.stop_detection()
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = YOLOv8GUI()
    window.show()
    sys.exit(app.exec_())
