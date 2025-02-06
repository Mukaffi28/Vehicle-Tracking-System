from ultralytics import YOLO

# Load YOLOv8 Model
model = YOLO("final.pt")  

# Export to ONNX format
model.export(format="onnx")

# Run this commad in Jetson Terminal
# trtexec --onnx=final.onnx --saveEngine=vehicles.trt --fp16