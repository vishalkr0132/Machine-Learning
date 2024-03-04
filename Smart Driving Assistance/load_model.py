import torch

# Load the YOLOv5s model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Use the model for object detection
results = model('image.jpg')