from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image

#Instance
model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
model = YOLO('yolov8n-seg.pt')  # Transfer the weights from a pretrained model (recommended for training)