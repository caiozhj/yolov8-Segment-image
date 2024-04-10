from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image

model = YOLO('yolov8n-seg.yaml')  # utilizando modelo  yolo
model = YOLO('yolov8n-seg.pt')  # pesos do modelo são transferidos de um modelo pré-treinado que está salvo em um arquivo .pt