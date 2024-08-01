import cv2
import numpy as np

from ultralytics import YOLO
from pathlib import Path

# Caminho para o modelo treinado
model_path = '/home/naruto/Documents/python/1000-epochs/yolov8-Segment-image/1000_epochs-/weights/last.pt'

# Load YOLO model with trained weights
m = YOLO(model_path)

# Perform prediction
new_image = '/home/naruto/Documents/python/1000-epochs/yolov8-Segment-image/images/img17_2018.jpg'
res = m.predict(new_image)

# Load the original image
img = cv2.imread(new_image)

# Iterate detection results
for r in res:
    img_name = Path(new_image).stem

    # Iterate each object contour
    for ci, c in enumerate(r):
        label = c.names[c.boxes.cls.tolist().pop()]

        # Create binary mask
        b_mask = np.zeros(img.shape[:2], np.uint8)

        # Create contour mask
        contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
        _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

        # Isolate object with transparent background
        isolated = np.dstack([img, b_mask])

        # Define output path for the isolated object
        output_path = f'/home/naruto/Documents/python/1000-epochs/yolov8-Segment-image/output_images/{img_name}_{label}_isolated.png'

        # Save the isolated object with a transparent background
        cv2.imwrite(output_path, isolated)

        print(f"Isolated object saved as {output_path}")
