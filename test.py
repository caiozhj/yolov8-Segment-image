from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
m = YOLO('../yolov8n-seg.pt')

# Perform prediction
new_image = '/home/caio/PycharmProjects/test_Segmentaion/images/3.jpg'
res = m.predict(new_image)

# Iterate detection results
for r in res:
    img = cv2.imread(new_image)  # Load the image data
    img_name = Path(new_image).stem

    # Iterate each object contour
    for ci, c in enumerate(r):
        label = c.names[c.boxes.cls.tolist().pop()]

        # Create binary mask
        b_mask = np.zeros(img.shape[:2], np.uint8)

        # Create contour mask
        contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
        _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

        # OPTION-2: Isolate object with transparent background (when saved as PNG)
        isolated = np.dstack([img, b_mask])

        # Define output path for the isolated object
        output_path = f'/home/caio/PycharmProjects/test_Segmentaion/output_images/{img_name}_isolated.png'

        # Save the isolated object with a transparent background
        cv2.imwrite(output_path, isolated)

        print(f"Isolated object saved as {output_path}")