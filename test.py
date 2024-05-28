import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Caminho para o modelo treinado
model_path = '/home/naruto/Documents/python/segment-v1/200_epochs-/weights/last.pt'

# Load YOLO model with trained weights
m = YOLO(model_path)

# Perform prediction
new_image = '/home/naruto/Documents/python/segment-v1/images/img15_2023.jpg'
res = m.predict(new_image)

# Load the original image
img = cv2.imread(new_image)

# Iterate detection results
for r in res:
    img_name = Path(new_image).stem

    # Create a copy of the original image
    output_img = img.copy()

    # Iterate each object contour
    for ci, c in enumerate(r):
        label = c.names[c.boxes.cls.tolist().pop()]

        # Create binary mask
        b_mask = np.zeros(img.shape[:2], np.uint8)

        # Create contour mask
        contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
        _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

        # Add the binary mask to the output image
        output_img[b_mask == 255] = [0, 0, 255]  # Red color for segmented pixels

        # Calculate area of the mask
        area = np.sum(b_mask == 255)  # Sum of all pixels with value 255 (white)

        print(f"Area of {label}: {area} pixels")

        # Isolate object with transparent background
        isolated = np.dstack([img, b_mask])

        # Define output path for the isolated object
        output_path = f'/home/naruto/Documents/python/segment-v1/output_images/{img_name}_{label}_isolated.png'

        # Save the isolated object with a transparent background
        cv2.imwrite(output_path, isolated)

        print(f"Isolated object saved as {output_path}")

    # Display the output image
    cv2.imshow('Segmented Image', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
