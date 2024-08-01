import cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt

# Carrega o modelo YOLO com os pesos treinados
my_new_model = YOLO('/home/naruto/Documents/python/1000-epochs/yolov8-Segment-image/1000_epochs-/weights/last.pt')

# caminho para a nova imagem na qual a detecção de objetos será realizada
new_image = '/home/naruto/Documents/python/1000-epochs/yolov8-Segment-image/images/img17_2018.jpg'

# previsão de objetos na nova imagem usando o modelo previamente treinado
new_results = my_new_model.predict(new_image, conf=0.8)

# representação visual das previsões feitas pelo modelo na imagem
new_result_array = new_results[0].plot()
plt.figure(figsize=(12, 12))
plt.imshow(new_result_array)

# Salvar a figura em um arquivo com as cores originais
# Converter a imagem de RGB para BGR antes de salvar
image_bgr = cv2.cvtColor(new_result_array, cv2.COLOR_RGB2BGR)
cv2.imwrite('predicted_1.jpg', image_bgr)