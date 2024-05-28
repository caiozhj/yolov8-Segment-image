import cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt

# Carrega o modelo YOLO com os pesos treinados
my_new_model = YOLO('/segment-v1/200_epochs-/weights/last.pt')

# caminho para a nova imagem na qual a detecção de objetos será realizada
new_image = '/home/naruto/Documents/python/segment-v1/images/img13_2023.jpg'

# previsão de objetos na nova imagem usando o modelo previamente treinado
new_results = my_new_model.predict(new_image, conf=0.3)

# representação visual das previsões feitas pelo modelo na imagem
new_result_array = new_results[0].plot()
plt.figure(figsize=(12, 12))
plt.imshow(new_result_array)

# Salvar a figura em um arquivo com as cores originais
# Converter a imagem de RGB para BGR antes de salvar
image_bgr = cv2.cvtColor(new_result_array, cv2.COLOR_RGB2BGR)
cv2.imwrite('pred_img13_2023_200ep.jpg', image_bgr)


