from ultralytics import YOLO
from matplotlib import pyplot as plt

my_new_model = YOLO('/home/caio/PycharmProjects/test_Segmentaion/results/30_epochs-/weights/last.pt')

new_image = '/home/caio/PycharmProjects/test_Segmentaion/teste2.jpg'
new_results = my_new_model.predict(new_image, conf=0.1)  # Ajuste o limite de confiança conforme necessário

new_result_array = new_results[0].plot()
plt.figure(figsize=(12, 12))
plt.imshow(new_result_array)

# Salve a figura em um arquivo
plt.savefig('predicted_image3.png')

# Exiba o caminho do arquivo
print("Imagem com previsões salva em predicted_image.png")
