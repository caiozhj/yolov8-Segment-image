#  inferência do modelo  previamente treinado em uma nova imagem e exibe visualmente as previsões feitas pelo modelo

from ultralytics import YOLO
from matplotlib import pyplot as plt

# instancia um novo objeto do yolo carregando os pesos do modelo treinado
my_new_model = YOLO('/home/caio/PycharmProjects/test_Segmentaion/results/30_epochs-/weights/last.pt')

# caminho para a nova imagem na qual a detecção de objetos será realizada
new_image = '/home/caio/PycharmProjects/test_Segmentaion/test.jpg'

#previsão de objetos na nova imagem usando o modelo previamente treinado
new_results = my_new_model.predict(new_image, conf=0.5)

# representação visual das previsões feitas pelo modelo na imagem
new_result_array = new_results[0].plot()
plt.figure(figsize=(12, 12))
plt.imshow(new_result_array)

# Salve a figura em um arquivo
plt.savefig('predicted_image4.png')

# Exiba o caminho do arquivo
print("Imagem com previsões salva em predicted_image.png")
