# define number of classes based on YAML
import yaml

# Importação  modelo yolo previamente definido no arquivo index.py
from index import model

# data.yaml contém informações sobre o conjunto de dados  e nele estão as classes presentes nas imagens
#  aberto em modo de leitura ('r').
# Carrega o conteúdo do arquivo YAML e extrai o número de classes ('nc')
# O número de classes é convertido em uma string e armazenado na variável num_classes.
with open("data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

# diretório de destino para resultados do treinamento
project = "/home/caio/PycharmProjects/test_Segmentaion/results"

# Define um subdiretório específico para este treinamento
name = "200_epochs-"


# treinamento do modelo

# método train do objeto model ( modelo DOCS YOLO).
results = model.train(data='/home/caio/PycharmProjects/test_Segmentaion/data.yaml',
                      project=project,
                      name=name, # subdiretório
                      epochs=200,
                      patience=0, # Define a paciência para o algoritmo de parada antecipada. Neste caso, está definido como 0 para desabilitar a parada antecipada.
                      batch=4, # O tamanho do lote (batch size) usado durante o treinamento, definido como 4.
                      imgsz=800) # tamanho das imagens de entrada

