# define number of classes based on YAML
import yaml

from index import model

with open("/home/caio/PycharmProjects/test_Segmentaion/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

#Define a project --> Destination directory for all results
project = "/home/caio/PycharmProjects/test_Segmentaion/results"
#Define subdirectory for this specific training
name = "30_epochs-"


# Train the model
results = model.train(data='/home/caio/PycharmProjects/test_Segmentaion/data.yaml',
                      project=project,
                      name=name,
                      epochs=40,
                      patience=0, #I am setting patience=0 to disable early stopping.
                      batch=4,
                      imgsz=800)