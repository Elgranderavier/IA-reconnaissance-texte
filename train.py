import mnist_loader as mload
from network import NeuralNetwork
import numpy as np
import random
path = "/home/tetreau/Documents/Projets-Data-Science/IA-reconnaissance-texte/MNSIT_DataBase"

train_images = mload.read_images(path +"/train-images-idx3-ubyte.gz")
train_labels = mload.read_labels(path+"/train-labels-idx1-ubyte.gz")
matrice_identité=np.eye(10)
train_labels_bis = []
for label in train_labels:
    train_labels_bis.append(matrice_identité[label].reshape(10,1))
train_labels = train_labels_bis
train_images_bis = []
for image in train_images:
    train_images_bis.append(image.reshape(784,1))
train_images = train_images_bis

neuralnetwork = NeuralNetwork([784,30,10])
training_data = list(zip(train_images,train_labels))
for epoch in range(5):
    taille_lot = 20
    random.shuffle(training_data)
    for k in range (0 , len(training_data),taille_lot):
        neuralnetwork.update_mini_batch(training_data[k:k+taille_lot],0.1)
    print(f"Epoque {epoch} terminée")