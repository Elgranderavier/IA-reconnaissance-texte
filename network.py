import numpy as np
class NeuralNetwork:
    def __init__(self, sizes):
        self.sizes = sizes #Liste avec le nombre de paramètre sur chaque couche du réseau
        self.num_layers = len(sizes) # Nombre de couches du réseau
        self.weights = [] #Création d'une liste des matrice des poids entre chaque couche avec pour lignes le nombre de sortie et pour colonnes le nombre d'entrée
        self.biases= [] #Création d'une liste des vecteur de biais entre chaque couche
        for i in range(len(sizes)-1):
            self.weights.append(np.random.randn(sizes[i+1],sizes[i])) #donne aux poids des valeurs aléatoires suivant une loi normale de moyenne nulle et de variance 1
        for i in sizes[1:]:
            self.biases.append(np.random.randn(i,1))