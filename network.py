import numpy as np
from activations import sigmoid,sigmoid_der
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
    def feedforward(self,a):
        for W,b in zip(self.weights,self.biases):
            z = W @ a + b
            a = sigmoid(z)
        return a
    def backprop(self,x,y):
        zs = []
        activations = [x]
        a=x
        for W,b in zip(self.weights,self.biases):
            z = W @ a + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        delta = (activations[-1]-y)*sigmoid_der(zs[-1])
        nabla_b = [delta]
        nabla_w = [delta @ activations[-2].T]
        for l in range(2,self.num_layers):
            delta = (self.weights[-l+1].T @ delta) * sigmoid_der(zs[-l])
            nabla_b.append(delta)
            nabla_w.append(delta @ activations[-l-1].T)
        nabla_b.reverse()
        nabla_w.reverse()
        return nabla_w,nabla_b
    def update_mini_batch(self,batch,eta):
        nabla_w=[]
        nabla_b=[]
        n_weights=[]
        n_biases=[]
        for W in self.weights:
            nabla_w.append(np.zeros_like(W))
        for b in self.biases:
            nabla_b.append(np.zeros_like(b))
        for (x, y) in batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            for i in range(len(nabla_w)):
                nabla_w[i] = nabla_w[i] + delta_nabla_w[i]
                nabla_b[i] = nabla_b[i] + delta_nabla_b[i]

        for W,b,n_W,n_b in zip(self.weights,self.biases,nabla_w,nabla_b):
            W = W - (eta/len(batch))*n_W
            b = b - (eta/len(batch))*n_b
            n_weights.append(W)
            n_biases.append(b)
        self.weights, self.biases = n_weights, n_biases
        return self.weights,self.biases
