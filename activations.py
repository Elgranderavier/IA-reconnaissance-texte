#z est un array
#sigmoid est la fonction d'activation du neurone, lui permettant de venir compresser chaque élément de  z entre 0 et 1
import numpy as np
def sigmoid(z):
        z=np.clip(z,-100,100)
        return 1/(1+np.exp(-z))
def sigmoid_der(z):
        return sigmoid(z)*(1-sigmoid(z))
