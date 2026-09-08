import gzip
import struct
import numpy as np
import os
import urllib

def read_images(path):
    with gzip.open(path, "rb") as f:
        # L'en-tete d'un fichier IMAGES contient 4 entiers non signes en big-endian :
        # magic number, nombre d'images (n), nombre de lignes, nombre de colonnes
        # Chaque entier "I" fait 4 octets -> 4 entiers = combien d'octets a lire ?
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))

        # Il reste maintenant n*rows*cols octets a lire : un octet (0-255) par pixel,
        # toutes les images mises bout a bout
        buf = f.read(n * rows * cols)

        # On transforme ces octets bruts en tableau NumPy 1D (encore "a plat")
        # Quel dtype correspond a des valeurs 0-255 sur 1 octet ?
        pixels_plats = np.frombuffer(buf, dtype=np.uint8)

        # On redecoupe ce tableau plat pour avoir UNE LIGNE PAR IMAGE
        # (n lignes, chacune contenant rows*cols pixels)
        images = pixels_plats.reshape( n, rows * cols)

        # Normalisation : on veut des valeurs entre 0.0 et 1.0 plutot que 0-255
        images = images / 255

    return images


def read_labels(path):
    with gzip.open(path, "rb") as f:
        # L'en-tete d'un fichier LABELS ne contient que 2 entiers :
        # magic number, nombre de labels (n) -> combien d'octets a lire ?
        magic, n = struct.unpack(">II", f.read(8))

        # Il reste n octets : un label (0-9) par image, dans le meme ordre
        # que les images du fichier correspondant
        buf = f.read(n)
        labels = np.frombuffer(buf, dtype=np.uint8)

    return labels

def download(url, path):
    if os.path.exists(path) == False:
        urllib.request.urlretrieve(url,path)
    return
