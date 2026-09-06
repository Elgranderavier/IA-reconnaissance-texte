import gzip
import numpy as np
import struct
def read_images(path):
    en_tete = struct.unpack(">IIII", path.read(16))
    np.frombuffer(path.read(), dtype = np.uint8)