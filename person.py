import numpy as np

def getPerson(data):
    return np.corrcoef(data)*0.5+0.5