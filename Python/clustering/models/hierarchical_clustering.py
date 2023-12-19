import numpy as np
from .model import Model




class HierarchicalClustering(Model):
    def __init__(self, data):
        self.data = data
    def fit(self):
        raise NotImplementedError

    def predict(self, linkage="ward", metric="euclidean", n_clusters=2):
        
    
