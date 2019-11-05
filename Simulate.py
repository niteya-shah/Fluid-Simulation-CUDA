import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

data = (np.random.rand(10000,3).astype(np.float32) - np.array([0.5,0.5,0.5]).astype(np.float32)) * 10
nn = NearestNeighbors(n_neighbors = 100)
nn.fit(data)
t = nn.kneighbors(data, return_distance=False)
t.shape
