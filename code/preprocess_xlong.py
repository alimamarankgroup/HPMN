import numpy as np

temp = np.loadtxt('../data/xlong/graph_emb.txt')
np.save('../data/xlong/graph_emb', temp)
