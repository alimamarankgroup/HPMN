import numpy as np

temp = np.loadtxt('../data/xlong/graph_emb.txt')
np.save('graph_emb', temp)