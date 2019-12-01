import pickle
import numpy as np
import torch

entityInput = open("data/outputData/entity2vec50.pickle", "rb")
relationInput = open("data/outputData/relation2vec.pickle", "rb")

tmpEntityEmbedding = pickle.load(entityInput)
tmpRelationEmbedding = pickle.load(relationInput)
entityInput.close()
relationInput.close()

np.savetxt('entity_emb_50.txt', tmpEntityEmbedding.cpu().numpy())
np.savetxt('relation_emb_50.txt', tmpRelationEmbedding.cpu().numpy())
