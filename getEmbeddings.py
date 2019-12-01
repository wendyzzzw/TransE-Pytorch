import pickle
import numpy as np
import torch

entityInput = open("data/outputData/entity2vec50.pickle", "rb")
relationInput = open("data/outputData/relation2vec.pickle", "rb")

tmpEntityEmbedding = pickle.load(entityInput)
tmpRelationEmbedding = pickle.load(relationInput)
entityInput.close()
relationInput.close()

entity_emb = tmpEntityEmbedding.cpu().numpy()
print(entity_emb.shape)

relation_emb = tmpRelationEmbedding.cpu().numpy()
print(relation_emb.shape)

np.savetxt('entity_emb_50.txt', entity_emb)
np.savetxt('relation_emb_50.txt', relation_emb)
