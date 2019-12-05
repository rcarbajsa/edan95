from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class collect_embeddings:

    def __init__(self):

        self.embeddings_dict = {}

    def collect(self):

        path = Path('/Users/rcarb/OneDrive/Escritorio/edan95/datasets')  
        f = open(path)

        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            self.embeddings_dict[word] = vector
        f.close()

    def cosine_similarity(self, word):
        
        vector1 = self.embeddings_dict[word]
        vector1 = vector1.reshape(1, len(vector1))
        cos = []
        for key, vector in self.embeddings_dict.items():
            vector = vector.reshape(1, len(vector))
            cos = cosine_similarity(vector1,vector)
            if len(cos) < 5:
                cos.append((word,cos))
            elif cos > min(cos,key=lambda item:item[1]): 
                cos.remove(min(cos,key=lambda item:item[1]))
                cos.append((word,cos))
        for word, _ in cos:
            print(word)
        
