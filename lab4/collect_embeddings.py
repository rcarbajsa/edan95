import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pdb

class collect_embeddings:

    def __init__(self):

        self.embeddings_dict = {}

    def collect(self):

        path = '/home/rcarbajsa/Escritorio/edan95_datasets/glove.6B.100d.txt'  
        f = open(path)

        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            self.embeddings_dict[word] = vector
        f.close()
        return self.embeddings_dict

    def cosine_similarity(self, word):
        
        vector1 = self.embeddings_dict[word]
        vector1 = vector1.reshape(1, len(vector1))
        list_cos = []
        for key, vector in self.embeddings_dict.items():
            if key is not word:
                vector = vector.reshape(1, len(vector))
                cos = cosine_similarity(vector1,vector)
                if len(list_cos) < 5:
                    list_cos.append((key,cos[0][0]))
                elif cos[0][0] > min(list_cos,key=lambda item:item[1])[1]: 
                    list_cos.remove(min(list_cos,key=lambda item:item[1]))
                    list_cos.append((key,cos[0][0]))
        for word, cos in list_cos:
            print(word + ' ' +str(cos))
        print()

if __name__ == '__main__':
    cl = collect_embeddings()
    cl.collect()
    #pdb.set_trace()
    cl.cosine_similarity('table')
    cl.cosine_similarity('france')
    cl.cosine_similarity('sweden')
    
