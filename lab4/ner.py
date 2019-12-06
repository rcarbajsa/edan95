from conll_dictorizer import CoNLLDictorizer, Token
from collect_embeddings import collect_embeddings 
import numpy as np 
import pdb

def load_conll2003_en():

    train_file = '/home/rcarbajsa/Escritorio/edan95_datasets/NER-data/eng.train'
    dev_file = '/home/rcarbajsa/Escritorio/edan95_datasets/NER-data/eng.valid'
    test_file = '/home/rcarbajsa/Escritorio/edan95_datasets/NER-data/eng.test'
    column_names = ['form', 'ppos', 'pchunk', 'ner']
    train_sentences = open(train_file).read().strip()
    dev_sentences = open(dev_file).read().strip()
    test_sentences = open(test_file).read().strip()
    return train_sentences, dev_sentences, test_sentences, column_names

def build_dict():

    x = []
    y = []
    for sentence in train_dict:
        xx = []
        yy = []
        for word in sentence:
            xx.append(word['form'])
            yy.append(word['ner'])
        x.append(xx)
        y.append(yy)
    return x,y

def create_indices(x, y, embedding_dict):

    xs, ys = [], []
    for sentence in x:
        for word in set(sentence):
            xs.append(word)
    for sentence in y:
        for word in set(sentence):
            ys.append(word)
    
    vocab = [' ', '?'] + list(set(xs)) + list(set(ys)) + embedding_dict.keys()
    print(len(vocab) - 2)
    return vocab

def building_embedding_matrix(vocab, embedding_dict):

    matrix = np.zeros((len(vocab), len(embedding_dict['.'])))
    i = 2
    for word in vocab[2:]:
        if word in embedding_dict:
            matrix[i] = embedding_dict[word]
        i+=1
    return matrix

if __name__ == '__main__':

    train_sentences, dev_sentences, test_sentences, column_names = load_conll2003_en()
    conll_dict = CoNLLDictorizer(column_names, col_sep=' +')
    train_dict = conll_dict.transform(train_sentences)
    #print(train_dict[0])
    #print(train_dict[1])
    x, y = build_dict()
    cl = collect_embeddings()
    embedding_dict = cl.collect()
    vocab = create_indices(x, y, embedding_dict)
    #pdb.set_trace()
    matrix = building_embedding_matrix(vocab, embedding_dict)

    
