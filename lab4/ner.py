from conll_dictorizer import CoNLLDictorizer, Token
from collect_embeddings import collect_embeddings 
import numpy as np 
import pdb
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, SimpleRNN
import matplotlib.pyplot as plt
from keras.utils import to_categorical

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

def convert_xy(x, vocab):
    xx =[]
    padding_seq = max(len(sentence) for sentence in x)
    for sentence in x:
        aux = [0] * padding_seq
        for word in sentence:
            aux.append(vocab.index(word))
        xx.append(aux)
    return xx

def build_rnn(max_words, embedding_dim, matrix, x, y):

    model = Sequential()
    model.add(Embedding(max_words, embedding_dim))
    model.add(SimpleRNN(32))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.layers[0].set_weights([matrix])
    model.layers[0].trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    #categ_y = to_categorical(y)

    history = model.fit(x, y,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

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
    x = convert_xy(x, vocab)
    y = convert_xy(y, vocab)
    print(x[14])
    print(y[56])
    max_words = len(vocab)
    embedding_dim = len(embedding_dict['.'])
    build_rnn(max_words, embedding_dim, matrix, x, y)

    
