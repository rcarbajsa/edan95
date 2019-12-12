from conll_dictorizer import CoNLLDictorizer, Token
from collect_embeddings import collect_embeddings 
import numpy as np 
import pdb
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, SimpleRNN, Dropout, Bidirectional, LSTM
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

def load_conll2003_en():

    train_file = '/home/rcarbajsa/Escritorio/edan95_datasets/NER-data/eng.train'
    dev_file = '/home/rcarbajsa/Escritorio/edan95_datasets/NER-data/eng.valid'
    test_file = '/home/rcarbajsa/Escritorio/edan95_datasets/NER-data/eng.test'
    column_names = ['form', 'ppos', 'pchunk', 'ner']
    train_sentences = open(train_file).read().strip()
    dev_sentences = open(dev_file).read().strip()
    test_sentences = open(test_file).read().strip()
    return train_sentences, dev_sentences, test_sentences, column_names

def build_dict(data):

    x = []
    y = []
    for sentence in data:
        xx = []
        yy = []
        for word in sentence:
            xx.append(word['form'])
            yy.append(word['ner'])
        x.append(xx)
        y.append(yy)
    return x,y

def create_indices(x, embedding_dict):

    xs, ys = [], []
    for sentence in x:
        for word in set(sentence):
            xs.append(word)
    xs = list(set(xs))
    vocab = [' ', '?'] + list(set(xs + list(embedding_dict.keys())))
    print(len(vocab) - 2)
    vocab = {k:v for v, k in enumerate(vocab)}
    return vocab

def create_ner_dict(ner_tags):

    ys = []
    for sentence in ner_tags:
        for word in set(sentence):
            ys.append(word)
    ner_list = [' ', '?'] + list(set(ys))
    ner_dict = {k:v for v, k in enumerate(ner_list)}
    return ner_dict


def building_embedding_matrix(vocab, embedding_dict):

    matrix = np.zeros((len(vocab), len(embedding_dict['.'])))
    i = 2
    for word in vocab.keys()[2:]:
        if word in embedding_dict:
            matrix[i] = embedding_dict[word]
        i+=1
    return matrix

def convert_xy(x, vocab):
    xx =[]
    for sentence in x:
        aux = []
        for word in sentence:
            aux.append(vocab.get(word, 1))
        xx.append(aux)
    return pad_sequences(xx, maxlen=150)

def build_rnn(max_words, embedding_dim, matrix, x_train_pad, y_train_pad, x_dev_pad, y_dev_pad, x_test_pad, y_test_pad, ner_len):

    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, mask_zero=True, input_length=150))
    model.layers[0].set_weights([matrix])
    model.layers[0].trainable = False
    model.add(Bidirectional(SimpleRNN(100, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Dense(ner_len,activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    categ_y_train = to_categorical(y_train_pad, num_classes=ner_len)
    categ_y_dev = to_categorical(y_dev_pad, num_classes=ner_len)
    categ_y_test = to_categorical(y_test_pad, num_classes=ner_len)
    
    history = model.fit(x_train_pad, categ_y_train,
                    epochs=10,
                    batch_size=128,
                    validation_data=(x_dev_pad,categ_y_dev))

    """acc = history.history['acc']
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

    plt.show()"""

    print('-----------TEST-----------')
    test_loss, test_accuracy = model.evaluate(x_test_pad, categ_y_test)
    print('Loss: ' + str(test_loss) + ' Accuracy: ' + str(test_accuracy))

    print('-----------EVALUATE-----------')
    prediction = model.predict(x_test_pad)
    prediction = prediction.argmax(axis=-1)
    print(prediction[1])
    inverted_ner_list = {v:k for k, v in ner_list.items()}
    print(inverted_ner_list.keys()[:3])
    predicted_tag = [[inverted_ner_list[i] for i in pred] for pred in prediction]
    predicted_tag_unpadded = []
    for i in range(len(predicted_tag)):
        row = []
        for j in range(len(predicted_tag[i])):
            if categ_y_test[i][j][0] == 0:
                row.append(predicted_tag[i][j])
        predicted_tag_unpadded.append(row)

    output = open('conlleval.txt', 'w')
    #pdb.set_trace()
    for i in range(len(predicted_tag_unpadded)):
        for j in range(len(predicted_tag_unpadded[i])):
            item = str(x_test[i][j]) + ' ' + str(y_test[i][j]) + ' ' + str(predicted_tag_unpadded[i][j]) + '\n'
            output.write(item)
        output.write('\n')
    output.close()    

def build_lstm(max_words, embedding_dim, matrix, x_train_pad, y_train_pad, x_dev_pad, y_dev_pad, x_test_pad, y_test_pad, ner_len):

    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, mask_zero=True, input_length=150))
    model.layers[0].set_weights([matrix])
    model.layers[0].trainable = True
    model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=0.5)))
    model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=0.5)))
    model.add(Dense(ner_len,activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    categ_y_train = to_categorical(y_train_pad, num_classes=ner_len)
    categ_y_dev = to_categorical(y_dev_pad, num_classes=ner_len)
    categ_y_test = to_categorical(y_test_pad, num_classes=ner_len)
    history = model.fit(x_train_pad, categ_y_train,
                    epochs=12,
                    batch_size=128,
                    validation_data=(x_dev_pad,categ_y_dev))
    """acc = history.history['acc']
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
    plt.show()"""
    print('-----------TEST-----------')
    test_loss, test_accuracy = model.evaluate(x_test_pad, categ_y_test)
    print('Loss: ' + str(test_loss) + ' Accuracy: ' + str(test_accuracy))
    print('-----------EVALUATE-----------')
    prediction = model.predict(x_test_pad)
    prediction = prediction.argmax(axis=-1)
    print(prediction[1])
    inverted_ner_list = {v:k for k, v in ner_list.items()}
    print(inverted_ner_list.keys()[:3])
    predicted_tag = [[inverted_ner_list[i] for i in pred] for pred in prediction]
    predicted_tag_unpadded = []
    for i in range(len(predicted_tag)):
        row = []
        for j in range(len(predicted_tag[i])):
            if categ_y_test[i][j][0] == 0:
                row.append(predicted_tag[i][j])
        predicted_tag_unpadded.append(row)
    output = open('conlleval.txt', 'w')
    #pdb.set_trace()
    for i in range(len(predicted_tag_unpadded)):
        for j in range(len(predicted_tag_unpadded[i])):
            item = str(x_test[i][j]) + ' ' + str(y_test[i][j]) + ' ' + str(predicted_tag_unpadded[i][j]) + '\n'
            output.write(item)
        output.write('\n')
    output.close()    



if __name__ == '__main__':

    train_sentences, dev_sentences, test_sentences, column_names = load_conll2003_en()
    conll_dict = CoNLLDictorizer(column_names, col_sep=' +')
    train_dict = conll_dict.transform(train_sentences)
    dev_dict = conll_dict.transform(dev_sentences)
    test_dict = conll_dict.transform(test_sentences)
    #print(train_dict[0])
    #print(train_dict[1])
    cl = collect_embeddings()
    embedding_dict = cl.collect()
    x_train, y_train = build_dict(train_dict)
    print(x_train[1])
    
    x_dev, y_dev = build_dict(dev_dict)
    x_test, y_test = build_dict(test_dict)
    
    vocab = create_indices(x_train, embedding_dict)
    ner_list = create_ner_dict(y_train)
    #pdb.set_trace()
    matrix = building_embedding_matrix(vocab, embedding_dict)
    max_words = len(vocab)
    print('Max_words: ' +str(max_words))
    print()
    embedding_dim = len(embedding_dict['.'])
    
    x_train_pad = convert_xy(x_train, vocab)
    print(x_train[1])
    y_train_pad = convert_xy(y_train, ner_list)

    x_dev_pad = convert_xy(x_dev, vocab)
    y_dev_pad = convert_xy(y_dev, ner_list)
    
    x_test_pad = convert_xy(x_test, vocab)
    y_test_pad = convert_xy(y_test, ner_list)

    #build_rnn(max_words, embedding_dim, matrix, x_train_pad, y_train_pad, x_dev_pad, y_dev_pad, x_test_pad, y_test_pad, ner_len=len(ner_list))
    build_lstm(max_words, embedding_dim, matrix, x_train_pad, y_train_pad, x_dev_pad, y_dev_pad, x_test_pad, y_test_pad, ner_len=len(ner_list))

    
