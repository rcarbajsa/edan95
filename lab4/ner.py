from conll_dictorizer import CoNLLDictorizer, Token
from pathlib import Path
from collect_embeddings import collect_embeddings 

def load_conll2003_en():

    train_file = Path('/Users/rcarb/OneDrive/Escritorio/edan95/datasets/NER-data/eng.train')
    dev_file = Path('/Users/rcarb/OneDrive/Escritorio/edan95/datasets/NER-data/eng.valid')
    test_file = Path('/Users/rcarb/OneDrive/Escritorio/edan95/datasets/NER-data/eng.test')
    column_names = ['form', 'ppos', 'pchunk', 'ner']
    train_sentences = open(train_file, encoding='utf8').read().strip()
    dev_sentences = open(dev_file, encoding='utf8').read().strip()
    test_sentences = open(test_file, encoding='utf8').read().strip()
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

    vocab = ' ' + '?' + set(set(sentence) for sentence in x) + set(set(sentence) for sentence in y) + embedding_dict.keys()
    print(len(vocab) - 2)
    return vocab

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

    
