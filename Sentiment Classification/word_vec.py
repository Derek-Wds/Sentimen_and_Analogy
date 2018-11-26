import io, math
import numpy as np
from gensim.models import KeyedVectors

# function to load Google's word2vec model
def Word2vec():
    google_vec = KeyedVectors.load_word2vec_format(
        '../word_vecs/word2vec.bin', binary=True, limit=200000)
    return google_vec


# function to load Facebook's fasttext model
def fastText():
    fin = io.open("../word_vecs/fasttext.vec", 'r',
                  encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = dict()
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data


# function to load Stanford's Glove model
def Glove():
    data = dict()
    with open("../word_vecs/glove.6B/glove.6B.300d.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split(" ")
            word = tokens[0]
            data[word] = np.array(list(map(float, tokens[1:])))
    return data

# function to transfer vanilla bag of words model to vectors
def BOW_to_vecs(neg_data, pos_data):
    neg_dict = dict()
    pos_dict = dict()
    for i in neg_data:
        for word in neg_data[i]:
            try:
                neg_dict[i][word] += 1
            except:
                neg_dict[i] = {word:1}
    for i in pos_data:
        for word in pos_data[i]:
            try:
                pos_dict[i][word] += 1
            except:
                pos_dict[i] = {word: 1}

    # now transfer from word to vectors
    x = []
    for i in neg_dict:
        count = 0
        doc_vec = np.zeros(300)
        for word in neg_dict[i]:
            idf = get_idf(neg_dict, pos_dict, word)
            doc_vec += idf * neg_dict[i][word]
            count += neg_dict[i][word]
        if(count != 0):
            doc_vec /= count
        x.append(doc_vec)
    
    for i in pos_dict:
        count = 0
        doc_vec = np.zeros(300)
        for word in pos_dict[i]:
            idf = get_idf(neg_dict, pos_dict, word)
            doc_vec += idf * pos_dict[i][word]
            count += pos_dict[i][word]
        if(count != 0):
            doc_vec /= count
        x.append(doc_vec)

    return np.array(x)

# function to calculate the idf of word
def get_idf(neg_data, pos_data, word):
    num = 0
    for i in neg_data:
        if word in neg_data[i]:
            num += 1
    for j in pos_data:
        if word in pos_data[j]:
            num += 1
    return math.log((len(neg_data) + len(pos_data))/num)


# function to apply model vectors to the dataset
def generate_vecs(model, neg_data, pos_data):
    dimension = len(model["hello"])
    x = []
    # negative
    for idx in neg_data:
        count = 0
        doc_vec = np.zeros(dimension)
        for word in neg_data[idx]:
            word = word.lower()
            try:
                doc_vec += model[word]
                count += 1
            except:
                continue
        if count != 0:
            doc_vec /= count
        x.append(doc_vec)
    
    # positive
    for idx in pos_data:
        count = 0
        doc_vec = np.zeros(dimension)
        for word in pos_data[idx]:
            word = word.lower()
            try:
                doc_vec += model[word]
                count += 1
            except:
                continue
        if count != 0:
            doc_vec /= count
        x.append(doc_vec)
    return np.array(x)


# function to get the target labels for the dataset
def generate_targets(neg_data, pos_data):
    y = []
    # negatvie
    for n in neg_data:
        y.append(0)
    # positive
    for p in pos_data:
        y.append(1)
    return np.array(y)
