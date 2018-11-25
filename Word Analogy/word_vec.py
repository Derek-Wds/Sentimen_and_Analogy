import io
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


# function to get X data
def generate_vecs(model, data):
    x = []
    dimension = len(model["hello"])
    length = len(data)
    for i in range(1, length + 1):
        i = str(i)
        for idx in data[i]:
            sum1 = np.zeros(dimension)
            sum2 = np.zeros(dimension)
            try:
                data1 = model[data[i][idx][0]]
            except:
                data1 = 0
            
            try:
                data2 = model[data[i][idx][2]]
            except:
                data2 = 0
            
            try:
                data3 = model[data[i][idx][1]]
            except:
                data3 = 0
            
            try:
                data4 = model[data[i][idx][3]]
            except:
                data4 = 0
            sum1 += (data1 + data2)/2
            sum2 += (data3 + data4)/2
            x.append(sum1)
            x.append(sum2)
    return np.array(x)


# function to get y target
def generate_targets(data):
    x = []
    length = len(data)
    for i in range(1, length + 1):
        i = str(i)
        for j in data[i]:
            x.append(0)
            x.append(1)
    return np.array(x)
