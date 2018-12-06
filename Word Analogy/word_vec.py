import io, collections
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
    result = {}
    dimension = len(model["hello"])
    for i in data:
        for idx in data[i]:
            sum1 = np.zeros(dimension)
            sum2 = np.zeros(dimension)
            try:
                data1 = model[data[i][idx][0]]
            except:
               continue
            
            try:
                data2 = model[data[i][idx][2]]
            except:
                continue
            
            try:
                data3 = model[data[i][idx][1]]
            except:
                continue
            
            try:
                data4 = model[data[i][idx][3]]
            except:
                continue
            sum1 += (data1 + data2)/2
            sum2 += (data3 + data4)/2
            x.append(sum1)
            x.append(sum2)
        result[i] = np.array(x)
    od = collections.OrderedDict(sorted(result.items()))
    return od


# function to get y target
def generate_targets(model, data):
    x = []
    result = {}
    for i in data:
        for j in data[i]:
            try:
                data1 = model[data[i][j][0]]
            except:
               continue

            try:
                data2 = model[data[i][j][1]]
            except:
                continue

            try:
                data3 = model[data[i][j][2]]
            except:
                continue

            try:
                data4 = model[data[i][j][3]]
            except:
                continue
            x.append(0)
            x.append(1)
        result[i] = np.array(x)
    od = collections.OrderedDict(sorted(result.items()))
    return od
