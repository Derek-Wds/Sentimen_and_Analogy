# file to train all the data and out put the result

import json
import numpy as np
from file_parser import read_dataset1_files, read_dataset2_files
from util import *
from word_vec import *
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# read all the files and polish those files' contents

def main():
    # read_dataset1_files()
    # neg_words = json.load(open("temp_data/dataset1_neg_words.json"))
    # pos_words = json.load(open("temp_data/dataset1_pos_words.json"))

    read_dataset2_files()
    neg_words = json.load(open("temp_data/dataset2_neg_words.json"))
    pos_words = json.load(open("temp_data/dataset2_pos_words.json"))

    # NOTE: this three models all have word vector with dimension 300
    word2vec = Word2vec()
    # fasttext = fastText()
    # glove = Glove()


    X = generate_vecs(word2vec, neg_words, pos_words)
    # X = generate_vecs(fasttext, neg_words, pos_words)
    # X = generate_vecs(glove, neg_words, pos_words)
    y = generate_targets(neg_words, pos_words)
    stratified_split = StratifiedShuffleSplit(n_splits=2, test_size=0.33)
    for train_index, test_index in stratified_split.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [
        1, 0.1, 0.01, 0.001, 0.00001, 10]}
    clf_grid = GridSearchCV(SVC(kernel="linear"), param_grid, verbose=1)
    clf_grid.fit(X_train, y_train)
    best_clf = clf_grid.best_estimator_
    y_pred = best_clf.predict(X_test)
    print(f1_score(y_test, y_pred))
    print(precision_score(y_test, y_pred))
    print(recall_score(y_test, y_pred))



if __name__ == "__main__":
    main()
