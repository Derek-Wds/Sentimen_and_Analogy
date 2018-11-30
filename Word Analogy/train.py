import json
import numpy as np
from word_vec import *
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def read_file():
    data = dict()
    with open("questions-words.txt") as f:
        lines = f.readlines()
        type = 0
        for line in lines:
            line = line.strip()
            if ":" in line:
                type += 1
                num = 1
                data[type] = dict()
                continue
            words = line.split()
            data[type][num] = words
            num += 1

    with open("data.json", "w") as f:
        json.dump(data, f)


def main():
    # NOTE: if you have not read the file please call the funtion read_file() first!
    # read_file()

    with open("data.json", "r", encoding='utf-8', newline='\n', errors='ignore') as f:
        data = json.load(f)
    
    # NOTE: this three models all have word vector with dimension 300
    # word2vec = Word2vec()
    # fasttext = fastText()
    glove = Glove()

    # X = generate_vecs(word2vec, data)
    # X = generate_vecs(fasttext, data)
    X = generate_vecs(glove, data)
    y = generate_targets(data)
    skf = StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    for train_index, test_index in skf.split(X, y):
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

# word2vec: result
# 0.39749387087986926
# 0.4250262146102761
# 0.3733115022513303

# fasttext: result
# 0.56508346581876
# 0.5491502510621862
# 0.5819688907081457

# glove: result
# 0.4758049212473907
# 0.6229915520954117
# 0.38487515349979534
