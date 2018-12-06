import json
import numpy as np
from word_vec import *
from utils import *
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def main():
    # NOTE: if you have not read the file please call the funtion read_file() first!
    data = read_files()

    # with open("data.json", "r", encoding='utf-8', newline='\n', errors='ignore') as f:
    #     data = json.load(f)
    
    # NOTE: this three models all have word vector with dimension 300
    word2vec = Word2vec()
    # fasttext = fastText()
    # glove = Glove()

    X = generate_vecs(word2vec, data)
    # X = generate_vecs(fasttext, data)
    # X = generate_vecs(glove, data)

    y = generate_targets(word2vec, data)
    # y = generate_targets(fasttext, data)
    # y = generate_targets(glove, data)

    accuracy = dict()

    
    for category in X:
        X_i = X[category]
        y_i = y[category]
        skf = StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
        for train_index, test_index in skf.split(X_i, y_i):
            X_train, X_test = X_i[train_index], X_i[test_index]
            y_train, y_test = y_i[train_index], y_i[test_index]

        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [
            1, 0.1, 0.01, 0.001, 0.00001, 10]}
        clf_grid = GridSearchCV(SVC(kernel="linear"), param_grid, verbose=1)
        clf_grid.fit(X_train, y_train)
        best_clf = clf_grid.best_estimator_
        y_pred = best_clf.predict(X_test)
        print(category)
        print(f1_score(y_test, y_pred))
        print(precision_score(y_test, y_pred))
        print(recall_score(y_test, y_pred))
        accuracy[category] = precision_score(y_test, y_pred)
        with open("word2vec.txt", 'a') as out:
            out.write(str(category) + '\n')
            out.write(str(precision_score(y_test, y_pred)) + '\n')
            out.write('\n')

    # with open("word2ec_accuracy.json", "w") as f:
    #     json.dump(accuracy, f)
    
    # with open("fasttext_accuracy.json", "w") as f:
    #     json.dump(accuracy, f)

    # with open("glove_accuracy.json", "w") as f:
    #     json.dump(accuracy, f)


if __name__ == "__main__":
    main()
