# file to train all the data and out put the result

import json
import numpy as np
from file_parser import read_dataset1_files, read_dataset2_files
from util import *
from word_vec import *
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit, learning_curve, ShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# read all the files and polish those files' contents

def main():
    # read_dataset1_files()
    neg_words = json.load(open("temp_data/dataset1_neg_words.json"))
    pos_words = json.load(open("temp_data/dataset1_pos_words.json"))

    # read_dataset2_files()
    # neg_words = json.load(open("temp_data/dataset2_neg_words.json"))
    # pos_words = json.load(open("temp_data/dataset2_pos_words.json"))

    # NOTE: this three models all have word vector with dimension 300
    # word2vec = Word2vec()
    # fasttext = fastText()
    glove = Glove()

    # X = BOW_to_vecs(neg_words, pos_words) # this is the baseline
    # X = generate_vecs(word2vec, neg_words, pos_words)
    # X = generate_vecs(fasttext, neg_words, pos_words)
    X = generate_vecs(glove, neg_words, pos_words)
    y = generate_targets(neg_words, pos_words)
    skf = StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    try:
        with open("parameters.json", 'r') as f:
            parameters = json.load(f)
        best_clf = SVC(kernel="linear", C=parameters["C"], gamma=parameters["gamma"])
        best_clf.fit(X_train, y_train)
    except:
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [
            1, 0.1, 0.01, 0.001, 0.00001, 10]}
        clf_grid = GridSearchCV(SVC(kernel="linear"), param_grid, verbose=1)
        clf_grid.fit(X_train, y_train)
        best_clf = clf_grid.best_estimator_
        with open("parameters.json", "w") as f:
            json.dump(clf_grid.best_params_, f)
    
    y_pred = best_clf.predict(X_test)
    print(f1_score(y_test, y_pred))
    print(precision_score(y_test, y_pred))
    print(recall_score(y_test, y_pred))


    title = "Learning Curves for {} (SVM, linear kernel, $\gamma=1$, C=1)".format("BOW")
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = SVC(kernel="linear",gamma=1, C=1)
    plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4)

    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


if __name__ == "__main__":
    main()
# total words: 46180/664680

# result: baseline of bag of words
# 0.2952243125904486
# 0.5340314136125655
# 0.204

# word2vec 20816/84751 oov
# 0.8217522658610271
# 0.8275862068965517
# 0.816

# fasttext 10167/27696 oov
# 0.821501014198783
# 0.8333333333333334
# 0.81

# glove 8779/14907 oov
# 0.8159509202453987
# 0.8347280334728033
# 0.798
