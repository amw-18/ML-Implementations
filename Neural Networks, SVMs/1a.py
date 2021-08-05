import numpy as np
from numpy.lib.arraysetops import unique 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from itertools import combinations
from utils import plot_confusion_mat, plot_decision_regions

def predict1v1(X, unique_classes, clf_dict):
    pred_labels = []
    for x in X:
        x = x.reshape(1, -1)
        votes = [0]*len(unique_classes)
        for clf in clf_dict:
            label = clf_dict[clf].predict(x)
            votes[int(label)] += 1
        pred_labels.append(votes.index(max(votes)))

    pred_labels = np.array(pred_labels, dtype='float64')

    return pred_labels


train = pd.read_csv("Dataset 1A/train.csv", header=None)
val = pd.read_csv("Dataset 1A/dev.csv", header=None)

# Perceptrons
if 0:
    unique_classes = list(set(train.to_numpy()[:, -1]))
    classwise_dict = dict()
    for label in unique_classes:
        data = train[train[train.columns[-1]] == label].to_numpy()
        X = data[:, :-1]
        y = data[:, -1]
        classwise_dict[label] = (X, y)

    clf_dict = dict()
    comb = combinations(unique_classes, 2)
    for label1, label2 in comb:
        X1, y1 = classwise_dict[label1]
        X2, y2 = classwise_dict[label2]
        X = np.append(X1, X2, axis=0)
        y = np.append(y1, y2, axis=0)
        clf = Perceptron(tol=1e-03, random_state=42)
        clf.fit(X, y)
        #print("TrainScore : {} for label {}".format(clf.score(X, y)), (label1, label2))
        clf_dict[(label1, label2)] = clf

    # for data, which in zip([train, val], ['train', 'test']):
    #     X, y = data.to_numpy()[:, :-1], data.to_numpy()[:, -1]
    #     pred_labels = predict1v1(X, unique_classes, clf_dict)
    #     print(f"Accuracy for {which} data : ", sum(y == pred_labels)/len(y)*100, "%")
    #     plot_confusion_mat(pred_labels, y, unique_classes, f"Percpetron ({which} data)")
    
    X = train.to_numpy()[:, :-1]
    plot_decision_regions(X, unique_classes, model=clf_dict, modelType="perceptron")
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision regions for each class for 1v1 Perceptrons based classifier")
    plt.tight_layout()
    plt.show()
        
# MLFFNN
if 0:
    Xy = train.to_numpy()
    X = Xy[:, :-1]
    y = Xy[:, -1]
    clf = MLPClassifier((5,), learning_rate_init=0.01 ,random_state=42)  
    clf.fit(X, y)
    # for data, which in zip([train, val], ['train', 'test']):
    #     X, y = data.to_numpy()[:, :-1], data.to_numpy()[:, -1]
    #     pred_labels = clf.predict(X)
    #     print(f"Accuracy for {which} data : ", clf.score(X, y)*100, '%')
    #     plot_confusion_mat(pred_labels, y, clf.classes_, f"MLFFNN (3 layered) with 5 neurons in hidden layer ({which} data)")
    
    plot_decision_regions(X, clf.classes_, model=clf, modelType="mlffnn")
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision regions for each class for MLFFNN")
    plt.tight_layout()
    plt.show()


# SVM  
if 1:
    unique_classes = list(set(train.to_numpy()[:, -1]))
    classwise_dict = dict()
    for label in unique_classes:
        data = train[train[train.columns[-1]] == label].to_numpy()
        X = data[:, :-1]
        y = data[:, -1]
        classwise_dict[label] = (X, y)

    clf_dict = dict()
    comb = combinations(unique_classes, 2)
    for label1, label2 in comb:
        X1, y1 = classwise_dict[label1]
        X2, y2 = classwise_dict[label2]
        X = np.append(X1, X2, axis=0)
        y = np.append(y1, y2, axis=0)
        clf = SVC(random_state=42)
        clf.fit(X, y)
        clf_dict[(label1, label2)] = clf
        # print(clf.score(X, y))

    # for data, which in zip([train, val], ['train', 'test']):
    #     X, y = data.to_numpy()[:, :-1], data.to_numpy()[:, -1]
    #     pred_labels = predict1v1(X, unique_classes, clf_dict)
    #     print(f"Accuracy for {which} data : ", sum(y == pred_labels)/len(y)*100, "%")
    #     plot_confusion_mat(pred_labels, y, unique_classes, f"Linear SVM based classifier ({which} data)")
    
    X = train.to_numpy()[:, :-1]
    plot_decision_regions(X, unique_classes, model=clf_dict, modelType="svc_one")
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision regions for each class for 1v1 SVMs based classifier")
    plt.show()