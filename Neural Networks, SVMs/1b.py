import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from utils import plot_confusion_mat, plot_decision_regions, EpochViz

def predict1vR(X, unique_classes, clf_dict):
    pred_labels = []
    for x in X:
        x = x.reshape(1, -1)
        probs = [0]*len(unique_classes)
        for label in clf_dict:
            p = clf_dict[label].predict_proba(x)[0][1]
            probs[int(label)] = p
        pred_labels.append(probs.index(max(probs)))

    pred_labels = np.array(pred_labels, dtype='float64')

    return pred_labels

train = pd.read_csv("Dataset 1B/train.csv", header=None)
val = pd.read_csv("Dataset 1B/dev.csv", header=None)


# MLFFNN
if 0:
    Xy = train.to_numpy()
    X = Xy[:, :-1]
    y = Xy[:, -1]
    clf = MLPClassifier((5, 3), random_state=42, max_iter=1000)
    clf.fit(X, y)
    # for data, which in zip([train, val], ['train', 'test']):
    #     X, y = data.to_numpy()[:, :-1], data.to_numpy()[:, -1]
    #     pred_labels = clf.predict(X)
    #     print(f"Accuracy for {which} data : ", clf.score(X, y)*100, '%')
        # plot_confusion_mat(pred_labels, y, clf.classes_, f"MLFFNN (4 layers) with (5, 3) neurons in hidden layers ({which})")

    plot_decision_regions(Xy[:, :-1], clf.classes_, model=clf, modelType="mlffnn")
    plt.scatter(Xy[:, :-1][:, 0], Xy[:, :-1][:, 1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision region for MLFFNN classifier")
    plt.show()


# MLFFNN
if 0:
    unique_classes = list(set(train.to_numpy()[:, -1]))
    Xy = train.to_numpy()
    X = Xy[:, :-1]
    y = Xy[:, -1]

    layers = [(5, 1), (3, 2), (3, 3)]
    for layer in layers:
        clf = MLPClassifier((5, 3), random_state=42, warm_start=True, max_iter=1000)
        EpochViz([1, 5, 20, 100, 1000], X, y, clf, unique_classes, layer_data = layer)

    
# Polynomial SVM  
if 1:
    unique_classes = list(set(train.to_numpy()[:, -1]))
    Xy = train.to_numpy()
    X = Xy[:, :-1]
    y = Xy[:, -1]
    clf_dict = dict()
    for label in unique_classes:
        y_new = (y == label)
        clf = SVC(C=1000, kernel="poly", random_state=42, probability=True)
        clf.fit(X, y_new)
        clf_dict[label] = clf
        # print(clf.score(X, y_new))

    # for data, which in zip([train, val], ['train', 'test']):
    #     X, y = data.to_numpy()[:, :-1], data.to_numpy()[:, -1]
    #     pred_labels = predict1vR(X, unique_classes, clf_dict)
    #     print(f"Accuracy for {which} data : ", sum(y == pred_labels)/len(y)*100, "%")
    #     plot_confusion_mat(pred_labels, y, unique_classes, f"Poly SVM based classifier ({which})")

    plot_decision_regions(X, unique_classes, model=clf_dict, modelType="svc_rest")
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision region for 1vRest Polynomial SVMs based classifier")
    plt.legend()
    plt.show()

#Gaussian SVM
if 0:
    unique_classes = list(set(train.to_numpy()[:, -1]))
    Xy = train.to_numpy()
    X = Xy[:, :-1]
    y = Xy[:, -1]
    clf_dict = dict()
    for label in unique_classes:
        y_new = (y == label)
        clf = SVC(C=1, kernel="rbf", random_state=42, probability=True)
        clf.fit(X, y_new)
        clf_dict[label] = clf
        # print(clf.score(X, y_new))

    # for data, which in zip([train, val], ['train', 'test']):
    #     X, y = data.to_numpy()[:, :-1], data.to_numpy()[:, -1]
    #     pred_labels = predict1vR(X, unique_classes, clf_dict)
    #     print(f"Accuracy for {which} data : ", sum(y == pred_labels)/len(y)*100, "%")
    #     plot_confusion_mat(pred_labels, y, unique_classes, f"Gaussian SVM based classifier ({which})")

    plot_decision_regions(X, unique_classes, model=clf_dict, modelType="svc_rest")
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision region for 1vRest Gaussian SVMs based classifier")
    plt.legend()
    plt.show()