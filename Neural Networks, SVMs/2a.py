import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from utils import plot_confusion_mat

def predict1vR(X, clf_dict):
    pred_labels = []
    for x in X:
        x = x.reshape(1, -1)
        probs = dict()
        for label in clf_dict:
            p = clf_dict[label].predict_proba(x)[0][1]
            probs[label] = p
        pred_labels.append(max(probs, key=probs.get))

    pred_labels = np.array(pred_labels)

    return pred_labels


classes = ["coast", "insidecity", "mountain", "opencountry", "street"]
X_train = []
y_train = []
X_test = []
y_test = []
for label in classes:
    train = pd.read_csv("Dataset 2A/" + label + "/train.csv").to_numpy()[:, 1:].astype('float')
    test = pd.read_csv("Dataset 2A/" + label + "/dev.csv").to_numpy()[:, 1:].astype('float')
    X_train.extend(train)
    y_train.extend([label]*train.shape[0])
    X_test.extend(test)
    y_test.extend([label]*test.shape[0])
    
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# MLFFNN
if 0:
    clf = MLPClassifier((225, 225), random_state=42)
    clf.fit(X_train, y_train)
    print(f"Accuracy for train data : ", clf.score(X_train, y_train)*100, '%')
    print(f"Accuracy for test data : ", clf.score(X_test, y_test)*100, '%')
    pred_labels_train = clf.predict(X_train)
    pred_labels_test = clf.predict(X_test)
    plot_confusion_mat(pred_labels_train, y_train, clf.classes_, "MLFFNN (4 layers) with (225, 225) neurons in hidden layers (train)")
    plot_confusion_mat(pred_labels_test, y_test, clf.classes_, "MLFFNN (4 layers) with (225, 225) neurons in hidden layers (test)")


#Gaussian SVM
if 1:
    clf_dict = dict()
    for label in classes:
        y_new = (y_train == label)
        clf = SVC(C=4, kernel='rbf', random_state=42, probability=True)
        clf.fit(X_train, y_new)
        clf_dict[label] = clf
        
    pred_labels_train = predict1vR(X_train, clf_dict)
    pred_labels_test = predict1vR(X_test, clf_dict)
    print(f"Accuracy for train data : ", sum(y_train == pred_labels_train)/len(y_train)*100, "%")
    print(f"Accuracy for test data : ", sum(y_test == pred_labels_test)/len(y_test)*100, "%")
    plot_confusion_mat(pred_labels_train, y_train, classes, "Gaussian SVM based classifier(train)")
    plot_confusion_mat(pred_labels_test, y_test, classes, "Gaussian SVM based classifier (test)")
    
        