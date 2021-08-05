from knn import KNNclassifier
from gmm import GMM
from bayes_knn import bayesKNN
from utils import plot_confusion_mat, plot_decision_regions
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


train = pd.read_csv("Dataset 1B/train.csv", header=None)
train_all = train.to_numpy()
val = pd.read_csv("Dataset 1B/dev.csv", header=None).to_numpy()
val_labels = val[:, -1]  # last column of all rows
train_labels = train_all[:, -1]

## Task 1
if 0:
    k_values = [1, 7, 15]
    for k in k_values:
        knn = KNNclassifier(train_all, k)

        pred_labels_train = knn.classify(train_all[:, :-1])
        pred_labels_test = knn.classify(val[:, :-1])
        
        plot_confusion_mat(pred_labels_test, val_labels, knn.classes, "Confusion Matrix (Test data) for KNN with k = {}".format(k))
        plot_confusion_mat(pred_labels_train, train_labels, knn.classes, "Confusion Matrix (Train Data) for KNN with k = {}".format(k))

        
        print(f"Accuracy for Train with k : {k} = ", sum(pred_labels_train == train_labels)/len(pred_labels_train)*100, "%")
        print(f"Accuracy for Test with k : {k} = ", sum(pred_labels_test == val_labels)/len(pred_labels_test)*100, "%")

## Task 2
if 1:
    Q = 4
    unique_classes = list(set(train_all[:, -1]))
    classifier_dict = dict()
    for label in unique_classes:
        gmm = GMM(Q)
        label_train = train[train[train.columns[-1]] == label].to_numpy()
        gmm.fit(label_train[:, :-1])
        classifier_dict[label] = gmm

    pred_labels_test = []
    for point in val[:, :-1]:
        pred_labels_test.append(max(classifier_dict, key=lambda x: classifier_dict[x].class_prob(point)))

    pred_labels_train = []
    for point in train_all[:, :-1]:
        pred_labels_train.append(max(classifier_dict, key=lambda x: classifier_dict[x].class_prob(point)))

    # plot_confusion_mat(pred_labels_test, val_labels, unique_classes, "Confusion Matrix (Test data) with full covariance matrices")
    # plot_confusion_mat(pred_labels_train, train_labels, unique_classes, "Confusion Matrix (Train data) with full covariance matrices")

    if True:#PlotAll
        LegArr2d = []
        LegArr3d = []
        #PlotData points
        plt.figure(1)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Contour Plots along with decision region of GMM")
        plt.scatter(train_all[:,0],train_all[:,1])
        #Plot decision boundaries 1st
        plot_decision_regions(train_all[:, :-1], unique_classes, classifier_dict)
        #Plot 2d & 3d
        plt.figure(2)

        ax = plt.axes(projection = '3d')
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("Confidance")
        ax.set_title("3D plot of GMMs along with contour plot as shadow")

        for label in unique_classes:
            label_train = train[train[train.columns[-1]] == label].to_numpy()
            ax.scatter(label_train[:,0],label_train[:,1], -2*np.ones(label_train.shape[0]), s=10)
        
        for ThreeDCnt,label in enumerate(unique_classes):
            Leg2d,Leg3d = classifier_dict[label].visualise(label_train,ax,ThreeDCnt)
            LegArr2d.append(Leg2d)
            LegArr3d.append(Leg3d)

        #plt.figure(1)
        #plt.legend(LegArr2d,map(lambda x: "Class: "+str(int(x)) ,unique_classes))
        plt.figure(2)
        ax.legend(LegArr3d,map(lambda x: "Class: "+str(int(x)) ,unique_classes))
        
        plt.show()
## Task 3
if 0:
    Q = 10
    unique_classes = list(set(train_all[:, -1]))
    classifier_dict = dict()
    for label in unique_classes:
        gmm = GMM(Q, full=False)
        label_train = train[train[train.columns[-1]] == label].to_numpy()
        gmm.fit(label_train[:, :-1])
        classifier_dict[label] = gmm
    
    pred_labels_test = []
    for point in val[:, :-1]:
        pred_labels_test.append(max(classifier_dict, key=lambda x: classifier_dict[x].class_prob(point)))

    pred_labels_train = []
    for point in train_all[:, :-1]:
        pred_labels_train.append(max(classifier_dict, key=lambda x: classifier_dict[x].class_prob(point)))

    # plot_confusion_mat(pred_labels_test, val_labels, unique_classes, "Confusion Matrix (Test data) with diagonal covariance matrices")
    # plot_confusion_mat(pred_labels_train, train_labels, unique_classes, "Confusion Matrix (Train data) with diagonal covariance matrices")

    if True:#PlotAll
        LegArr2d = []
        LegArr3d = []
        #PlotData points
        plt.figure(1)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Contour Plots along with decision region of GMM")
        plt.scatter(train_all[:,0],train_all[:,1])
        #Plot decision boundaries 1st
        plot_decision_regions(train_all[:, :-1], unique_classes, classifier_dict)
        #Plot 2d & 3d
        plt.figure(2)

        ax = plt.axes(projection = '3d')
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("Confidance")
        ax.set_title("3D plot of GMMs along with contour plot as shadow")

        for label in unique_classes:
            label_train = train[train[train.columns[-1]] == label].to_numpy()
            ax.scatter(label_train[:,0],label_train[:,1], -2*np.ones(label_train.shape[0]), s=10)
        
        for ThreeDCnt,label in enumerate(unique_classes):
            Leg2d,Leg3d = classifier_dict[label].visualise(label_train,ax,ThreeDCnt)
            LegArr2d.append(Leg2d)
            LegArr3d.append(Leg3d)

        #plt.figure(1)
        #plt.legend(LegArr2d,map(lambda x: "Class: "+str(int(x)) ,unique_classes))
        plt.figure(2)
        ax.legend(LegArr3d,map(lambda x: "Class: "+str(int(x)) ,unique_classes))
        
        plt.show()



## Task 4
if 0:
    k_values = [10, 20]

    for k in k_values:
        bayes_knn = bayesKNN(train_all, k)

        pred_labels_test = bayes_knn.classify(val[:,:-1])
        pred_labels_train = bayes_knn.classify(train_all[:, :-1])

        plot_confusion_mat(pred_labels_test, val_labels, bayes_knn.classes, "Confusion Matrix (Test data) for bayes KNN with k = {}".format(k))
        plot_confusion_mat(pred_labels_train, train_labels, bayes_knn.classes, "Confusion Matrix (Train Data) for bayes KNN with k = {}".format(k))

        print(f"Accuracy for Train with k : {k} = ", sum(pred_labels_train == train_labels)/len(pred_labels_train)*100, "%")
        print(f"Accuracy for Test with k : {k} = ", sum(pred_labels_test == val_labels)/len(pred_labels_test)*100, "%")

        