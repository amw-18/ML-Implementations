from knn import KNNclassifier
from nb_gaussian import NBGaussian
from utils import plot_confusion_mat, plot_decision_regions
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


train = pd.read_csv("Dataset 1A/train.csv", header=None).to_numpy()
val = pd.read_csv("Dataset 1A/dev.csv", header=None).to_numpy()
val_labels = val[:, -1]  # last column of all rows
train_labels = train[:, -1]

## Task 1
if 0:
    k_values = [1, 7, 15]
    for k in k_values:
        knn = KNNclassifier(train, k)
        
        pred_labels_train = knn.classify(train[:, :-1])
        pred_labels_test = knn.classify(val[:, :-1])
        
        plot_confusion_mat(pred_labels_test, val_labels, knn.classes, "Confusion Matrix (Test data) for KNN with k = {}".format(k))
        plot_confusion_mat(pred_labels_train, train_labels, knn.classes, "Confusion Matrix (Train Data) for KNN with k = {}".format(k))
        
        print(f"Accuracy for Train with k : {k} = ", sum(pred_labels_train == train_labels)/len(pred_labels_train)*100, "%")
        print(f"Accuracy for Test with k : {k} = ", sum(pred_labels_test == val_labels)/len(pred_labels_test)*100, "%")

## Task 2
if 1:
    # types = [1, 2, 3]
    types = [3]
    for type in types:
        nbg = NBGaussian(type)
        nbg.fit(train)

        pred_labels_train = nbg.classify(train[:, :-1])
        pred_labels_test = nbg.classify(val[:, :-1])

        # plot_confusion_mat(pred_labels, val_labels, list(nbg.class_dict.keys()))
        if type == 1:
            title = r"Covariance matrix is same and is $\sigma^2$I"
        elif type == 2:
            title = "Covariance matrix is same and is C"
        elif type == 3:
            title = "Covariance matrix is different"

        # plot_confusion_mat(pred_labels_test, val_labels, list(nbg.class_dict.keys()), "NBGaussian (Test data) \n " + title)
        # plot_confusion_mat(pred_labels_train, train_labels, list(nbg.class_dict.keys()), "NBGaussian (Train data) \n " + title)
        
        print(f"Accuracy for Train for {title} = ", sum(pred_labels_train == train_labels)/len(pred_labels_train)*100, "%")
        print(f"Accuracy for Test for {title}= ", sum(pred_labels_test == val_labels)/len(pred_labels_test)*100, "%")

        # nbg.visualise(train)

        if True:
            LegArr2d = []
            LegArr3d = []
            #PlotData points
            plt.figure(1)
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.title("Contour Plots along with decision region of NBG with \n" + title)
            plt.scatter(train[:,0],train[:,1])
            #Compute scores
            nbg.compute_ClassesScore(train)
            #Plot decision boundaries 1st
            plot_decision_regions(train, sorted(nbg.class_dict.keys()),nbg.computedClassesScore,"nbg")
            
            #Plot 2d & 3d
            plt.figure(2)
            ax = plt.axes(projection = '3d')
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("Confidence")
            ax.set_title("3D plot of NBGs along with contour plot as shadow with \n" + title)
            
            for ThreeDCnt,label in enumerate(sorted(nbg.class_dict.keys())):
                LegArr2d,LegArr3d = nbg.visualise(train, ax)

            #plt.figure(1)
            #plt.legend(LegArr2d,list(map(lambda x: "Class: "+str(int(x)) ,sorted(nbg.class_dict.keys()))))
            #plt.figure(2)
            #ax.legend(LegArr3d,list(map(lambda x: "Class: "+str(int(x)) ,sorted(nbg.class_dict.keys()))))
            
            plt.show()
