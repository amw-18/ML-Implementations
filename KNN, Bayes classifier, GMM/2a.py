from gmm import GMM
from utils import plot_confusion_mat, PCA
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


classes = ["coast", "insidecity", "mountain", "opencountry", "street"]
classifier_dict = dict()
class_priors = dict()
param_dict = dict() 
Q = 5
for label in classes:
    train = pd.read_csv("Dataset 2A/" + label + "/train.csv").to_numpy()[:, 1:].astype('float')
    train, params, proj_matrix = PCA(train)
    # pcaObj = PCA_red(3)
    # pcaObj.fit(train)
    # train = pcaObj.transform(train)

    gmm = GMM(Q, full=False)
    gmm.fit(train)
    classifier_dict[label] = gmm
    class_priors[label] = len(train)
    param_dict[label] = [params, proj_matrix]
    # param_dict[label] = pcaObj

tot = sum(class_priors.values())
for label in class_priors:
    class_priors[label] /= tot

all_val_labels = []
all_pred_labels = []
for test_on in classes:
    val = pd.read_csv("Dataset 2A/" + test_on + "/train.csv").to_numpy()[:, 1:].astype('float')
    params, proj_matrix = param_dict[test_on]
    val = PCA(val, params, proj_matrix)[0]
    # pcaObj = param_dict[label]
    # val = pcaObj.transform(val)
    
    all_val_labels.extend([test_on]*len(val))
    pred_labels = []
    for point in val:
        pred_labels.append(max(classifier_dict, key=lambda x: classifier_dict[x].class_prob(point)*class_priors[x]))
    
    all_pred_labels.extend(pred_labels)

plot_confusion_mat(all_pred_labels, all_val_labels, classes, "Q = 5 for all classes and Diagonal Covariance Matrix")
print("Accuracy =", sum(np.array(all_pred_labels) == np.array(all_val_labels))/len(all_pred_labels)*100, "%")


# test 66.571428
# train 84.5648

# test 59.4285
# train 73.56321