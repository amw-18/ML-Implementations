from gmm import GMM2
from utils import plot_confusion_mat
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


classes = ["coast", "insidecity", "mountain", "opencountry", "street"]
# classes = ["coast", "insidecity"]
classifier_dict = dict()
Q = 2
for label in classes:
    train = pd.read_csv("Dataset 2B CSV/train/" + label + ".csv", index_col=0).to_numpy().astype('float')
    train = train.reshape((int(train.size/23), 23))
    gmm = GMM2(Q, full=False)
    gmm.fit(train)
    classifier_dict[label] = gmm

# classes = ["coast", "insidecity", "mountain", "opencountry", "street"]
all_val_labels = []
all_pred_labels = []
for test_on in classes:
    val = pd.read_csv("Dataset 2B CSV/train/" + test_on + ".csv", index_col=0).to_numpy().astype('float')
    all_val_labels.extend([test_on]*len(val))
    pred_labels = []
    for point in val:
        ps = list(classifier_dict[x].class_prob(point) for x in classifier_dict)
        pred_labels.append(max(classifier_dict, key=lambda x: classifier_dict[x].class_prob(point)))
    
    all_pred_labels.extend(pred_labels)

plot_confusion_mat(all_pred_labels, all_val_labels, classes, "Q = 2 for Diagonal Covariance Matrices")
print("Accuracy =", sum(np.array(all_pred_labels) == np.array(all_val_labels))/len(all_pred_labels)*100, "%")


# test 65.1428
# train 68.7192

# test 68.57142
# train 67.898
