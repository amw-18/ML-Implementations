import numpy as np 
import pandas as pd
from collections import defaultdict


class bayesKNN:
    def __init__(self, dataset , K):
        self.dataset = dataset
        self.classes = list(set(dataset[:, -1]))
        self.K = K
        self.class_dict = defaultdict(list)
        self.N = len(dataset)
        for x in dataset:
            self.class_dict[x[-1]].append(x[:-1])

    def classify(self, points):
        pred_labels = []
        for point in points:
            class_R = {}
            for label in self.classes:
                class_examples = self.class_dict[label]
                dist = np.sort(np.linalg.norm(class_examples - point, axis=1))
                class_R[label] = dist[self.K - 1]

            pred_labels.append(min(class_R, key=class_R.get))

        return pred_labels

if __name__ == "__main__":
    data = pd.read_csv("Dataset 1B/train.csv", header=None).to_numpy()

    val = pd.read_csv("Dataset 1B/dev.csv", header=None).to_numpy()
    val_labels = val[:,-1]

    bayes_knn = bayesKNN(data, K=20)
    pred_labels = bayes_knn.classify(val[:,:-1])
    
    print("Accuracy =", sum(pred_labels == val_labels)/len(pred_labels)*100, "%")


        