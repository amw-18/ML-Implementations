import numpy as np 
import pandas as pd 

class KNNclassifier:
    def __init__(self, dataset, K):
        """
        dataset : each column is a training example (numpy array of arrays)
        K : no. of nearest neighbors to consider
        """
        self.dataset = dataset # last column is labels
        self.classes = list(set(dataset[:, -1]))
        self.K = K

    def classify(self, points):
        """
        Classifies new datapoints according to knn rule.
        """
        labels = []
        dist = np.zeros((self.dataset.shape[0], 1))
        X = list(np.concatenate((self.dataset, dist), axis=1))
        for point in points:
            for example in X:
                example[-1] = np.linalg.norm(example[:-2] - point)

            X.sort(key=lambda x: x[-1])
            best_K_labels = [x[-2] for x in X[:self.K]]
            d = self.to_prob(best_K_labels)
            best_label = max(d, key=d.get)
            labels.append(best_label)

        return labels


    def to_prob(self, best_K_labels):
        """
        Calculates probabilites of classes from labels of K nearest neighbors.
        """
        d = {}
        for label in best_K_labels:
            if not (label in d):
                d[label] = 1
            else:
                d[label] += 1

        for label in d:
            d[label] /= self.K

        return d
        
            

if __name__ == "__main__":
    data = pd.read_csv("Dataset 1B/train.csv", header=None).to_numpy()

    val = pd.read_csv("Dataset 1B/dev.csv", header=None).to_numpy()
    val_labels = val[:,-1]

    knn = KNNclassifier(data, K=200)
    pred_labels = knn.classify(val[:,:-1])
    
    print("Accuracy =", sum(pred_labels == val_labels)/len(pred_labels)*100, "%")