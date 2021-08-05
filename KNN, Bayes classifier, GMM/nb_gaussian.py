import numpy as np 
from collections import defaultdict
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math

class NBGaussian:
    def __init__(self, type = 3):
        self.params = dict()  # dictionary with labels as key
        self.n_classes = None
        self.d = None
        self.class_dict = None
        self.class_priors = dict()
        self.N = None
        self.type = type # type 1 or 2 or 3 (default) according to the assignment statement
        self.computedClassesScore = None


    def fit(self, dataset):
        self.class_dict = defaultdict(list)
        self.N = len(dataset)
        for x in dataset:
            self.class_dict[x[-1]].append(x[:-1])
        
        self.n_classes = len(self.class_dict)

        for i in self.class_dict.keys():
            class_examples = np.array(self.class_dict[i])

            self.d = len(class_examples[0])
            Ni = len(class_examples)
            self.class_priors[i] = Ni/self.N

            mean = np.mean(class_examples, axis=0)

            var = np.zeros((self.d, self.d))
            for x in class_examples:
                v = (x - mean).reshape((self.d, 1))
                var += np.matmul(v, v.T)

            var /= Ni
            var *= np.eye(self.d)
            self.params[i] = [mean, var]

        if self.type == 1:
            var = np.zeros((self.d, self.d))
            for label in self.params:
                var += self.params[label][1]*self.class_priors[label]

            var = (np.sum(var)/self.d)*np.eye(self.d)

            for label in self.params:
                self.params[label][1] = var

        elif self.type == 2:
            var = np.zeros((self.d, self.d))
            for label in self.params:
                var += self.params[label][1]*self.class_priors[label]

            for label in self.params:
                self.params[label][1] = var
        
        elif self.type == 3:
            pass
        else: 
            raise TypeError("Invalid type")

    def get_prob(self, point, label):
        mean, var = self.params[label]
        v = (point - mean).reshape((self.d, 1))
        return (np.exp(-0.5*np.matmul(v.T, np.matmul(np.linalg.inv(var), v)))/np.sqrt(((2*np.pi)**self.d)*np.linalg.det(var)))[0, 0]


    def classify(self, points):
        pred_labels = []
        for point in points:
            class_prob = dict()
            for label in self.class_dict:
                class_prob[label] = self.get_prob(point, label)*self.class_priors[label]

            pred_labels.append(max(class_prob, key=class_prob.get))

        return pred_labels

    def compute_ClassesScore(self, X):
        res = 200
        score_acc={}
        
        for clsKey in self.class_dict.keys():
            score_acc[clsKey]= np.empty((res,res))

        x1_min,x1_max = np.min(X[:,0]),np.max(X[:,0])
        x2_min,x2_max = np.min(X[:,1]),np.max(X[:,0])

        x1 = np.linspace(x1_min, x1_max, res)
        x2 = np.linspace(x2_min, x2_max, res)

        X1,X2 = np.meshgrid(x1,x2)
        self.compResX1=X1
        self.compResX2=X2

        for clsKey in self.class_dict.keys():
            for i in range(res):
                for j in range(res):
                    score_acc[clsKey][i,j] = self.get_prob([X1[i,j],X2[i,j]],clsKey)
        
        self.computedClassesScore = score_acc


    def visualise(self, X, ThreeD_ax):
        LegArr2d=[]
        if(self.computedClassesScore is None):
            self.compute_ClassesScore(X)
        
        plt.figure(1)
        for clsKey in sorted(self.class_dict.keys()):
            leg2d=plt.contour(self.compResX1,self.compResX2, self.computedClassesScore[clsKey])
            LegArr2d.append(leg2d)
            #Plot mean points
            plt.scatter(self.params[clsKey][0][0],self.params[clsKey][0][1],color = 'r')
        plt.grid(True)
        LegArr3d = self.visualise_3d(X,self.compResX1,self.compResX2,self.computedClassesScore,ThreeD_ax)
        
        return (LegArr2d,LegArr3d)

    def visualise_3d(self,X,x1,x2,z,ax):
        LegArr3d=[]
        AvailColors = np.array([[102,102,240],
                    [245,104,104],
                    [51,240,51],
                    [178,102,240],
                    [240,102,178]])/255

        cmapArr=["Blues","Greens","Oranges","Purples","Greys"]

        plt.figure(2)
        #ax.set_title(f"Plot for {self.basis} Basis Bivariate X matrix"  + "\n" + f"{self.regType} Regularisation, deg = {self.deg}, lambda = {self.lamda}")
        oSet = -0.3
        ax.scatter(X[:,0],X[:,1],oSet*np.ones(X.shape[0]), s=5, color = 'black', alpha=0.1)
        for enm,clsKey in enumerate(sorted(self.class_dict.keys())):
            leg3d =ax.contourf(x1, x2, z[clsKey], 100, cmap = cmapArr[enm], alpha = 0.8)
            LegArr3d.append(leg3d)
            ax.contour(x1,x2,z[clsKey],zdir='z',cmap= cm.autumn, offset=oSet)
            ax.scatter(self.params[clsKey][0][0],self.params[clsKey][0][1],oSet*np.ones(self.params[clsKey][0].shape[0]),
                    color = 'red', s=40)

        return LegArr3d

if __name__ == "__main__":
    data = pd.read_csv("Dataset 1B/train.csv", header=None).to_numpy()

    val = pd.read_csv("Dataset 1B/dev.csv", header=None).to_numpy()
    val_labels = val[:,-1]

    nbg = NBGaussian(type = 1)
    nbg.fit(data)
    pred_labels = nbg.classify(val[:,:-1])
    