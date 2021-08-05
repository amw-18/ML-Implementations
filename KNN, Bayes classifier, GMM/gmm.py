from utils import kmeans_clustering
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math

class GMM:
    def __init__(self, Q, full = True):
        self.Q = Q
        self.mix_weights = None
        self.means = None
        self.var = None
        self.logll = 0 # log likelihood
        self.d = None
        self.N = None
        self.resp_matrix = None
        self.full = full

    def fit(self, X):
        self.N, self.d = X.shape  # dimensionality of the data

        # Initializing the values
        self.initialize(X)
        
        threshold = 1e-04
        diff = np.inf
        while abs(diff) > threshold:
            print(diff)
            for n in range(self.N):
                for k in range(self.Q):
                    self.resp_matrix[n,k] = self.mix_weights[k]*self.get_gaussian_prob(X[n], self.means[k], self.var[k])
                self.resp_matrix[n,:] /= sum(self.resp_matrix[n,:])

            for k in range(self.Q):
                Nk = sum(self.resp_matrix[:, k])
                self.means[k] = np.sum(self.resp_matrix[:, k].reshape((self.N, 1))*X, axis=0)/Nk 
                self.var[k] = np.zeros((self.d, self.d))
                for n in range(self.N):
                    v = (X[n] - self.means[k]).reshape((self.d, 1))
                    self.var[k] += self.resp_matrix[n, k]*np.matmul(v, v.T)
                self.var[k] /= Nk
                if not self.full:
                    self.var[k] = self.var[k]*np.eye(self.d)
                self.mix_weights[k] = Nk/self.N

            old = self.logll/self.N
            self.update_logll(X)
            new = self.logll/self.N
            diff = old - new

    def get_gaussian_prob(self, point, mean, var):
        v = (point - mean).reshape((self.d, 1))
        return (np.exp(-0.5*np.matmul(v.T, np.matmul(np.linalg.inv(var), v)))/np.sqrt(((2*np.pi)**self.d)*np.linalg.det(var)))[0, 0]

    def initialize(self, X):
        self.means, z = kmeans_clustering(self.Q, X)
        self.mix_weights = np.array([sum(z[i])/len(z[i]) for i in range(self.Q)])
        self.var = []
        for q, mean in enumerate(self.means):
            Z = []
            for i in range(self.N):
                if z[q][i]:
                    Z.append(X[i])

            Z = np.array(Z)
            a = Z - mean
            var = np.matmul(a.T, a)/self.N

            if not self.full:
                var = var*np.eye(self.d)

            self.var.append(var)

        self.update_logll(X)
        self.resp_matrix = np.zeros((self.N, self.Q))
            
    def update_logll(self, X):
        self.logll = 0
        for x in X:
            temp = 0
            for k in range(self.Q):
                temp += self.mix_weights[k]*self.get_gaussian_prob(x, self.means[k], self.var[k])
            self.logll += np.log10(temp)
    
    def class_prob(self, point):
        ans = 0
        for k in range(self.Q):
            ans += self.mix_weights[k]*self.get_gaussian_prob(point, self.means[k], self.var[k])
        
        return ans

    def visualise(self, X, ax3d, ThreeDCnt):
        res = 200
        x1_min,x2_min = min(X[:,0]),min(X[:,1])
        x1_max,x2_max = max(X[:,0]),max(X[:,1])

        x1 = np.linspace(x1_min,x1_max, res)
        x2 = np.linspace(x2_min,x2_max, res)

        X1,X2 = np.meshgrid(x1,x2)
        
        score_mat = np.empty((res,res))
        
        for x_crd in range(res):
            for y_crd in range(res):
                score_mat[x_crd,y_crd] = self.class_prob([X1[x_crd,y_crd],X2[x_crd,y_crd]])

        # normP = lambda x: (x-np.min(x))/(np.max(x)-np.min(x))
        # picture_vis = normP(score_mat)
        #plt.figure(1)
        # plt.imshow(picture_vis)

        plt.figure(1)
        
        for mu, var in zip(self.means, self.var):
            plt.scatter(mu[0],mu[1],color = 'red')
            
        Leg2d = plt.contour(X1,X2,score_mat, levels = 10)
        plt.grid(True)

        Leg3d = self.visualise_3d(X,X1,X2,score_mat,ax3d,ThreeDCnt)
        return [Leg2d,Leg3d]

    def visualise_3d(self,X,x1,x2,z,ax,ThreeDCnt):
        
        AvailColors = np.array([[102,102,240],
                    [245,104,104],
                    [51,240,51],
                    [178,102,240],
                    [240,102,178]])/255

        cmapArr=["Blues","Oranges","Greens","Purples","Greys"]
        
        plt.figure(2)
        oSet = -2
        #ax.set_title(f"Plot for {self.basis} Basis Bivariate X matrix"  + "\n" + f"{self.regType} Regularisation, deg = {self.deg}, lambda = {self.lamda}")
        Leg3d = ax.contourf(x1, x2, z, 100, cmap = cmapArr[ThreeDCnt] , alpha = 0.8)
        ax.contour(x1,x2,z,zdir='z',cmap= cm.autumn, offset=oSet)
        ax.scatter(self.means[:,0],self.means[:,1], oSet*np.ones(self.means.shape[0]),
                   color = 'red', s=40)

        return Leg3d

class GMM2(GMM):
    def class_prob(self, point):
        ans = 0
        point = point.reshape((int(len(point)/23), 23))
        for feature in point:
            p = 0
            for k in range(self.Q):
                p += self.mix_weights[k]*self.get_gaussian_prob(feature, self.means[k], self.var[k])
            ans += np.log10(p)
        
        return ans



if __name__ == "__main__":
    df = pd.read_csv("Dataset 1B/train.csv", header=None)
    df_test = pd.read_csv("Dataset 1B/dev.csv", header=None).to_numpy()[2,:-1]
    df1 = df[df[df.columns[-1]] == 1.0].to_numpy()
    gmm1 = GMM(4, full=True)
    gmm1.fit(df1[:, :-1])
    print(gmm1.class_prob(df_test))
    gmm1.visualise(df1[:, :-1])
    

    