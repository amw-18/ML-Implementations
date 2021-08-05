import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PCA_red():
    def __init__(self,n_comp=10):
        self.n_comp = n_comp
        self.mu = None
        self.std = None
        self.Projector = None

    def fit(self,X):
        print(X.shape)
        self.mu = X.mean(axis=0).reshape(1,-1)
        self.std = X.std(axis=0).reshape(1,-1)
        X_New = (X-X.mean())/X.std()
        C = np.cov(X_New.T)
        lamda,v= np.linalg.eig(C)
        IndxArr=(-lamda).argsort()
        PrincBasis=v[:,IndxArr]
        FeaNo= self.n_comp
        DimRedcPrincBasis=PrincBasis[:,:FeaNo]
        self.Projector = DimRedcPrincBasis
        x_PrincTr=np.matmul(DimRedcPrincBasis.T,X_New.T)
        print(x_PrincTr.T.shape)

    def transform(self,Xinp):
        Xinp = (Xinp-self.mu)/self.std
        return np.matmul(self.Projector.T,Xinp.T).T


if __name__=="__main__":
    X = pd.read_csv("Dataset 2A/" + "coast" + "/train.csv").to_numpy()[:, 1:].astype('float')
    pcaObj = PCA_red(3)
    pcaObj.fit(X)
    print(X[4:9].shape)
    X_trans = pcaObj.transform(X[4:9])
    print(X_trans.shape)