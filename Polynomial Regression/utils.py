import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

class Regressor:
    def __init__(self, deg, basis, regType = "Quadratic", lamda = 0, seed=None):
        ''' 
        deg     = (type:int)    (default:<empty>)       (Description:Is polynomial degree / No of Gaussian basis, context depends on the basis)
        basis   = (type:str)    (default:<empty>)       (Description:["Polynomial","Gaussian"])
        regType = (type:str)    (default:"Quadratic")   (Description:["Quadratic","Tikhonov"])
        lamda   = (type:int)    (default:0)             (Description:Regularisation Coefficient)
        seed    = (type:int)    (default:None)          (Description:Seed Value for KMeans clustering used in Gaussian Basis)
        '''
        self.deg = deg
        self.basis = basis
        self.regType = regType
        self.lamda = lamda
        self.sigma = None
        self.mus = None
        self.seed = seed

    def fit(self, x, y):
        """
        Fits the model on the given dataset and other parameters
        """
        self.x = x
        self.y = y
        self.dataMat = self.getBasis(self.x)  # Design matrix stored as self.dataMat
        if self.regType == "Quadratic":
            A1 = np.matmul(self.dataMat.T, self.dataMat)
            A2 = (self.lamda)*np.eye(self.dataMat.shape[1])
            self.params = np.matmul(np.matmul(np.linalg.inv(A1 + A2), self.dataMat.T), y)  # Final params
            self.y_Pred = self.predict()
            self.rmsErr = self.rmsError()
        elif self.regType == "Tikhonov":
            mus = self.mus
            A1 = np.matmul(self.dataMat.T, self.dataMat)
            phi_Tild = np.array([[np.exp(-np.linalg.norm(mus[i] - mus[j])**2/self.sigma**2) for j in range(self.deg-1)] for i in range(self.deg-1)])
            phi_Tild = np.concatenate((np.zeros((phi_Tild.shape[0],1)),phi_Tild),axis=1)
            phi_Tild = np.concatenate((np.zeros((1,phi_Tild.shape[1])),phi_Tild),axis=0)
            A2 = (self.lamda)*phi_Tild
            self.phi_Tild = phi_Tild
            self.params = np.matmul(np.matmul(np.linalg.inv(A1 + A2), self.dataMat.T), y)  # Final params
            self.y_Pred = self.predict()
            self.rmsErr = self.rmsError()  
        
    def predict(self,passed_x = None):
        """
        Predicts the output based on input based on the trained model
        """
        if passed_x is None:
            y_Pred = np.matmul(self.dataMat, self.params)
        else:
            PredMat = self.getBasis(passed_x)
            y_Pred = np.matmul(PredMat, self.params)
        return y_Pred

    def rmsError(self, passed_y=None, passed_y_Pred=None):
        """
        Returns the RMS Error based on predicted Vs True values
        """
        if passed_y is None:
            passed_y = self.y
            passed_y_Pred = self.y_Pred

        rmsErr = np.sqrt(np.sum((passed_y-passed_y_Pred)**2)/passed_y.shape[0])
        return rmsErr

    def getBasis(self, passed_x):
        """
        Transforms the input features to based on the basis function type and its parameters to return new feature
        constructed matrix
        """
        self_x = passed_x
        if self.basis=='Polynomial':
            #
            if(self_x.shape[1]==1):
                dataMat = np.ones((self_x.shape[0],1))
                for i in range(1,self.deg+1):
                    dataMat = np.concatenate((dataMat,self_x**i),axis=1)
                
            elif(self_x.shape[1]==2):
                dataMat = np.ones((self_x.shape[0],1))
                for i in range(self.deg+1):
                    for j in range(self.deg+1):
                        if((i+j<=self.deg) and (not ((i==0)and(j==0)))):
                            dataMat = np.concatenate((dataMat,((self_x[:,0]**i)*(self_x[:,1]**j)).reshape(-1,1)),axis=1)
            return dataMat

        elif self.basis=='Gaussian':
            if self.mus is None:
                self.mus, z = self.kmeans_clustering(self.deg - 1, self_x)
                
                # Storing Cluster Indices for later use especially in visualisation part
                self.ClusterIndices = z

                norm_sq = 0
                for j in range(len(self.mus)):
                    for i in range(self_x.shape[0]):
                        if z[j][i] == 1:
                            norm_sq += np.linalg.norm(self_x[i] - self.mus[j])**2

                self.sigma = np.sqrt(norm_sq/self_x.shape[0])*10
                # self.sigma = 19

            dataMat = np.ones((self_x.shape[0], self.deg))
            for i in range(self_x.shape[0]):
                phi = np.exp(-np.linalg.norm(self_x[i,:] - self.mus, axis=1)**2/self.sigma**2) 
                dataMat[i,1:] = phi
            return dataMat


    def kmeans_clustering(self, k, passed_x):
        """
        Calling this function invokes k means clustering for passed feature matrix
        """
        # Matrix to keep track of clusters
        z = [[0]*passed_x.shape[0] for i in range(k)]

        # mus = passed_x[:k].copy()
        # # mus = np.float(mus)
        # for j in range(k):
        #     z[j][j] = 1

        # Initializing cluster centers
        mus = np.zeros((k, passed_x.shape[1]))
        to_choose = list(range(passed_x.shape[0]))
        if(self.seed is not None):
            random.seed(self.seed)
        chosen = random.choices(to_choose, k=k)
        for i in range(k):
            z[i][chosen[i]] = 1
            mus[i] = passed_x[chosen[i]][:]

        # Flag
        update = True
        while update:
            update = False
            for i in range(passed_x.shape[0]):
                norm = np.linalg.norm(mus - passed_x[i], axis=1)
                assigned_cluster = np.argmin(norm)
                # Cluster assignment
                for j in range(k):
                    if j != assigned_cluster:
                        z[j][i] = 0
                    else:
                        if z[assigned_cluster][i] != 1:
                            update = True
                        z[assigned_cluster][i] = 1
            
            # Updating the cluster centers
            for j in range(k):
                Nj = sum(z[j])
                if Nj == 0:
                    continue
                my_sum = np.array([0.0]*passed_x.shape[1])
                for i in range(passed_x.shape[0]):
                    my_sum += z[j][i]*passed_x[i]

                mus[j] = my_sum/Nj

        return mus, z

    def VisualiseClusters(self):
        '''
        Call this function to Visualise the clusters along with cluster centres
        '''
        z = self.ClusterIndices
        if(self.x.shape[1]==2):
            k = len(z)
            colors = np.random.rand(k,3)
            indxArr = np.array(z)
            colorRegister = np.empty((self.x.shape[0],3))

            FullcolorIndxPalete = np.arange(k)

            for i in range(self.x.shape[0]):
                colorRegister[i] = colors[FullcolorIndxPalete[indxArr[:,i]==1]]

            plt.figure()
            plt.scatter(self.x[:,0],self.x[:,1],color = colorRegister)
            plt.scatter(self.mus[:,0],self.mus[:,1],color = 'black',marker = 'v', label="cluster centers")
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.legend()
            plt.title('Clusters Visualised with cluster centers')
            # plt.savefig("T3plots/T3_D2_cluster_visualization_T.jpg")

            plt.show()


    def Visualise(self):
        '''
        Call this function to Visualise the results of fitted function Vs original values
        '''
        if self.x.shape[1]==1:
            L_extre, R_extre = np.min(self.x),np.max(self.x)
            X_Generated = np.linspace(L_extre,R_extre,num=int((R_extre-L_extre)*10)).reshape(-1,1)
            y_Predicted = self.predict(passed_x=X_Generated)
            #Plotting 2D
            plt.figure()
            plt.title(f"Plot for {self.basis} Basis Univariate X matrix" + "\n" + f"{self.regType} Regularisation, deg = {self.deg}, lambda = {self.lamda}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.plot(X_Generated,y_Predicted,color = 'b', linewidth = 1.5 )
            plt.scatter(self.x,self.y, color= 'r')
            plt.legend(["Predicted Function","Original Datapoints"])
            plt.grid(True)
            # plt.savefig(f"T1plots/T1_{self.deg}_{np.around(self.lamda,decimals=2)}_{self.x.shape[0]}.jpg")
            #Display Result
            plt.show()
            
        elif self.x.shape[1]==2:
            L_extre_1, R_extre_1 = np.min(self.x[:,0]),np.max(self.x[:,0])
            L_extre_2, R_extre_2 = np.min(self.x[:,1]),np.max(self.x[:,1])
            X1_Generated = np.linspace(L_extre_1,R_extre_1,num=int((R_extre_1-L_extre_1)*10)).reshape(-1,1)
            X2_Generated = np.linspace(L_extre_2,R_extre_2,num=int((R_extre_2-L_extre_2)*10)).reshape(-1,1)

            J1,J2 = np.meshgrid(X1_Generated,X2_Generated)
            
            X_Generated = np.empty((J1.size,2))
            for i in range(J1.shape[0]):
                X_Generated[i*J1.shape[1]:(i+1)*J1.shape[1]] = np.concatenate((J1[i].reshape(-1,1),J2[i].reshape(-1,1)),axis=1)

            y_Predicted = self.predict(passed_x=X_Generated)
            #Reshape y_Predicted for 3D plotting purpose
            Z = y_Predicted.reshape(*J1.shape)

            #Plotting 2D
            plt.figure()
            ax = plt.axes(projection = '3d')
            ax.set_title(f"Plot for {self.basis} Basis Bivariate X matrix"  + "\n" + f"{self.regType} Regularisation, deg = {self.deg}, lambda = {self.lamda}")
            ax.plot_wireframe(J1, J2, Z, color='green')
            # ax.plot_surface(J1, J2, Z, color='green')
            
            ax.scatter(self.x[:,0], self.x[:,1], self.y, color='r', marker='o', s=15, alpha = 1.0)
            ax.legend(["Predicted Function","Original Datapoints"],loc=(0.55,0.8))

            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_zlabel('Y')
            
            # plt.savefig(f"T3plots/T3_D2_{self.deg}_{np.around(self.lamda,decimals=2)}_T.jpg")
            #Display Result
            plt.show()
    
def crossVal(X, Y, deg, basis, regType = "Quadratic", lamda = 0, kFold=3):
    """
    Performs Kfold cross validation and returns the best model
    """
    pieceSize = int(X.shape[0]/kFold)
    x_Pieces = [None]*kFold
    y_Pieces = [None]*kFold
    for i in range(kFold):
        if(i!=kFold-1):
            x_Pieces[i], y_Pieces[i] = X[i*pieceSize:(i+1)*pieceSize], Y[i*pieceSize:(i+1)*pieceSize]
        else:
            x_Pieces[i], y_Pieces[i] = X[i*pieceSize:], Y[i*pieceSize:]

    regs = [None]*kFold
    val_rms_error = [None]*kFold
    train_rms_error = [None]*kFold
    for i in range(kFold):
        Xval, Yval = x_Pieces[i], y_Pieces[i]
        Xtrain, Ytrain = [], []
        for j in range(kFold):
            if j!=i:
                Xtrain.extend(x_Pieces[j])
                Ytrain.extend(y_Pieces[j])
        regs[i] = Regressor(deg, basis, regType, lamda)
        regs[i].fit(np.array(Xtrain), np.array(Ytrain))
        predictions = regs[i].predict(Xval)
        val_rms_error[i] = regs[i].rmsError(Yval, predictions)
        train_rms_error[i] = regs[i].rmsErr
    
    bestModIndx = np.argmax(np.array(val_rms_error))
    bestModRMS = val_rms_error[bestModIndx]

    return (regs[bestModIndx], bestModRMS, sum(val_rms_error)/kFold)


if __name__ == "__main__":
    data_df = pd.read_csv("function1_2d.csv",index_col = 0)
    x = data_df[:200].to_numpy()[:,:-1]
    y = data_df[:200].to_numpy()[:,-1]

    reg = Regressor(40, "Gaussian", lamda=0)
    reg.fit(x, y)
    xtest, ytest = data_df[200:230].to_numpy()[:,:-1], data_df[200:230].to_numpy()[:,-1]
    ytest_pred = reg.predict(xtest)
    rms = reg.rmsError(ytest, ytest_pred)
    # reg.Visualise()
    print(reg.rmsErr, rms)
    
    
    