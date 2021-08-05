from utils import *



# reading the data
df = pd.read_csv("function1_2d.csv",index_col = 0).to_numpy()
X = df[:, :-1]
Y = df[:, -1]

# determining train, val, test splits
# determining sizes
train_size, val_size = (np.array([0.7, 0.2])*X.shape[0]).astype(int)
test_size = X.shape[0] - train_size - val_size
# distributing data
Xtrain, Ytrain = X[:train_size], Y[:train_size]
Xval, Yval = X[train_size:train_size+val_size], Y[train_size:train_size+val_size]
Xtest, Ytest = X[train_size+val_size:], Y[train_size+val_size:]

# Model Training
reg = Regressor(30, "Gaussian", lamda=0, regType="Quadratic", seed=14)
reg.fit(Xtrain, Ytrain)
Yval_Pred = reg.predict(Xval)
val_rms = reg.rmsError(Yval, Yval_Pred)
Ytest_Pred = reg.predict(Xtest)
test_rms = reg.rmsError(Ytest, Ytest_Pred)
print(reg.rmsErr, val_rms, test_rms)

#Visualisations
reg.Visualise()
reg.VisualiseClusters()
plt.figure()
plt.scatter(Ytrain, reg.y_Pred)
plt.title("Original vs Predicted for train set (Tikhonov)")
plt.xlabel("Original target values")
plt.ylabel("Predicted target values")
plt.tight_layout()
plt.grid()
# plt.savefig("T3plots/T3_D2_train_T.jpg")
plt.figure()
plt.scatter(Ytest, Ytest_Pred)
plt.title("Original vs Predicted for test set (Tikhonov)")
plt.xlabel("Original target values")
plt.ylabel("Predicted target values")
plt.tight_layout()
plt.grid()
# plt.savefig("T3plots/T3_D2_test_T.jpg")
plt.show()


#seed rms error
#14 108.39511525248552
#141 142.43931736023845
#223 117.22195075165256
#357 152.1849269592288
#451 153.86490427015337