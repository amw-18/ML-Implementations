from utils import *


# reading the data
df = pd.read_csv("2_music.txt", header=None).to_numpy()
X = df[:, :-2]
Y = df[:, -2:]

# determining train, val, test splits
# determining sizes
train_size, val_size = (np.array([0.7, 0.2])*X.shape[0]).astype(int)
test_size = X.shape[0] - train_size - val_size
# distributing data
Xtrain, Ytrain = X[:train_size], Y[:train_size]
Xval, Yval = X[train_size:train_size+val_size], Y[train_size:train_size+val_size]
Xtest, Ytest = X[train_size+val_size:], Y[train_size+val_size:]

# Model Training
# degs = np.arange(50, 60, 2).astype(int)
degs = [114]
# lamdas = [0, np.exp(-18), 1]
lamdas = [1]
regs = []
for deg in degs:
    for lamda in lamdas:
        reg = Regressor(deg, "Gaussian", regType="Quadratic", lamda=lamda, seed=0)
        reg.fit(Xtrain, Ytrain)
        Yval_Pred = reg.predict(Xval)
        val_rms = reg.rmsError(Yval, Yval_Pred)
        Ytest_Pred = reg.predict(Xtest)
        test_rms = reg.rmsError(Ytest, Ytest_Pred)

        regs.append((reg, val_rms, test_rms, Ytest, Ytest_Pred))
        print(regs[-1][0].rmsErr, val_rms, test_rms)
    
for reg in regs:
    print(f"{np.around(reg[0].deg,3)} & {np.around(reg[0].lamda,3)} & {np.around(reg[0].rmsErr,3)} & {np.around(reg[1],3)} & {np.around(reg[2],3)}\\\\")
    

# best = min(regs, key=lambda x: x[1])
# print("best:" , best[0].deg, best[0].lamda)


# plt.figure()
# plt.scatter(Ytrain[:,0], reg.y_Pred[:,0])
# plt.title("Original vs Predicted - Target variable 1 - Train set (Tikhonov)")
# plt.xlabel("Original values")
# plt.ylabel("Predicted values")
# plt.tight_layout()
# plt.grid()
# # plt.savefig("T3plots/T3_D3_train_v1_T")
# plt.figure()
# plt.scatter(Ytrain[:,1], reg.y_Pred[:,1])
# plt.title("Original vs Predicted - Target variable 2 - Train set (Tikhonov)")
# plt.xlabel("Original values")
# plt.ylabel("Predicted values")
# plt.tight_layout()
# plt.grid()
# # plt.savefig("T3plots/T3_D3_train_v2_T")
# plt.figure()
# plt.scatter(Ytest[:,0], Ytest_Pred[:,0])
# plt.title("Original vs Predicted - Target variable 1 - Test set (Tikhonov)")
# plt.xlabel("Original values")
# plt.ylabel("Predicted values")
# plt.tight_layout()
# plt.grid()
# # plt.savefig("T3plots/T3_D3_test_v1_T")
# plt.figure()
# plt.scatter(Ytest[:,1], Ytest_Pred[:,1])
# plt.title("Original vs Predicted - Target variable 2 - Test set (Tikhonov)")
# plt.xlabel("Original values")
# plt.ylabel("Predicted values")
# plt.tight_layout()
# plt.grid()
# # plt.savefig("T3plots/T3_D3_test_v2_T")
# plt.show()




