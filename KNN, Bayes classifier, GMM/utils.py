import numpy as np
import random 
import matplotlib.pyplot as plt
import seaborn as sns 


def kmeans_clustering(k, passed_x, seed=42):
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
        if(seed is not None):
            random.seed(seed)
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

def standardize(X, params=None):  
    rows, columns = X.shape
    
    X_std = np.zeros(shape=(rows, columns))
    temp = np.zeros(rows)

    if params is None:
        params = []
        for column in range(columns):
            mean = np.mean(X[:,column])
            std = np.std(X[:,column])
            params.append([std, mean])
            temp = np.empty(0)
            
            for element in X[:,column]:
                
                temp = np.append(temp, ((element - mean) / std))
    
            X_std[:,column] = temp
        
    else:
        for column in range(columns):
            std, mean = params[column]
            temp = np.empty(0)
            
            for element in X[:,column]:
                
                temp = np.append(temp, ((element - mean) / std))
    
            X_std[:,column] = temp
        
    return X_std, params


def PCA(X, params=None, projection_matrix=None):
    X, params = standardize(X, params)
    if projection_matrix is None:
        cov = np.cov(X.T)
        eigen_values, eigen_vectors = np.linalg.eig(cov)

        i = 24
        projection_matrix = (eigen_vectors.T[:][:i]).T

    X_pca = X.dot(projection_matrix)

    return X_pca, params, projection_matrix


def plot_confusion_mat(pred_labels, true_labels, classes,title=None):
    n_classes = len(classes)
    confusion_mat = np.zeros((n_classes, n_classes))
    n_examples = len(true_labels)

    for i in range(n_classes):
        for j in range(n_classes):
            confusion_mat[i, j] = sum([((pred_labels[k] == classes[j]) and (true_labels[k] == classes[i])) for k in range(n_examples)])
        confusion_mat[i] /= sum(confusion_mat[i])

    sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='.2%')
    if title:
        plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


def plot_decision_regions(X,labels,gmmsOrNbg=None, modelType="gmm"):
    
    AvailColors = np.array([[102,102,240],
                    [245,104,104],
                    [51,240,51],
                    [178,102,240],
                    [240,102,178]])/255

    if(len(labels)<=len(AvailColors)):
        colorsArr = AvailColors
    else:
        colorsArr = np.random.rand(len(labels),3)

    res = 200

    x1_min,x2_min = min(X[:,0]),min(X[:,1])
    x1_max,x2_max = max(X[:,0]),max(X[:,1])

    x1 = np.linspace(x1_min,x1_max, res)
    x2 = np.linspace(x2_min,x2_max, res)

    X1,X2 = np.meshgrid(x1,x2)
        
    decision_boundary = np.empty((res,res,3))

    for i in range(res):
        for j in range(res):
            max_conf = -1
            for enum,label in enumerate(labels):
                if(modelType=="gmm"):
                    val = gmmsOrNbg[label].class_prob([X1[i,j],X2[i,j]])
                elif(modelType=="nbg"):
                    val = gmmsOrNbg[label][i,j]
                else:
                    raise Exception("modelType error possible parameters are 'gmm' or 'nbg' ")
                if(val>max_conf):
                    max_conf = val
                    max_label = enum
            decision_boundary[i,j] = colorsArr[max_label]
    
    #plt.contour(X1,X2,decision_boundary,levels=range(len(labels)))
    decision_boundary = np.rot90(decision_boundary,k=2)
    decision_boundary = np.flip(decision_boundary, axis=1)
    plt.figure(1)
    plt.imshow(decision_boundary,interpolation='bilinear',
                extent=[x1_min,x1_max,x2_min,x2_max], alpha =0.5)

