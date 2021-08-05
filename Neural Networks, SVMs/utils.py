import matplotlib.pyplot as plt
from numpy.lib.arraysetops import unique 
import seaborn as sns 
import numpy as np
from seaborn.utils import relative_luminance 

def predict1v1(X, unique_classes, clf_dict):
    pred_labels = []
    for x in X:
        x = x.reshape(1, -1)
        votes = [0]*len(unique_classes)
        for clf in clf_dict:
            label = clf_dict[clf].predict(x)
            votes[int(label)] += 1
        pred_labels.append(votes.index(max(votes)))

    pred_labels = np.array(pred_labels, dtype='float64')

    return pred_labels

def predict1vR(X, unique_classes, clf_dict):
    pred_labels = []
    for x in X:
        x = x.reshape(1, -1)
        probs = [0]*len(unique_classes)
        for label in clf_dict:
            p = clf_dict[label].predict_proba(x)[0][1]
            probs[int(label)] = p
        pred_labels.append(probs.index(max(probs)))

    pred_labels = np.array(pred_labels, dtype='float64')

    return pred_labels

def plot_confusion_mat(pred_labels, true_labels, classes, title=None):
    n_classes = len(classes)
    confusion_mat = np.zeros((n_classes, n_classes))
    n_examples = len(true_labels)

    for i in range(n_classes):
        for j in range(n_classes):
            confusion_mat[i, j] = sum([((pred_labels[k] == classes[j]) and (true_labels[k] == classes[i])) for k in range(n_examples)])
        confusion_mat[i] /= sum(confusion_mat[i])

    sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='.2%', xticklabels=classes, yticklabels=classes)
    if title:
        plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.show()


def plot_decision_regions(X, labels, model=None, modelType="perceptron"):
    
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
            x = np.array([X1[i,j],X2[i,j]]).reshape(1,-1)
            if modelType == "perceptron":
                pred_label = predict1v1(x, labels, model)

            elif modelType == "mlffnn":
                pred_label = model.predict(x)

            elif modelType == "svc_one":
                pred_label = predict1v1(x, labels, model)

            elif modelType == "svc_rest":
                pred_label = predict1vR(x, labels, model)

            else:
                raise Exception("Supported Model types are only 'perceptron' or 'svc_one' or 'svc_rest' or 'mlffnn' ")
            # max_conf = -1
            # for enum,label in enumerate(labels):
            #     if(modelType=="gmm"):
            #         val = gmmsOrNbg[label].class_prob([X1[i,j],X2[i,j]])
            #     elif(modelType=="nbg"):
            #         val = gmmsOrNbg[label][i,j]
            #     else:
            #         raise Exception("modelType error possible parameters are 'gmm' or 'nbg' ")
            #     if(val>max_conf):
            #         max_conf = val
            #         max_label = enum
            decision_boundary[i,j] = colorsArr[int(pred_label)]
    
    if modelType == "svc_one" or modelType == "svc_rest":
        svs = []
        for clf in model:
            svs.extend(model[clf].support_vectors_)
            
        svs = np.array(svs)
        plt.scatter(svs[:, 0], svs[:, 1], color="black", label="support vectors")


    #plt.contour(X1,X2,decision_boundary,levels=range(len(labels)))
    decision_boundary = np.rot90(decision_boundary,k=2)
    decision_boundary = np.flip(decision_boundary, axis=1)
    plt.figure(1)
    plt.imshow(decision_boundary,interpolation='bilinear',
                extent=[x1_min,x1_max,x2_min,x2_max], alpha =0.5)

def relU(X):
    return np.array([[max(0, y) for y in x] for x in X])

def partial_forward_pass(x, clf, N_HNeurons):
    if N_HNeurons != -1:
        for p in range(N_HNeurons):
            x = np.matmul(x, clf.coefs_[p]) + clf.intercepts_[p]
            x = relU(x)
    else:
        for p in range(clf.n_layers_ - 2):
            x = np.matmul(x, clf.coefs_[p]) + clf.intercepts_[p]
            x = relU(x)
        
        x = np.matmul(x, clf.coefs_[-1]) + clf.intercepts_[-1]
        x = np.exp(x)
        x /= sum(x)
 
    return x


def HiddenLayerHeatMaps(X, clf, N_HNeurons):
    res = 200
    x1_min,x2_min = min(X[:,0]),min(X[:,1])
    x1_max,x2_max = max(X[:,0]),max(X[:,1])

    x1 = np.linspace(x1_min,x1_max, res)
    x2 = np.linspace(x2_min,x2_max, res)

    X1,X2 = np.meshgrid(x1,x2)

    Z = None

    for i in range(res):
        for j in range(res):

            x = np.array([X1[i,j], X2[i,j]]).reshape(1,-1)

            NeuronRes = partial_forward_pass(x, clf, N_HNeurons)
            NNeurons = NeuronRes.shape[1]
            
            if Z is None:
                Z = np.empty((NNeurons,res,res))

            for k in range(NNeurons):
                Z[k, i, j] = NeuronRes[0, k]
    
    for heatmap in range(NNeurons):
        Z[heatmap] = np.rot90(Z[heatmap],k=2)
        Z[heatmap] = np.flip(Z[heatmap], axis=1)

    return Z, NNeurons

def VisualiseHiddenLayer(X, clf, N_HNeurons, ax):
    Z, NNeurons = HiddenLayerHeatMaps(X, clf, N_HNeurons)

    for j in range(NNeurons):
        ax[j].set_aspect(1)
        sns.heatmap(Z[j], ax=ax[j], xticklabels=False, yticklabels=False)
        # plt.figure()
        # sns.heatmap(Z[j])
    
    

def EpochViz(epochs, X, y, clf, unique_classes, layer_data):
    
    NNeurons, N_HNeurons = layer_data
    fig, ax = plt.subplots(NNeurons, len(epochs))

    for i in range(len(epochs)):
        if(i!=0):
            ExtraTrEp = epochs[i] - epochs[i-1]
        else:
            ExtraTrEp = epochs[0]

        for Sub_epNo in range(ExtraTrEp):
            clf.partial_fit(X, y, unique_classes)
        
        VisualiseHiddenLayer(X, clf, N_HNeurons, ax[:, i])

    fig.suptitle(f"Layer No. {N_HNeurons + 1}")
    plt.show()