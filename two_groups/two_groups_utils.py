# imports from installed libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import pandas as pd
import scipy
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# models
import GPy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# import some functions


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def generate_response(X, y_noise_scale):
    
    if type(X) is list:
        Y = []
        for i in range(len(X)):
            x = X[i]
            Y.append(sigmoid(x) + y_noise_scale* np.random.normal( size = len(x))  )
        return Y
    else:
        # X is (n,)
        return sigmoid(X) + y_noise_scale*np.random.normal(size = len(X.reshape(-1)))

    
def generate_visualization_data(nseq, means, stds, y_noise_scale, test_mv):
    K = 2
    Xcal = [np.random.normal(loc = means[i], scale = stds[i], 
                          size = nseq[i]).reshape(-1)  for i in range(K)]
    Ycal = generate_response(Xcal, y_noise_scale)
   

    # check coverage
    Xtest = np.random.normal(loc = test_mv[0], scale = test_mv[1], size = max(nseq)).reshape(-1)
    Ytest = generate_response(Xtest, y_noise_scale)
    
    # lists of np.array with dim (n,)
    return  Xcal, Ycal,  Xtest, Ytest
    
    
def generate_data(nseq, means, stds, y_noise_scale, test_mv):
    K = 2
    Xcal = [np.random.normal(loc = means[i], scale = stds[i], 
                          size = nseq[i]).reshape(-1)  for i in range(K)]
    Ycal = generate_response(Xcal, y_noise_scale)
   
    # check coverage
    Xtest = np.random.normal(loc = test_mv[0], scale = test_mv[1], size = 1).reshape(-1)
    Ytest = generate_response(Xtest, y_noise_scale)
    
    # lists of np.array with dim (n,)
    return  Xcal, Ycal,  Xtest[0], Ytest[0]
    
    
def replications(R, f, w, w_pool,  nseq, means, stds, test_mv, scale ):
    alpha = 0.1
    mean_test, std_test = test_mv[0], test_mv[1]
    
    num_methods = 5
    cov_mat = np.zeros((R, num_methods))
    len_mat = np.zeros((R, num_methods))
    fin_mat = np.zeros((R, num_methods))
    
    for i in range(R):
        Xcal, Ycal, x, y = generate_data(nseq, means, stds, scale, test_mv)
        # check coverage record length
        PIs = get_intervals(x, f, Xcal, Ycal, w, w_pool, nseq, means, stds, test_mv, alpha)
        cov_mat[i,:] = (PIs[:,0] <= y)*(PIs[:,1] >= y)
        len_mat[i,:] = PIs[:,1] - PIs[:,0]
        fin_mat[i,:] = ~np.isinf(PIs[:,1] - PIs[:,0])
    
    return cov_mat, len_mat, fin_mat
    


    
    

def w(x, mean_train, std_train, mean_test, std_test):
    return np.exp(-0.5/std_test**2*(x - mean_test)**2 + 0.5/std_train**2*(x - mean_train)**2)
    
def w_pool(x, nseq, means, stds, mean_test, std_test):
    
    gw = nseq/np.sum(nseq)
    K = len(nseq)
    dens = np.zeros_like(x)
    
    for k in range(K):
        mean = means[k]
        std = stds[k]
        dens += gw[k]*1.0/std*np.exp(-0.5*( (x-mean)/std )**2)
    
    return np.exp(-0.5/std_test**2*(x-mean_test)**2)/(dens)


def WCP_pool(x, f, Xcal_all, Ycal_all, w_pool, nseq, means, stds, mean_test, std_test, alpha):
    
    # xcal_all: one row
    scores = np.abs(f(Xcal_all.reshape(-1,1))[0] - Ycal_all.reshape(-1,1))
    scores = np.append(scores.reshape(-1), np.inf)
    
    # compute weights
    wall = np.append(w_pool(Xcal_all, nseq, means, stds, mean_test, std_test), w_pool(x, nseq, means, stds, mean_test, std_test))
    # compute quantiles
    qhat = wquantile(scores, wall, 1 - alpha)
 
    return np.array([f(np.array([x]).reshape(-1,1))[0] - qhat, 
                     f(np.array([x]).reshape(-1,1))[0] + qhat]).reshape(-1)
    
   
    
def wquantile(values, weights, level):
    # values: np.array

    data = np.column_stack((values, weights) )     
    sorted_data = data[data[:, 0].argsort()]
    # compute cumulative weights
    cumulative_weights = np.cumsum(sorted_data[:, 1])

    # normalize cumulative weights
    normalized_weights = cumulative_weights / np.sum(sorted_data[:, 1])
    index = np.argmax(normalized_weights >= level)
    
    # return value
    return sorted_data[index,0]


    
    
####################################### wcp functions

def WCP(x, f, X, Y, w, mean_train, std_train, mean_test, std_test,alpha):
    # x: scalar
    # w: likelihood function
    scores = np.abs(f(X.reshape(-1,1))[0] - Y.reshape(-1,1))
    scores = np.append(scores.reshape(-1), np.inf)
    
    # compute weights
    wall = np.append(w(X, mean_train, std_train,
                       mean_test, std_test), w(x, mean_train, std_train, mean_test, std_test))
    # compute quantiles
    qhat = wquantile(scores, wall, 1 - alpha)
  
    return np.array([f(np.array([x]).reshape(-1,1))[0] - qhat, 
                     f(np.array([x]).reshape(-1,1))[0] + qhat]).reshape(-1)

def WCPq(x, f, X, Y, w, mean_train, std_train, mean_test, std_test,alpha):
    # x: scalar
    # w: likelihood function
    scores = np.abs(f(X.reshape(-1,1))[0] - Y.reshape(-1,1))
    scores = np.append(scores.reshape(-1), np.inf)
    
    # compute weights
    wall = np.append(w(X, mean_train, std_train,
                       mean_test, std_test), w(x, mean_train, std_train, mean_test, std_test))
    
    return wquantile(scores, wall, 1 - alpha)
    
    
    
def get_intervals(x, f, Xcal, Ycal, w, w_pool, nseq, means, stds, test_mv, alpha = 0.1):
    # x: scalar
    # f: trained model to do prediction f()
    
    mean_test, std_test = test_mv[0], test_mv[1]
    
    # two WCP intervals
    WCPs = []
    PIlist = []
  
    for i in range(2):
        mean_train, std_train = means[i], stds[i]
        Xcali = Xcal[i]
        Ycali = Ycal[i]
        wcpi = WCP(x, f, Xcali, Ycali, w, mean_train, std_train, mean_test, std_test, alpha)
        PIlist.append( wcpi)
        WCPs.append(wcpi)
        
    # list of wcp intervals and find shorter one
    WCPs = np.vstack(WCPs)
    lengths = WCPs[:,1] - WCPs[:,0]
    short_index = np.argmin(lengths)
    PIlist.append(WCPs[short_index,:])
   
 
    # Bonferroni's correction
    # wcp intervals at level alpha/2
    WCPs = []
    for i in range(2):
        mean_train, std_train = means[i], stds[i]
        Xcali = Xcal[i]
        Ycali = Ycal[i]
        WCPs.append( WCP(x, f, Xcali, Ycali, w, mean_train, std_train, mean_test, std_test, alpha*1.0/2) )
                  
    # stack by row
    WCPs = np.vstack(WCPs)
    lengths = WCPs[:,1] - WCPs[:,0]
    short_index = np.argmin(lengths)
    PIlist.append(WCPs[short_index,:])
    
    
    
    # pool intervals
    Xcal_all = np.concatenate(Xcal)
    Ycal_all = np.concatenate(Ycal)
    PIlist.append(WCP_pool(x, f, Xcal_all, Ycal_all, w_pool, nseq, means, stds, mean_test, std_test, alpha))
    
    
    
    return np.vstack(PIlist)





def Visualize_observations_model(Xcal, Ycal, Xtest, Ytest, f, model_name):
    
    # f: model trained on training data
    x = np.linspace(-10, 10, 100)
    y = sigmoid(x)

    # Create sigmoid line plot
    plt.figure(figsize=(8, 6))

    yfit,_ = f(x.reshape(-1,1))
    plt.plot(x, yfit.reshape(-1), linestyle = 'dashdot', color='black',
             linewidth= 4, alpha = 0.8, label= model_name)
  
        
        
    # add scatter points for group 1
    plt.scatter(Xcal[0], Ycal[0], color='blue', s=50, label='Group 1', alpha=0.6)

    # add scatter points for group 2
    plt.scatter(Xcal[1], Ycal[1] ,color='red', s=50, label='Group 2', alpha=0.6)

    plt.xlabel('Covariate', fontsize=20)
    plt.ylabel('Response', fontsize=20)
    plt.title('Visualization of observed groups', fontsize=24, y = 1.02)
    
    
    plt.yticks(np.arange(-0.25, 1.3, 0.25))
    plt.xticks(np.arange(-9, 10.1, 3))
     # set plot limits
    plt.xlim([-10, 10])
    plt.ylim([-0.25, 1.3])
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc = 'lower right', fontsize = 20)


    
    
def Visualize_test_group( Xtest, Ytest):
    
    # f: model trained on training data
    x = np.linspace(-10, 10, 100)
    y = sigmoid(x)

    # Create sigmoid line plot
    plt.figure(figsize=(8, 6))

    plt.plot(x, y, color='black', linestyle = '--', 
             linewidth=4, alpha = 0.8, label='E(Y|X)')
  

    # add scatter points for group 2
    plt.scatter(Xtest, Ytest ,color='gray', marker = '*', s=80, label='Group 0', alpha=0.6)

    plt.xlabel('Covariate', fontsize=20)
    plt.ylabel('Response', fontsize=20)
    plt.title('Visualization of new group', fontsize=24, y = 1.02)
    
    
    plt.yticks(np.arange(-0.25, 1.3, 0.25))
    plt.xticks(np.arange(-9, 10.1, 3))
     # set plot limits
    plt.xlim([-10, 10])
    plt.ylim([-0.25, 1.3])
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc = 'lower right', fontsize = 20)
    
    

