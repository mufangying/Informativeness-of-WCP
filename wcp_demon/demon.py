# imports from installed libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# models
import GPy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# define some functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def generate_response(X, y_scale):
    
    if type(X) is list:
        Y = []
        for i in range(len(X)):
            x = X[i]
            Y.append(sigmoid(x) + y_scale* np.random.normal( size = len(x))  )
        return Y
    else:
        # X is (n,)
        return sigmoid(X) + y_scale*np.random.normal(size = len(X.reshape(-1)))

    

    
def generate_data( n, mean_train, std_train, mean_test, std_test, y_noise_scale):
   
    # generate labeled calibration data
    Xtr = np.random.normal(loc = mean_train, scale = std_train, 
                          size = n).reshape(-1)  
    
    Ytr = generate_response(Xtr, y_noise_scale)
   

    # check coverage
    Xtest = np.random.normal(loc = mean_test, scale = std_test, size = 1).reshape(-1)
   
    Ytest = generate_response(Xtest, y_noise_scale)
    
    # lists of np.array with dim (n,)
    return  Xtr, Ytr, Xtest, Ytest
    
    
    
# compute weighted quantile
    
def wquantile(values, weights, level):
    # values: np.array

    data = np.column_stack((values, weights) )     
    sorted_data = data[data[:, 0].argsort()]
    # Compute cumulative weights
    cumulative_weights = np.cumsum(sorted_data[:, 1])

    # Normalize cumulative weights
    normalized_weights = cumulative_weights / np.sum(sorted_data[:, 1])
    index = np.argmax(normalized_weights >= level)
    
    return sorted_data[index,0]




#######################################
####################################### WCP
#######################################

def w(x, mean_train, std_train, mean_test, std_test):
    return np.exp(-0.5/std_test**2*(x - mean_test)**2 + 0.5/std_train**2*(x - mean_train)**2)
    
    

def WCP(x, f, X, Y, w, mean_train, std_train, mean_test, std_test, alpha):
    # x: scalar
    # w: likelihood function
    scores = np.abs(f(X.reshape(-1,1))[0] - Y.reshape(-1,1))
    scores = np.append(scores.reshape(-1), np.inf)
    
    # compute weights
    wall = np.append(w(X, mean_train, std_train, mean_test, std_test), w(x, mean_train, std_train, mean_test, std_test))
    # compute quantiles
    qhat = wquantile(scores, wall, 1 - alpha)
 
    return qhat


def compute_p(f, generate_data, WCP, n, R):
    stds = np.linspace(0.5, 2.5, num = 20)
    mean_train,  mean_test, std_test, y_noise_scale = 0, 0, 3 , 0.1
    finite_count = np.zeros((R, len(stds)))
    cover = np.zeros((R, len(stds)))

    for i in range(len(stds)):
        std_train = stds[i]
        for r in range(R):
            Xtr, Ytr, Xtest, Ytest = generate_data( n,
                                                   mean_train, std_train,
                                                   mean_test, std_test, y_noise_scale)
            qhat = WCP(Xtest[0], f, Xtr, Ytr, w,  mean_train, std_train,
                       mean_test, std_test, 0.1)

            finite_count[r, i] += ( qhat < np.inf)

            cover[r, i] = np.abs(f(Xtest.reshape(-1,1))[0] - Ytest[0]) < qhat
    
    fp =  np.mean(finite_count, axis = 0)
    cp = np.mean(cover, axis = 0)
    icp = np.sum(finite_count*cover, axis = 0)/np.sum(finite_count, axis = 0)
    
    return  fp, icp, cp



def compute_length(f_list, generate_data, WCP, n, R):
    fg = f_list[0]
    fb = f_list[1]
    
    stds = np.linspace(0.5, 2.5, num = 20)
    mean_train,  mean_test, std_test, y_noise_scale = 0, 0, 3 , 0.1
    
    fg_len = np.zeros((R, len(stds)))
    fb_len = np.zeros((R, len(stds)))
    
    fg_mean_len = np.zeros(len(stds))
    fb_mean_len = np.zeros(len(stds))
    
    for i in range(len(stds)):
        std_train = stds[i]
        for r in range(R):
            Xtr, Ytr, Xtest, Ytest = generate_data( n,
                                                   mean_train, std_train,
                                                   mean_test, std_test, y_noise_scale)
            qhat = WCP(Xtest[0], fg, Xtr, Ytr, w,  mean_train, std_train,
                       mean_test, std_test, 0.1)
            
            fg_len[r,i] = 2*qhat
            
            qhat = WCP(Xtest[0], fb, Xtr, Ytr, w,  mean_train, std_train,
                       mean_test, std_test, 0.1)
            fb_len[r,i] = 2*qhat

            
        fg_mean_len[i] = np.mean( fg_len[~np.isinf(fg_len[:,i]),i])
        fb_mean_len[i] = np.mean( fb_len[~np.isinf(fb_len[:,i]),i])


    return  fg_mean_len, fb_mean_len


##############################################

def compute_pp(f, generate_data, WCP, n, R):
    mus = np.linspace(0, 6, num = 20)
    std_train, mean_test, std_test, y_noise_scale = 3, 0, 3 , 0.1
    finite_count = np.zeros((R, len(mus)))
    cover = np.zeros((R, len(mus)))

    for i in range(len(mus)):
        mean_train = mus[i]
        for r in range(R):
            Xtr, Ytr, Xtest, Ytest = generate_data( n,
                                                   mean_train, std_train,
                                                   mean_test, std_test, y_noise_scale)
            qhat = WCP(Xtest[0], f, Xtr, Ytr, w,  mean_train, std_train,
                       mean_test, std_test, 0.1)

            finite_count[r, i] += ( qhat < np.inf)

            cover[r, i] = np.abs(f(Xtest.reshape(-1,1))[0] - Ytest[0]) < qhat
    
    fp =  np.mean(finite_count, axis = 0)
    cp = np.mean(cover, axis = 0)
    icp = np.sum(finite_count*cover, axis = 0)/np.sum(finite_count, axis = 0)
    
    return  fp, icp, cp




    


