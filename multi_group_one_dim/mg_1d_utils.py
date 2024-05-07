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

    

    
def generate_data(K, nseq, means, stds, y_noise_scale, test_mv):
    
    # generate unlabled training data
    Xtr = [np.random.normal(loc = means[i], scale = stds[i], 
                          size = nseq[i]).reshape(-1)  for i in range(K)]
   
    # generate labeled calibration data
    Xcal = [np.random.normal(loc = means[i], scale = stds[i], 
                          size = nseq[i]).reshape(-1)  for i in range(K)]
    
    Ycal = generate_response(Xcal, y_noise_scale)
   

    # check coverage
    Xtest0 = np.random.normal(loc = test_mv[0], scale = test_mv[1], size = max(nseq)).reshape(-1)
    Xtest = np.random.normal(loc = test_mv[0], scale = test_mv[1], size = 1).reshape(-1)
    Ytest = generate_response(Xtest, y_noise_scale)
    
    # lists of np.array with dim (n,)
    return  Xtr, Xcal, Ycal,  Xtest0, Xtest[0], Ytest[0]
    
    
    
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
####################################### IWCP
#######################################


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
    

def WCP(x, f, X, Y, w, mean_train, std_train, mean_test, std_test,alpha):
    # x: scalar
    # w: likelihood function
    scores = np.abs(f(X.reshape(-1,1))[0] - Y.reshape(-1,1))
    scores = np.append(scores.reshape(-1), np.inf)
    
    # compute weights
    wall = np.append(w(X,  mean_train, std_train, mean_test, std_test), w(x,  mean_train, std_train, mean_test, std_test))
    # compute quantiles
    qhat = wquantile(scores, wall, 1 - alpha)
 
    return np.array([f(np.array([x]).reshape(-1,1))[0] - qhat, 
                     f(np.array([x]).reshape(-1,1))[0] + qhat]).reshape(-1)

    
    
def get_intervals(x, f, K, topK, ab_threshold, Xtr, Xcal, Ycal, Xtest0, w, w_pool, nseq, means, stds,  test_mv, alpha):
    
    # x: scalar
    # f: trained model to do prediction f()
    # am_threshold is the tuning parameter for adjusted majority vote procedure
    # topK, ab_threshold are the tuning parameters for adjusted Bonferroni's method
    
    # w: likelihood ratio function
    
    mean_test, std_test = test_mv[0], test_mv[1]
    
    # create a list to store prediction intervals
    PIlist = []
    
    working_group_list = []
    
    # compute the predicted value at x
    pred = f(np.array([x]).reshape(-1,1))[0]
    
    # we will compare 5 methods
    
    
    
    
    # Method 1 - pick the shortest WCP interval among all the groups
    WCPs = []
    for i in range(K):
        Xcali = Xcal[i]
        Ycali = Ycal[i]
        mean = means[i]
        std = stds[i]
        WCPs.append( WCP(x, f, Xcali, Ycali, w, mean, std, mean_test, std_test, alpha))
    # pick the shortest one
    WCPs = np.vstack(WCPs)
    
    # compute the lengths of WCP intervals
    wcpl = WCPs[:,1] - WCPs[:,0]
    
    PIlist.append(WCPs[np.argmin(wcpl),:])
    
    working_group_list.append(K)
    
    
   

    # Proposed method:
    ntest0 = Xtest0.shape[0]
    store_matrix = np.zeros((K, ntest0))
    
    # compute ratio w(X0)/(w(X0) + sum w(Xki))
    for k in range(K):
        wsum_tr = np.sum(w(Xtr[k], means[k], stds[k], mean_test, std_test))
        w_te = w(Xtest0, means[k], stds[k], mean_test, std_test)
        store_matrix[k,:] = w_te/(wsum_tr + w_te)   
    
  
    # Kada = 1
    bon_level = alpha
    check_matrix = store_matrix < bon_level
    PIlist.append(WCPs[   np.argmax(np.sum(check_matrix, axis = 1)) , :])
    working_group_list.append(1)

    
    # Kada = topK
    
    bon_level = alpha*1.0/topK
    check_matrix = store_matrix < bon_level

    # select groups
    selected_group = []
    for j in range(topK):

        # row sum use axis = 1
        max_index = np.argmax(np.sum(check_matrix, axis = 1))
        if np.sum(check_matrix[max_index,:]) <= ntest0 * ab_threshold:
            break
        selected_group.append(max_index)

        # delete covered points use axis = 1
        arr = check_matrix[max_index,:]
        check_matrix = check_matrix[:, ~arr]

        # if all points have been covered
        if check_matrix.shape[1] == 0:
            break
        

    if len(selected_group) == 0:
        # use 1
        bon_level = alpha
        check_matrix = store_matrix < bon_level
        PIlist.append(WCPs[   np.argmax(np.sum(check_matrix, axis = 1)) , :])
        working_group_list.append(1)

    else:
        
        bon_level = alpha*1.0/len(selected_group)
                               
        abpis = []
        for group in selected_group:
            Xcali = Xcal[group]
            Ycali = Ycal[group]
            mean = means[group]
            std = stds[group]
            abpis.append( WCP(x, f, Xcali, Ycali, w, mean, std, mean_test, std_test, bon_level))

        # stack by row
        abpis = np.vstack(abpis)
        
        min_q = np.min(abpis[:,1] - abpis[:,0])
        PIlist.append(np.array([-0.5, 0.5])*min_q + pred)
        working_group_list.append(len(selected_group))
        
    
    
    
    # Method 6 Pooling groups
    Xcal_all = np.concatenate(Xcal)
    Ycal_all = np.concatenate(Ycal)
    
    PIlist.append(  WCP_pool(x, f, Xcal_all, Ycal_all, w_pool, nseq, means, stds, mean_test, std_test, alpha) )
    working_group_list.append( K )
    
    
    return np.vstack(PIlist), np.array(working_group_list)


def replications(R, f, K_int, topK,  ab_threshold,
                       n_int, mean_int, std_int, y_noise_scale, test_mv, w, w_pool, alpha):
    
    num_method = 4
    # consider topK list being 
    cov_mat = np.zeros((R, num_method))
    len_mat = np.zeros((R, num_method))
    fin_mat = np.zeros((R, num_method))
    working_groups_mat = np.zeros((R, num_method))
    
    for i in range(R):
        K = np.random.randint(K_int[0], K_int[1])
        means = np.random.uniform(mean_int[0], mean_int[1], K)
        stds = np.random.uniform(std_int[0], std_int[1], K)
        nseq = np.random.randint(n_int[0], n_int[1], size=K)
        
        
        Xtr, Xcal, Ycal,  Xtest0, x, y  = generate_data(K, nseq, 
                                                        means, stds,
                                                        y_noise_scale, test_mv)
        
        
       
        PIs, working_groups = get_intervals(x, f, K, topK,  ab_threshold, 
                                                 Xtr, Xcal, Ycal, Xtest0, w, w_pool,
                                            nseq, means, stds, test_mv, alpha)
        
        fin_mat[i,:] = ~np.isinf(PIs[:,1] - PIs[:,0])
        cov_mat[i,:] = (PIs[:,0] <= y)*(PIs[:,1] >= y)
        len_mat[i,:] = PIs[:,1] - PIs[:,0]
        working_groups_mat[i,:] = working_groups
        
        
    return cov_mat, len_mat, fin_mat, working_groups_mat


