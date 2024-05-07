# imports from installed libraries
import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy


# models
import GPy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# define some functions

def response(X):
    # X: 2-d array
    return 4.0 / (1 + np.exp(-X[:,0])) / (1 + np.exp(-X[:, 1])) 



def generate_response(X, y_scale):
    
    if type(X) is list:
        Y = []
        for i in range(len(X)):
            x = X[i]
            # x: 2-d array
            Y.append(response(x) + y_scale* np.random.normal( size = x.shape[0])  )
        return Y
    else:
        # X is 2-d array
        return response(X) + y_scale*np.random.normal(size = X.shape[0])

    
def generate_equicorrelated_gaussian(d, n, cor):
    
    M = np.random.normal(0,1, n*d).reshape(n,d)

    cov_matrix = np.ones((d,d))*cor + (1-cor)*np.diag(np.ones(d))
    L = np.linalg.cholesky(cov_matrix)

    return M@L.T

    
def generate_data(K, d, nseq, means, stds, cors, y_noise_scale, test_msc):
    

    # traing, calibration data
    Xtr = [means[i].reshape(-1) + stds[i]*generate_equicorrelated_gaussian(d, nseq[i], cors[i] ) for i in range(K) ]

    Xcal = [ means[i].reshape(-1) + stds[i]*generate_equicorrelated_gaussian(d, nseq[i], cors[i] ) for i in range(K) ]
    
    Ycal = [ generate_response(Xcal[i], y_noise_scale)  for i in range(K)]
    
    # test distribution
    mean_test, std_test, cor_test = test_msc[0], test_msc[1], test_msc[2]
    Xtest0 = mean_test.reshape(-1) + std_test*generate_equicorrelated_gaussian(d, int(np.sum(nseq)), cor_test)
   
    Xtest = mean_test.reshape(-1) + std_test*generate_equicorrelated_gaussian(d, 1, cor_test)
    Ytest = generate_response(Xtest, y_noise_scale)
 
    # lists of np.array with dim (n,)
    return  Xtr, Xcal, Ycal,  Xtest0, Xtest, Ytest[0]
    
    
    
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
####################################### WCP with estimated likelihood ratios
#######################################




def WCP(x, f, X, Y, ew, alpha):
    # x: row vector
    # w: likelihood function
    scores = np.abs(f(X)[0] - Y.reshape(-1,1))
    scores = np.append(scores.reshape(-1), np.inf)
    
   
    # p_0
    wx = ew.predict_proba(x)[:, 1].reshape(-1)
    wx = wx/(1-wx + 1e-10)
    
   
    # p_i
    wX = ew.predict_proba(X)[:, 1].reshape(-1)
    wX = wX/(1-wX + 1e-10)
    
    # compute weights
    wall = np.append(wX, wx)
    
    
    # compute quantiles
    qhat = wquantile(scores, wall, 1 - alpha)
 
    return np.array([f(x)[0] - qhat, 
                     f(x)[0] + qhat]).reshape(-1)

    
    
def get_intervals(x, d, f, K, topK, ab_threshold, Xtr, Xcal, Ycal, Xtest0, alpha):
    
    # x: scalar
    # f: trained model to do prediction f()
    # am_threshold is the tuning parameter for adjusted majority vote procedure
    # topK, ab_threshold are the tuning parameters for adjusted Bonferroni's method
    
    
    topK = min([K, topK])
    
    # create a list to store prediction intervals
    PIlist = []
    
    working_group_list = []
    
    # compute the predicted value at vector x
    pred = f(x)[0]
    
    # we will compare 4 methods
    
    ntest0 = Xtest0.shape[0]
    RFs = []
    for i in range(K):
        rf = RandomForestClassifier(n_estimators=100, random_state = 666)
        ntr = Xtr[i].shape[0]
        
        # balanced classification
        idx = np.arange(ntest0)
        np.random.shuffle(idx)
        rfx = np.concatenate((Xtr[i], Xtest0[idx[:ntr],:]))
        rfy = np.concatenate((np.zeros(ntr), np.ones(ntr))).reshape(-1)
        rf.fit(rfx, rfy)
        
        RFs.append(rf)
    
    
    # Method 1 - pick the shortest WCP interval among all the groups
    WCPs = []
    for i in range(K):
        Xcali = Xcal[i]
        Ycali = Ycal[i]
        ew = RFs[i]
        WCPs.append( WCP(x, f, Xcali, Ycali, ew, alpha))
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
        ew = RFs[k]
        w0 = ew.predict_proba(Xtest0)[:, 1].reshape(-1)
        w0 = w0/(1-w0 + 1e-10)
        
        wtr_sum = ew.predict_proba(Xtr[k])[:, 1].reshape(-1)
        wtr_sum = wtr_sum/(1-wtr_sum + 1e-10)
        wtr_sum = np.sum(wtr_sum)
       
        store_matrix[k,:] = w0/(wtr_sum + w0)   
    
  
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
        # use 1 group
        bon_level = alpha
        check_matrix = store_matrix < bon_level
        PIlist.append(WCPs[   np.argmax(np.sum(check_matrix, axis = 1)) , :])
        working_group_list.append(1)

    else:
        
        bon_level = alpha*1.0/len(selected_group)
                               
        abpis = []
        for group in selected_group:
            ew = RFs[group]
            Xcali = Xcal[group]
            Ycali = Ycal[group]
            abpis.append( WCP(x, f, Xcali, Ycali, ew, bon_level))

        # stack by row
        abpis = np.vstack(abpis)
        
        min_q = np.min(abpis[:,1] - abpis[:,0])
        PIlist.append(np.array([-0.5, 0.5])*min_q + pred)
        working_group_list.append(len(selected_group))
        
    
    
    
    # Method 6 Pooling groups
    Xcal_all = np.concatenate(Xcal)
    Ycal_all = np.concatenate(Ycal)
    
    Xtr_all = np.concatenate(Xtr)
    
    RF_pool = RandomForestClassifier(n_estimators=100, random_state = 666)
    ntr_all = sum([Xtr[i].shape[0] for i in range(K) ]  )
        
    rfx = np.concatenate((Xtr_all, Xtest0))
    rfy = np.concatenate((np.zeros(ntr_all), np.ones(ntest0))).reshape(-1)
        
    RF_pool.fit(rfx, rfy)
   
    
    PIlist.append(  WCP(x, f, Xcal_all, Ycal_all, RF_pool, alpha))
    working_group_list.append( K )
    
    
    return np.vstack(PIlist), np.array(working_group_list)



def replications(R, d, f, K_int, topK,  ab_threshold,
                       n_int, mean_int, std_int, cor_int, y_noise_scale, test_msc, alpha):
    
    num_method = 4
    # consider topK list being 
    cov_mat = np.zeros((R, num_method))
    len_mat = np.zeros((R, num_method))
    fin_mat = np.zeros((R, num_method))
    working_groups_mat = np.zeros((R, num_method))
    
    for i in range(R):
        K = np.random.randint(K_int[0], K_int[1])
        means = [ np.random.uniform(mean_int[0], mean_int[1], d) for i in range(K)]
        stds = np.random.uniform(std_int[0], std_int[1], K)
        cors = np.random.uniform(cor_int[0], cor_int[1], K)
        nseq = np.random.randint(n_int[0], n_int[1], size= K)
        
        
        Xtr, Xcal, Ycal,  Xtest0, x, y  = generate_data(K, d, nseq, 
                                                        means, stds, cors,
                                                        y_noise_scale, test_msc)
        
        
       
        PIs, working_groups = get_intervals(x,d, f, K, topK,  ab_threshold, 
                                                 Xtr, Xcal, Ycal, Xtest0, alpha)
        
        fin_mat[i,:] = ~np.isinf(PIs[:,1] - PIs[:,0])
        cov_mat[i,:] = (PIs[:,0] <= y)*(PIs[:,1] >= y)
        len_mat[i,:] = PIs[:,1] - PIs[:,0]
        working_groups_mat[i,:] = working_groups
        
        
        
    return cov_mat, len_mat, fin_mat, working_groups_mat


