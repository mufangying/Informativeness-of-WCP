#######################################
####################################### IWCP
#######################################


def check_cover(M, testy,  tau = 0.5):
    # check if testy is cover by any of intervals in M (mx2)
    
    ifcover = (M[:,0] <= testy) * (M[:,1] >= testy) 
    
    return np.sum(ifcover) > 0,  np.sum(M[:,1] - M[:,0])



# non-equal weights
def merge_interval(M, weights,  tau = 0.5):
    # M two columns
    breaks = np.unique(M)
    if len(breaks) == 2 and np.isinf(breaks).any():
        # all are np.inf
        return np.array([[-np.inf, np.inf]])
    else:
        breaks_center = [(breaks[i]+breaks[i+1])/2 for i in range(len(breaks) - 1)]
        intervals = []
        for i in range(len(breaks) - 1):
            x = breaks_center[i]
            count = (x>= M[:,0]) * (x<= M[:,1]) 
            count = np.sum(count*weights) > tau
            if count:
                intervals.append([breaks[i], breaks[i+1]])
                
        merged = []
        for interval in intervals:
            if not merged or interval[0] - 1e-9 > merged[-1][1]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        
        return np.array(merged)

    
