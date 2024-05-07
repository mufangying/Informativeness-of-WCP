from high_dim_utils import *

d = 20
    
y_noise_scale = 0.1
test_msc = [np.zeros(d), 1, 0]

print('Dimension:')
print(d)
print('')

# a pretrained model
np.random.seed(324)
sample_size = d*100
Xtr = np.random.uniform(-3, 3, d*sample_size).reshape(sample_size, d)
Ytr = generate_response(Xtr, 0.1)

kernel = GPy.kern.RBF(input_dim=d)
model = GPy.models.GPRegression(Xtr, Ytr.reshape(-1,1), kernel)

# optimize the model parameters
model.optimize()
f = model.predict

K_int = [2, 11]
topK = 3
ab_threshold = 0.01


n_int = [100, 501]
mean_int = [-1,1]

std_int = [0.8, 1]
alpha = 0.1

R = 5000
num_method = 4




print('Low cor:')
print('')

cor_int = [0, 0.2]
covs, lens, counts, working_groups_mat = replications(R, d, f, K_int, topK,  ab_threshold,
                       n_int, mean_int, std_int, cor_int, y_noise_scale, test_msc, 0.1 )

# WCP then merged
print('Marginal coverage:')
print(np.round(np.mean(covs, axis = 0), decimals = 3))
print('')

print('Probabiilty of getting finite prediction intervals')
print(np.round(np.mean(counts, axis = 0), decimals = 3))
print('')

print('Informative coverage probability:')
print( np.round( np.sum(counts*covs, axis = 0) /np.sum(counts, axis = 0), decimals = 3)  )
print('')

avg_finite_len = []
for k in range(num_method):
    lenk = lens[:,k]
    lenk = lenk[~np.isinf(lenk)]
    if len(lenk) == 0:
        avg_finite_len.append(  np.inf )
    else:
        avg_finite_len.append( np.round( np.mean(lenk), decimals=3) )
print('Lengths of informative intervals:')
print(avg_finite_len )
print('')


print('Working groups number:')
print(np.round( np.quantile(working_groups_mat,0.5, axis = 0) ,decimals = 3))
print('')


print('High cor:')
print('')

cor_int = [0.7, 0.9]
covs, lens, counts, working_groups_mat = replications(R, d, f, K_int, topK,  ab_threshold,
                       n_int, mean_int, std_int, cor_int, y_noise_scale, test_msc,0.1 )

# WCP then merged
print('Marginal coverage:')
print(np.round(np.mean(covs, axis = 0), decimals = 3))
print('')

print('Probabiilty of getting finite prediction intervals')
print(np.round(np.mean(counts, axis = 0), decimals = 3))
print('')

print('Informative coverage probability:')
print( np.round( np.sum(counts*covs, axis = 0) /np.sum(counts, axis = 0), decimals = 3)  )
print('')

avg_finite_len = []
for k in range(num_method):
    lenk = lens[:,k]
    lenk = lenk[~np.isinf(lenk)]
    if len(lenk) == 0:
        avg_finite_len.append(  np.inf )
    else:
        avg_finite_len.append( np.round( np.mean(lenk), decimals=3) )
print('Lengths of informative intervals:')
print(avg_finite_len )
print('')


print('Working groups number:')
print(np.round( np.quantile(working_groups_mat,0.5, axis = 0) ,decimals = 3))