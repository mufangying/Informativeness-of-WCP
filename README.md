# Informativeness of Weighted Conformal Prediction 

## Paper link

[arxiv](http://www.example.com)

## Paper abstract

Weighted conformal prediction (WCP), a recently proposed framework, provides uncertainty quantification for test data points with a different covariate distribution while maintaining a consistent conditional distribution of $Y$ given $X$ with training data. However, the effectiveness of this approach heavily relies on the overlap between covariate distributions; insufficient overlap can lead to uninformative prediction intervals with high probability. To tackle this challenge, we propose two methods for scenarios involving multiple sources with varied covariate distributions but consistent conditional distributions of $Y$ given $X$. 
We establish theoretical guarantees for our proposed methods and demonstrate their efficacy through simulations.

## Codes:

### Demonstration: constructed WCP intervals can be uninformative

We provide jupyter notebooks in folder ./wcp_demon. We demonstrate when the overlap between covariate distributions of training and testing is reduced, the probability of getting uninformative prediction intervals increases.
<p align="center">
  <img src="figures/fix_var_marg_prob_better_f.png" alt="Marginal coverage probability" width="33%" />
  <img src="figures/fix_var_finite_prob_better_f.png" alt="Probability of getting finite prediction intervals" width="33%" />
  <img src="figures/fix_var_infor_prob_better_f.png" alt="Informative coverage probability" width="33%" />
</p>

### Numerical experiments

We provide jupyter notebooks and bash files in folders ./two_groups, ./multi_group_one_dim, and ./multi_group_high_dim respectively.  For the example with higher covariate dimension  and unknown likelihood ratios, 
we use cluster to obtain the results. All results in numerical experiments are obtained through 5000 replications.


