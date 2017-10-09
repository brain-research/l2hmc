import numpy as np

def effective_n(x):
    """ Returns estimate of the effective sample size of a set of traces.
    Parameters
    ----------
    x : array-like
      An array containing the 2 or more traces of a stochastic parameter. That is, an array of dimension m x n x k, where m is the number of traces, n the number of samples, and k the dimension of the stochastic.
    
    Returns
    -------
    n_eff : float
      Return the effective sample size, :math:`\hat{n}_{eff}`
    Notes
    -----
    The diagnostic is computed by:
      .. math:: \hat{n}_{eff} = \frac{mn}}{1 + 2 \sum_{t=1}^T \hat{\rho}_t}
    where :math:`\hat{\rho}_t` is the estimated autocorrelation at lag t, and T
    is the first odd positive integer for which the sum :math:`\hat{\rho}_{T+1} + \hat{\rho}_{T+1}` 
    is negative.
    References
    ----------
    Gelman et al. (2014)"""
    
    if np.shape(x) < (2,):
        raise ValueError(
            'Calculation of effective sample size requires multiple chains of the same length.')

    try:
        m, n = np.shape(x)
    except ValueError:
        return [effective_n(np.transpose(y)) for y in np.transpose(x)]
        
    s2 = gelman_rubin(x, return_var=True)
    
    negative_autocorr = False
    t = 1
    
    variogram = lambda t: (sum(sum((x[j][i] - x[j][i-t])**2 for i in range(t,n)) for j in range(m)) 
                                / (m*(n - t)))
    rho = np.ones(n)
    # Iterate until the sum of consecutive estimates of autocorrelation is negative
    while not negative_autocorr and (t < n):
        
        rho[t] = 1. - variogram(t)/(2.*s2)
        
        if not t % 2:
            negative_autocorr = sum(rho[t-1:t+1]) < 0
        
        t += 1
     
    return m*n / (1 + 2*rho[1:t].sum())
    

def gelman_rubin(x, return_var=False):
    """ Returns estimate of R for a set of traces.
    The Gelman-Rubin diagnostic tests for lack of convergence by comparing
    the variance between multiple chains to the variance within each chain.
    If convergence has been achieved, the between-chain and within-chain
    variances should be identical. To be most effective in detecting evidence
    for nonconvergence, each chain should have been initialized to starting
    values that are dispersed relative to the target distribution.
    Parameters
    ----------
    x : array-like
      An array containing the 2 or more traces of a stochastic parameter. That is, an array of dimension m x n x k, where m is the number of traces, n the number of samples, and k the dimension of the stochastic.
      
    return_var : bool
      Flag for returning the marginal posterior variance instead of R-hat (defaults of False).
    Returns
    -------
    Rhat : float
      Return the potential scale reduction factor, :math:`\hat{R}`
    Notes
    -----
    The diagnostic is computed by:
      .. math:: \hat{R} = \sqrt{\frac{\hat{V}}{W}}
    where :math:`W` is the within-chain variance and :math:`\hat{V}` is
    the posterior variance estimate for the pooled traces. This is the
    potential scale reduction factor, which converges to unity when each
    of the traces is a sample from the target posterior. Values greater
    than one indicate that one or more chains have not yet converged.
    References
    ----------
    Brooks and Gelman (1998)
    Gelman and Rubin (1992)"""

    if np.shape(x) < (2,):
        raise ValueError(
            'Gelman-Rubin diagnostic requires multiple chains of the same length.')

    try:
        m, n = np.shape(x)
    except ValueError:
        return [gelman_rubin(np.transpose(y)) for y in np.transpose(x)]

    # Calculate between-chain variance
    B_over_n = np.sum((np.mean(x, 1) - np.mean(x)) ** 2) / (m - 1)

    # Calculate within-chain variances
    W = np.sum(
        [(x[i] - xbar) ** 2 for i,
         xbar in enumerate(np.mean(x,
                                   1))]) / (m * (n - 1))

    # (over) estimate of variance
    s2 = W * (n - 1) / n + B_over_n
    
    if return_var:
        return s2

    # Pooled posterior variance estimate
    V = s2 + B_over_n / m

    # Calculate PSRF
    R = V / W

    return np.sqrt(R)