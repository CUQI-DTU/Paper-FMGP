import numpy as np
import numpy.linalg as linalg
import scipy.stats as stats
from scipy.special import erfc
from tqdm import tqdm
import cuqi

# Helper functions
def mat_minor(arr, i, j):
    return arr[np.array(list(range(i))+list(range(i+1, arr.shape[0])))[:,np.newaxis],
               np.array(list(range(j))+list(range(j+1, arr.shape[1])))]

def vec_minor(arr, i):
    return arr[np.array(list(range(i))+list(range(i+1, arr.shape[0])))]

def col(arr, j):
    return arr[:, j]

def row(arr, i):
    return arr[i, :]

def row_minor(arr, i):
    return vec_minor(row(arr, i), i)

def col_minor(arr, i):
    return vec_minor(col(arr, i), i)

def truncated_Gaussian(mean, cov, num_samples):
    """
    Samples from distribution of the form
    pi(x) propto GaussianPDF(x; mean, cov)*non-negative indicator(x)
    through gibbs
    """
        
    n = mean.shape[0]

    def cov_minor(i, j):
        return mat_minor(cov, i, j)

    def cov_col(j):
        return cov[:, j]

    def cov_row(i):
        return cov[:, i]
    
    samples = np.zeros((num_samples, n))
    # current = np.zeros(n)
    current = np.clip(mean, np.zeros(len(mean)), np.ones(len(mean))*np.inf)
    samples[0] = current

    def truncnorm(lower, upper, mu, sigma):
        # print([lower, upper, mu, sigma])
        return stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs()

    # for i in range(1,num_samples):
    pbar = tqdm(range(1, num_samples), "Sample: ")
    for i in pbar:
        for j in range(n):
            # mu = mean[j] - vec_minor(cov_row(j), j)@linalg.solve(cov_minor(j, j), vec_minor(current, j) - vec_minor(mean, j))
            mu = mean[j] + vec_minor(cov_row(j), j)@linalg.solve(cov_minor(j, j), vec_minor(current, j) - vec_minor(mean, j))
            nu = cov[j, j] - vec_minor(cov_col(j), j).T@linalg.solve(cov_minor(j, j), vec_minor(cov_col(j), j))
            # print(mu, nu)
            val = truncnorm(0, np.inf, mu, np.sqrt(nu))
            current[j] = val
        samples[i] = current

    return cuqi.samples.Samples(samples.T)

# Rectified prior
def partially_rectified_Gaussian(S, y, A, B, num_samples, initial_point=None):
    """
    Samples from distribution of the form
    pi(x) propto exp(-0.5 x^TS^{-1}x -0.5(y-A max(0, z))^TB^{-1}(y-A max(0, z)))

    returns samples of x and max(x, 0)
    """
    n = S.shape[0]
    m = A.shape[1]
    I = np.eye(n)
    _Sinv = linalg.inv(S)

    # Pre-computations and helper functions
    def nu_comp(i):
        return S[i, i] - row_minor(S, i)@linalg.solve(mat_minor(S, i, i), col_minor(S, i))
    nu = [nu_comp(i) for i in np.arange(n)]

    def delta_comp(i):
        return 1.0/((1.0/nu[i]) + col(A, i).T@linalg.solve(B, col(A, i)))
    delta = [delta_comp(i) for i in np.arange(n)]

    def kappa(i, z):
        return row_minor(S, i)@linalg.solve(mat_minor(S, i, i), vec_minor(z, i))

    def rho(i, z):
        z_0 = z.copy()
        z_0[i] = 0
        z_0 = np.maximum(0, z_0)
        return delta[i]*(col(A, i).T@linalg.solve(B, y-A@z_0)+kappa(i, current)/nu[i])

    def cdf(v):
        return stats.norm.cdf(v)

    
    def Iplus(mu, sigma):
        return sigma*cdf(mu/sigma)#(1-erf(mu/sigma))
        # v = mu / sigma
        # return sigma * np.exp(stats.norm.logcdf(v))
    
    def Imin(mu, sigma):
        return sigma*(1-cdf(mu/sigma))#(erf(mu/sigma) + 1)
        # v = mu / sigma
        # return sigma * 0.5 * erfc(v / np.sqrt(2))

    def p(i, z):

        #fac = np.exp(-0.5*kn**2+0.5*rd**2)
        """        
        z_0 = z.copy()
        z_0[i] = 0
        z_0_un = z_0.copy()
        z_0 = np.maximum(0, z_0)
        y_bar = y-A@z_0

        facplus = -0.5*y_bar.T@linalg.solve(B, y_bar)
        facplus += -0.5*vec_minor(z, i).T@linalg.solve(mat_minor(S, i, i), vec_minor(z, i))
        facplus += 0.5*((y_bar.T@linalg.solve(B, col(A, i)) + z_0_un.T@linalg.solve(S, row(I, i)))**2)/(_Sinv[i,i] + col(A, i).T@linalg.solve(B, col(A, i)))
        facplus = np.exp(facplus)

        #print(kappa(i, z), rho(i, z))
        facmin = np.exp(-0.5*y_bar@linalg.solve(B,y_bar))"""
        facmin = 1
        facplus = np.exp(-kappa(i, z)**2/(2*nu[i]) + rho(i, z)**2/(2*delta[i]))
        min = facmin*Imin(kappa(i, z), np.sqrt(nu[i]))
        plus = facplus*Iplus(rho(i, z), np.sqrt(delta[i]))

        #print(min, plus, z)
        num = min
        denom = min + plus
        #print(num/denom, cdf(kn)/(cdf(kn) + np.sqrt(delta[i]/nu[i])*fac*(1-cdf(rd))))
        return num/denom

    def r_sample(i, z):
        try:
            return stats.bernoulli(p(i, z)).rvs()
        except Exception as e:
            raise RuntimeError(f"Failed to sample from Bernoulli distribution at i: {i}, z: {z} and p: {p(i, z)} with error: {e}")
    
    # Main sampling
    samples = np.zeros((num_samples, n))
    samples[0,:] = initial_point if initial_point is not None else np.ones(n)
    current = samples[0,:]

    def truncnorm(lower, upper, mu, sigma):
        return stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs()

    # for j in range(1,num_samples):
    pbar = tqdm(range(1, num_samples), "Sample: ")
    for j in pbar:
        # for i in range(n): #[::-1]
        sampling_order = np.arange(n)
        np.random.shuffle(sampling_order)
        for i in sampling_order:
            r = r_sample(i, current)
            if r == 1: # Prior only
                val = truncnorm(-np.inf, 0, kappa(i, current), np.sqrt(nu[i]))
            else:  # Mixed
                val = truncnorm(0, np.inf, rho(i, current), np.sqrt(delta[i]))

            current[i] = val
        samples[j] = current

    return cuqi.samples.Samples(samples.T), cuqi.samples.Samples(np.maximum(0, samples).T)