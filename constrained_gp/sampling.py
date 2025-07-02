import cuqi
import torch
import cuqipy_pytorch
import numpy as np
from .gibbs import truncated_Gaussian, partially_rectified_Gaussian
from .utils import relu, drelu, ReparameterizedGaussian

def rto_sample(K01ts, K11ss, K10st, K00tt, fun_noise_matrix, grad_noise_matrix, f0t_prior_mean, f0t, no_of_warmups, no_of_samples):
    def linearforward(v):
        return K01ts @ np.linalg.solve(K11ss + grad_noise_matrix, v) + f0t_prior_mean.flatten()

    def linearadjoint(v):
        return np.linalg.solve(K11ss + grad_noise_matrix, K10st @ v)

    s_dim = K11ss.shape[0]
    t_dim = K00tt.shape[0]

    A = cuqi.model.LinearModel(forward=linearforward, adjoint=linearadjoint, domain_geometry=s_dim, range_geometry=t_dim)

    cov_matrix = K00tt - K01ts @ (np.linalg.solve(K11ss + grad_noise_matrix, K10st)) + fun_noise_matrix

    y_obs = f0t.flatten()

    x = cuqi.implicitprior.RegularizedGaussian(np.zeros(s_dim), cov=K11ss + grad_noise_matrix, constraint = "nonnegativity")

    L = np.linalg.cholesky(cov_matrix)
    L_inv = np.linalg.inv(L)

    y = cuqi.distribution.Gaussian(A(x), sqrtprec=L_inv)
    joint = cuqi.distribution.JointDistribution(x, y)

    post = joint(y=y_obs)

    sampler = cuqi.experimental.mcmc.RegularizedLinearRTO(post, maxit=100, solver="ScipyMinimizer") # , solver="ScipyLinearLSQ"

    sampler.warmup(no_of_warmups)
    sampler.sample(no_of_samples)

    return sampler.get_samples().burnthin(no_of_warmups)

def gibbs_truncated_sample(K01ts, K11ss, K10st, K00tt, fun_noise_matrix, grad_noise_matrix, f0t_prior_mean, f1s_prior_mean, f0t, no_of_warmups, no_of_samples):
    f1s_post_mean = f1s_prior_mean + (K10st @ np.linalg.solve(K00tt + fun_noise_matrix, f0t - f0t_prior_mean)).flatten()

    f1s_post_cov = K11ss - K10st @ np.linalg.solve(K00tt + fun_noise_matrix, K01ts) + grad_noise_matrix

    # f1s_post_std = np.sqrt(np.diag(f1s_post_cov))

    return truncated_Gaussian(f1s_post_mean, f1s_post_cov, no_of_warmups+no_of_samples).burnthin(no_of_warmups)

def gibbs_nonlinear_sample(K01ts, K11ss, K10st, K00tt, fun_noise_matrix, grad_noise_matrix, f0t_prior_mean, f1s_prior_mean, f0t, no_of_warmups, no_of_samples):
    m = K00tt.shape[0] # length of t, 50
    n = K11ss.shape[0] # length of s, 20
    s_dim = K11ss.shape[0]
    # f0t = f0t.flatten()
    S = K11ss + grad_noise_matrix
    # AA1 = K01ts@np.linalg.inv(S)
    A = K01ts @ np.linalg.solve(S, np.eye(S.shape[0])) # seems equivalent to jnp.linalg.solve(K11ss + grad_noise_matrix, K10st).T
    # B = k00tt - k01ts@(np.linalg.inv(k11ss+np.eye(n)*np.exp(2.0*virtual_noise)))@k10st + np.eye(m)*np.exp(2.0*data_noise)
    B = K00tt - K01ts@(np.linalg.solve(S, K10st)) + fun_noise_matrix
    # The above two lines are equivalent, and the second one might be faster
    samples, transformed_samples = partially_rectified_Gaussian(S, f0t.flatten()-f0t_prior_mean.flatten(), A, B, no_of_samples+no_of_warmups, np.ones(s_dim)*0.1)

    return samples.burnthin(no_of_warmups), transformed_samples.burnthin(no_of_warmups)

def nuts_nonlinear_sample(K01ts, K11ss, K10st, K00tt, fun_noise_matrix, grad_noise_matrix, f0t_prior_mean, f0t, no_of_warmups, no_of_samples):
    def linearforward(v):
        return K01ts @ np.linalg.solve(K11ss + grad_noise_matrix, v) + f0t_prior_mean.flatten()

    def linearadjoint(v):
        return np.linalg.solve(K11ss + grad_noise_matrix, K10st @ v)

    def nonlinearforward(v):
        return linearforward(relu(v))

    def nonlinearadjoint(v, x):
        return drelu(x) * linearadjoint(v)

    s_dim = K11ss.shape[0]
    t_dim = K00tt.shape[0]

    A_nonlinear = cuqi.model.Model(forward=nonlinearforward, gradient=nonlinearadjoint, domain_geometry=s_dim, range_geometry=t_dim)

    cov_matrix = K00tt - K01ts @ (np.linalg.solve(K11ss + grad_noise_matrix, K10st)) + fun_noise_matrix

    y_obs = f0t.flatten()

    x = cuqi.distribution.Gaussian(np.zeros(s_dim), cov=K11ss + grad_noise_matrix)

    y = cuqi.distribution.Gaussian(A_nonlinear(x), cov=cov_matrix)
    joint = cuqi.distribution.JointDistribution(x, y)

    post = joint(y=y_obs)

    solver = cuqi.solver.ScipyMaximizer(post.logd, x0 = np.zeros(x.dim), gradfunc=post.gradient)
    x_MAP, info = solver.solve()

    sampler = cuqi.experimental.mcmc.NUTS(post, max_depth=7, initial_point=x_MAP) #, initial_point=x_MAP

    # sampler_nonlinear_nuts.warmup(no_of_warmups+no_of_samples)
    # # sampler_nonlinear_nuts.sample(no_of_samples)
    sampler.warmup(no_of_warmups)
    sampler.sample(no_of_samples)

    samples = sampler.get_samples()
    transformed_samples = cuqi.samples.Samples(np.maximum(samples.samples, 0))

    return samples.burnthin(no_of_warmups), transformed_samples.burnthin(no_of_warmups)

def nuts_truncated_sample(K01ts, K11ss, K10st, K00tt, fun_noise_matrix, grad_noise_matrix, f0t_prior_mean, f0t, no_of_warmups, no_of_samples):
    def linearforward(v):
        return K01ts @ np.linalg.solve(K11ss + grad_noise_matrix, v) + f0t_prior_mean.flatten()

    def linearadjoint(v):
        return np.linalg.solve(K11ss + grad_noise_matrix, K10st @ v)

    def expforward(v):
        return linearforward(np.exp(v))

    def expadjoint(v, x):
        return np.exp(x) * linearadjoint(v)

    s_dim = K11ss.shape[0]
    t_dim = K00tt.shape[0]

    A_nonlinear = cuqi.model.Model(forward=expforward, gradient=expadjoint, domain_geometry=s_dim, range_geometry=t_dim)

    cov_matrix = K00tt - K01ts @ (np.linalg.solve(K11ss + grad_noise_matrix, K10st)) + fun_noise_matrix

    y_obs = f0t.flatten()

    # x = cuqi.distribution.Gaussian(np.zeros(s_dim), cov=K11ss + grad_noise_matrix)
    x = ReparameterizedGaussian(np.zeros(s_dim), cov=K11ss + grad_noise_matrix)

    y = cuqi.distribution.Gaussian(A_nonlinear(x), cov=cov_matrix)
    joint = cuqi.distribution.JointDistribution(x, y)

    post = joint(y=y_obs)

    solver = cuqi.solver.ScipyMaximizer(post.logd, x0 = np.zeros(x.dim), gradfunc=post.gradient)
    x_MAP, info = solver.solve()

    sampler = cuqi.experimental.mcmc.NUTS(post, max_depth=7, initial_point=x_MAP) #, initial_point=x_MAP

    # sampler_nonlinear_nuts.warmup(no_of_warmups+no_of_samples)
    # # sampler_nonlinear_nuts.sample(no_of_samples)
    sampler.warmup(no_of_warmups)
    sampler.sample(no_of_samples)

    samples = sampler.get_samples()
    transformed_samples = cuqi.samples.Samples(np.exp(samples.samples))

    return samples.burnthin(no_of_warmups), transformed_samples.burnthin(no_of_warmups)

def nuts_nonlinear_sample_torch(K01ts, K11ss, K10st, K00tt, fun_noise_matrix, grad_noise_matrix, f0t_prior_mean, f0t, no_of_warmups, no_of_samples):
    K01ts = torch.as_tensor(K01ts.copy(), dtype=torch.float64)
    K11ss = torch.as_tensor(K11ss.copy(), dtype=torch.float64)
    K10st = torch.as_tensor(K10st.copy(), dtype=torch.float64)
    K00tt = torch.as_tensor(K00tt.copy(), dtype=torch.float64)
    fun_noise_matrix = torch.as_tensor(fun_noise_matrix.copy(), dtype=torch.float64)
    grad_noise_matrix = torch.as_tensor(grad_noise_matrix.copy(), dtype=torch.float64)
    f0t_prior_mean = torch.as_tensor(f0t_prior_mean.copy(), dtype=torch.float64)
    f0t = torch.as_tensor(f0t.copy(), dtype=torch.float64)
    def linearforward(v):
        v = v.to(K01ts.dtype)
        return K01ts @ torch.linalg.solve(K11ss + grad_noise_matrix, v) + f0t_prior_mean.flatten()

    def linearadjoint(v):
        v = v.to(K10st.dtype)
        return torch.linalg.solve(K11ss + grad_noise_matrix, K10st @ v)

    def relu_torch(x):
        return torch.maximum(x, torch.tensor(0.0, dtype=x.dtype))

    def drelu_torch(x):
        return torch.where(x > 0, torch.tensor(1.0, dtype=x.dtype), torch.tensor(0.0, dtype=x.dtype))

    def nonlinearforward(v):
        return linearforward(relu_torch(v))

    def nonlinearadjoint(v, x):
        return drelu_torch(x) * linearadjoint(v)

    s_dim = K11ss.shape[0]
    t_dim = K00tt.shape[0]

    A_nonlinear = cuqi.model.Model(forward=nonlinearforward, gradient=nonlinearadjoint, domain_geometry=s_dim, range_geometry=t_dim)

    cov_matrix = torch.as_tensor(K00tt - K01ts @ (np.linalg.solve(K11ss + grad_noise_matrix, K10st)) + fun_noise_matrix)

    y_obs = f0t.flatten()
    y_obs = torch.as_tensor(y_obs, dtype=torch.float64)

    x = cuqipy_pytorch.distribution.Gaussian(torch.zeros(s_dim), cov=torch.as_tensor(K11ss + grad_noise_matrix))

    y = cuqipy_pytorch.distribution.Gaussian(A_nonlinear(x), cov=cov_matrix)
    joint = cuqi.distribution.JointDistribution(x, y)

    post = joint(y=y_obs)

    # solver = cuqi.solver.ScipyMaximizer(post.logd, x0 = np.zeros(x.dim), gradfunc=post.gradient)
    # x_MAP, info = solver.solve()

    # sampler = cuqi.experimental.mcmc.NUTS(post, max_depth=7, initial_point=x_MAP) #, initial_point=x_MAP

    sampler = cuqipy_pytorch.sampler.NUTS(post)
    samples = sampler.sample(no_of_samples, no_of_warmups)['x']
    transformed_samples = cuqi.samples.Samples(np.maximum(samples.samples, 0))

    # sampler_nonlinear_nuts.warmup(no_of_warmups+no_of_samples)
    # # sampler_nonlinear_nuts.sample(no_of_samples)
    # sampler.warmup(no_of_warmups)
    # sampler.sample(no_of_samples)

    # samples = sampler.get_samples()
    # transformed_samples = cuqi.samples.Samples(np.maximum(samples.samples, 0))

    return samples, transformed_samples

class ReparameterizedGaussianTorch(cuqipy_pytorch.distribution.Gaussian):
    def __init__(self, mean=None, cov=None, **kwargs):
        super().__init__(mean=mean, cov=cov, **kwargs)
    def logpdf(self, x):
        sigma = torch.exp(x)
        return super().logpdf(sigma) + torch.sum(x)
    def gradient(self, x):
        sigma = torch.exp(x)
        return super().gradient(sigma) * sigma + 1.0
    def _sample(self,N=1,rng=None):
        return None

def nuts_truncated_sample_torch(K01ts, K11ss, K10st, K00tt, fun_noise_matrix, grad_noise_matrix, f0t_prior_mean, f0t, no_of_warmups, no_of_samples):
    K01ts = torch.as_tensor(K01ts.copy(), dtype=torch.float64)
    K11ss = torch.as_tensor(K11ss.copy(), dtype=torch.float64)
    K10st = torch.as_tensor(K10st.copy(), dtype=torch.float64)
    K00tt = torch.as_tensor(K00tt.copy(), dtype=torch.float64)
    fun_noise_matrix = torch.as_tensor(fun_noise_matrix.copy(), dtype=torch.float64)
    grad_noise_matrix = torch.as_tensor(grad_noise_matrix.copy(), dtype=torch.float64)
    f0t_prior_mean = torch.as_tensor(f0t_prior_mean.copy(), dtype=torch.float64)
    f0t = torch.as_tensor(f0t.copy(), dtype=torch.float64)
    def linearforward(v):
        v = v.to(K01ts.dtype)
        return K01ts @ torch.linalg.solve(K11ss + grad_noise_matrix, v) + f0t_prior_mean.flatten()

    def linearadjoint(v):
        v = v.to(K10st.dtype)
        return torch.linalg.solve(K11ss + grad_noise_matrix, K10st @ v)

    def relu_torch(x):
        return torch.exp(x)

    def drelu_torch(x):
        return torch.exp(x)

    def nonlinearforward(v):
        return linearforward(relu_torch(v))

    def nonlinearadjoint(v, x):
        return drelu_torch(x) * linearadjoint(v)

    s_dim = K11ss.shape[0]
    t_dim = K00tt.shape[0]

    A_nonlinear = cuqi.model.Model(forward=nonlinearforward, gradient=nonlinearadjoint, domain_geometry=s_dim, range_geometry=t_dim)

    cov_matrix = torch.as_tensor(K00tt - K01ts @ (np.linalg.solve(K11ss + grad_noise_matrix, K10st)) + fun_noise_matrix)

    y_obs = f0t.flatten()
    y_obs = torch.as_tensor(y_obs, dtype=torch.float64)

    x = ReparameterizedGaussianTorch(torch.zeros(s_dim), cov=torch.as_tensor(K11ss + grad_noise_matrix))

    y = cuqipy_pytorch.distribution.Gaussian(A_nonlinear(x), cov=cov_matrix)
    joint = cuqi.distribution.JointDistribution(x, y)

    post = joint(y=y_obs)

    # solver = cuqi.solver.ScipyMaximizer(post.logd, x0 = np.zeros(x.dim), gradfunc=post.gradient)
    # x_MAP, info = solver.solve()

    # sampler = cuqi.experimental.mcmc.NUTS(post, max_depth=7, initial_point=x_MAP) #, initial_point=x_MAP

    sampler = cuqipy_pytorch.sampler.NUTS(post)
    samples = sampler.sample(no_of_samples, no_of_warmups)['x']
    transformed_samples = cuqi.samples.Samples(np.exp(samples.samples))

    # sampler_nonlinear_nuts.warmup(no_of_warmups+no_of_samples)
    # # sampler_nonlinear_nuts.sample(no_of_samples)
    # sampler.warmup(no_of_warmups)
    # sampler.sample(no_of_samples)

    # samples = sampler.get_samples()
    # transformed_samples = cuqi.samples.Samples(np.maximum(samples.samples, 0))

    return samples, transformed_samples