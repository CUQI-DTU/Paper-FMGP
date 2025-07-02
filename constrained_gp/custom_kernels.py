import jax.numpy as jnp
from jaxtyping import Float
import tensorflow_probability.substrates.jax as tfp

from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.kernels.stationary.utils import squared_distance
from gpjax.typing import (
    Array,
    ScalarFloat,
)

class RBF(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel.

    Computes the covariance for pair of inputs $(x, y)$ with lengthscale parameter
    $\ell$ and variance $\sigma^2$:
    $$
    k(x,y)=\sigma^2\exp\Bigg(- \frac{\lVert x - y \rVert^2_2}{2 \ell^2} \Bigg)
    $$
    """

    name: str = "RBF"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        K = self.variance.value * jnp.exp(-0.5 * squared_distance(x, y))
        return K.squeeze()
    
    def dxy(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> Float[Array, " D"]:
        """Compute the gradient of the kernel with respect to the first element of x.
        
        Args:
            x: First input of shape (D,)
            y: Second input of shape (D,)
            
        Returns:
            Gradient of k(x,y) with respect to x of shape (D,)
        """
        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        
        diff = x - y
        K = self.variance.value * jnp.exp(-0.5 * jnp.sum(diff ** 2))
        
        # Gradient of kernel with respect to x
        grad_K = -K * diff / (self.lengthscale.value ** 2)
        
        return grad_K

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFDX1Y(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel.

    Computes the covariance between derivatives df/dx[0] and f for inputs $(x, y)$ 
    with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFDX1Y"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        
        diff = x - y
        squared_dist = jnp.sum(diff ** 2)
        K = self.variance.value * jnp.exp(-0.5 * squared_dist)
        
        # Derivative with respect to first dimension of x
        # For RBF kernel, cov(df/dx[0], f) = -K * (x[0]-y[0])/lengthscale²
        dK_dx = -K * diff[0] / self.lengthscale.value
        
        return dK_dx.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFDX2Y(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel.

    Computes the covariance between derivatives df/dx[1] and f for inputs $(x, y)$ 
    with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFDX2Y"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        
        diff = x - y
        squared_dist = jnp.sum(diff ** 2)
        K = self.variance.value * jnp.exp(-0.5 * squared_dist)
        
        # Derivative with respect to first dimension of x
        # For RBF kernel, cov(df/dx[0], f) = -K * (x[0]-y[0])/lengthscale²
        dK_dx = -K * diff[1] / self.lengthscale.value
        
        return dK_dx.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFDX3Y(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel.

    Computes the covariance between derivatives df/dx[2] and f for inputs $(x, y)$ 
    with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFDX3Y"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        
        diff = x - y
        squared_dist = jnp.sum(diff ** 2)
        K = self.variance.value * jnp.exp(-0.5 * squared_dist)
        
        # Derivative with respect to first dimension of x
        # For RBF kernel, cov(df/dx[0], f) = -K * (x[0]-y[0])/lengthscale²
        dK_dx = -K * diff[2] / self.lengthscale.value
        
        return dK_dx.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)
class RBFXDY1(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel.

    Computes the covariance between derivatives f and df/dy[0] for inputs $(x, y)$ 
    with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFDX1Y"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        
        diff = x - y
        squared_dist = jnp.sum(diff ** 2)
        K = self.variance.value * jnp.exp(-0.5 * squared_dist)
        
        # Derivative with respect to first dimension of y
        # For RBF kernel, cov(f, df/dy[0]) = K * (x[0]-y[0])/lengthscale²
        # Note: The sign is positive because derivative is wrt y, not x
        dK_dy = K * diff[0] / self.lengthscale.value
        
        return dK_dy.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFXDY2(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel.

    Computes the covariance between derivatives f and df/dy[1] for inputs $(x, y)$ 
    with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFDX2Y"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        
        diff = x - y
        squared_dist = jnp.sum(diff ** 2)
        K = self.variance.value * jnp.exp(-0.5 * squared_dist)
        
        # Derivative with respect to first dimension of y
        # For RBF kernel, cov(f, df/dy[0]) = K * (x[0]-y[0])/lengthscale²
        # Note: The sign is positive because derivative is wrt y, not x
        dK_dy = K * diff[1] / self.lengthscale.value
        
        return dK_dy.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFXDY3(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel.

    Computes the covariance between derivatives f and df/dy[2] for inputs $(x, y)$ 
    with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFXDY3"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        
        diff = x - y
        squared_dist = jnp.sum(diff ** 2)
        K = self.variance.value * jnp.exp(-0.5 * squared_dist)
        
        # Derivative with respect to first dimension of y
        # For RBF kernel, cov(f, df/dy[0]) = K * (x[0]-y[0])/lengthscale²
        # Note: The sign is positive because derivative is wrt y, not x
        dK_dy = K * diff[2] / self.lengthscale.value
        
        return dK_dy.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFDX1DY1(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel for derivatives.

    Computes the covariance between derivatives df/dx[0] and df/dy[0] for inputs $(x, y)$ 
    with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFDX1DY1"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:

        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        
        diff = x - y
        squared_dist = jnp.sum(diff ** 2)
        K = self.variance.value * jnp.exp(-0.5 * squared_dist)
        
        # For RBF kernel, cov(df/dx[0], df/dy[0]) = K * ((x[0]-y[0])²/lengthscale² - 1/lengthscale²)
        # Since diff is already normalized by lengthscale, the formula simplifies
        dK_dxdy = -K * (diff[0]**2 - 1.0) / (self.lengthscale.value**2)
        
        return dK_dxdy.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFDX2DY2(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel for derivatives.

    Computes the covariance between derivatives df/dx[1] and df/dy[1] for inputs $(x, y)$ 
    with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFDX2DY2"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:

        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        
        diff = x - y
        squared_dist = jnp.sum(diff ** 2)
        K = self.variance.value * jnp.exp(-0.5 * squared_dist)
        
        # For RBF kernel, cov(df/dx[0], df/dy[0]) = K * ((x[0]-y[0])²/lengthscale² - 1/lengthscale²)
        # Since diff is already normalized by lengthscale, the formula simplifies
        dK_dxdy = -K * (diff[1]**2 - 1.0) / (self.lengthscale.value**2)
        
        return dK_dxdy.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFDX3DY3(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel for derivatives.

    Computes the covariance between derivatives df/dx[2] and df/dy[2] for inputs $(x, y)$ 
    with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFDX3DY3"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:

        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        
        diff = x - y
        squared_dist = jnp.sum(diff ** 2)
        K = self.variance.value * jnp.exp(-0.5 * squared_dist)
        
        # For RBF kernel, cov(df/dx[0], df/dy[0]) = K * ((x[0]-y[0])²/lengthscale² - 1/lengthscale²)
        # Since diff is already normalized by lengthscale, the formula simplifies
        dK_dxdy = -K * (diff[2]**2 - 1.0) / (self.lengthscale.value**2)
        
        return dK_dxdy.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFDX1DY2(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel for derivatives.

    Computes the covariance between derivatives df/dx[0] and df/dy[1] for inputs $(x, y)$
    with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFDX1DY2"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x)
        y = self.slice_input(y)
        # Original formula with explicit dimensional scaling
        K = self.variance.value * jnp.exp(-0.5*jnp.sum((x-y)**2)/self.lengthscale.value**2)
        # For RBF kernel, cov(df/dx[0], df/dy[1]) = K * ((x[0]-y[0])*(x[1]-y[1]))/lengthscale^4
        # This is the mixed derivative of the kernel with respect to x[0] and y[1]
        dK_dxdy = -K * ((x[0]-y[0])*(x[1]-y[1])) / (self.lengthscale.value**4)

        return dK_dxdy.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)
    
class RBFDX2DY3(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel for derivatives.

    Computes the covariance between derivatives df/dx[1] and df/dy[2] for inputs $(x, y)$
    with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFDX2DY3"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x)
        y = self.slice_input(y)
        # Original formula with explicit dimensional scaling
        K = self.variance.value * jnp.exp(-0.5*jnp.sum((x-y)**2)/self.lengthscale.value**2)
        # For RBF kernel, cov(df/dx[0], df/dy[1]) = K * ((x[0]-y[0])*(x[1]-y[1]))/lengthscale^4
        # This is the mixed derivative of the kernel with respect to x[0] and y[1]
        dK_dxdy = -K * ((x[1]-y[1])*(x[2]-y[2])) / (self.lengthscale.value**4)

        return dK_dxdy.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFDX1DY3(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel for derivatives.

    Computes the covariance between derivatives df/dx[0] and df/dy[2] for inputs $(x, y)$
    with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFDX1DY3"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x)
        y = self.slice_input(y)
        # Original formula with explicit dimensional scaling
        K = self.variance.value * jnp.exp(-0.5*jnp.sum((x-y)**2)/self.lengthscale.value**2)
        # For RBF kernel, cov(df/dx[0], df/dy[1]) = K * ((x[0]-y[0])*(x[1]-y[1]))/lengthscale^4
        # This is the mixed derivative of the kernel with respect to x[0] and y[1]
        dK_dxdy = -K * ((x[0]-y[0])*(x[2]-y[2])) / (self.lengthscale.value**4)

        return dK_dxdy.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFDX2DY1(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel for derivatives.

    Computes the covariance between derivatives df/dx[1] and df/dy[0] for inputs $(x, y)$
    with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFDX2DY1"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x)
        y = self.slice_input(y)
        # Original formula with explicit dimensional scaling
        K = self.variance.value * jnp.exp(-0.5*jnp.sum((x-y)**2)/self.lengthscale.value**2)
        # For RBF kernel, cov(df/dx[0], df/dy[1]) = K * ((x[0]-y[0])*(x[1]-y[1]))/lengthscale^4
        # This is the mixed derivative of the kernel with respect to x[0] and y[1]
        dK_dxdy = -K * ((x[1]-y[1])*(x[0]-y[0])) / (self.lengthscale.value**4)
        return dK_dxdy.squeeze()
    
    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFDX3DY2(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel for derivatives.

    Computes the covariance between derivatives df/dx[2] and df/dy[1] for inputs $(x, y)$
    with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFDX3DY2"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x)
        y = self.slice_input(y)
        # Original formula with explicit dimensional scaling
        K = self.variance.value * jnp.exp(-0.5*jnp.sum((x-y)**2)/self.lengthscale.value**2)
        # For RBF kernel, cov(df/dx[0], df/dy[1]) = K * ((x[0]-y[0])*(x[1]-y[1]))/lengthscale^4
        # This is the mixed derivative of the kernel with respect to x[0] and y[1]
        dK_dxdy = -K * ((x[2]-y[2])*(x[1]-y[1])) / (self.lengthscale.value**4)
        return dK_dxdy.squeeze()
    
    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFDX3DY1(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel for derivatives.

    Computes the covariance between derivatives df/dx[2] and df/dy[0] for inputs $(x, y)$
    with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFDX3DY1"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x)
        y = self.slice_input(y)
        # Original formula with explicit dimensional scaling
        K = self.variance.value * jnp.exp(-0.5*jnp.sum((x-y)**2)/self.lengthscale.value**2)
        # For RBF kernel, cov(df/dx[0], df/dy[1]) = K * ((x[0]-y[0])*(x[1]-y[1]))/lengthscale^4
        # This is the mixed derivative of the kernel with respect to x[0] and y[1]
        dK_dxdy = -K * ((x[2]-y[2])*(x[0]-y[0])) / (self.lengthscale.value**4)
        return dK_dxdy.squeeze()
    
    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFDX1X1Y(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel for second derivatives.

    Computes the covariance between second derivative d²f/dx[0]² and function value f(y)
    for inputs $(x, y)$ with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "MYRBFDX1X1Y"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:


        # x = self.slice_input(x)
        # y = self.slice_input(y)

        # print("x: ",x)
        # print("y: ",y)

        # dK_dx1x1 = self.variance.value*jnp.exp(-0.5*jnp.sum((x-y)**2)/self.lengthscale.value**2) * ((x[0]-y[0])**2/self.lengthscale.value**4 - 1/self.lengthscale.value**2)

        # print(dK_dx1x1)

        # x = self.slice_input(x) / self.lengthscale.value
        # y = self.slice_input(y) / self.lengthscale.value
        
        # diff = x - y
        # squared_dist = jnp.sum(diff ** 2)
        # K = self.variance.value * jnp.exp(-0.5 * squared_dist)
        
        # # For RBF kernel, cov(d²f/dx[0]², f(y)) = K * ((x[0]-y[0])²/lengthscale² - 1)/lengthscale²
        # # This is the second derivative of the kernel with respect to x[0]
        # dK_dx1x1_2 = K * ((diff[0]**2 / self.lengthscale.value**2) - 1.0) / self.lengthscale.value**2

        # print(dK_dx1x1_2)

        x = self.slice_input(x)
        y = self.slice_input(y)
        
        # Original formula with explicit dimensional scaling
        K = self.variance.value * jnp.exp(-0.5*jnp.sum((x-y)**2)/self.lengthscale.value**2)
        dK_dx1x1 = K * ((x[0]-y[0])**2/self.lengthscale.value**4 - 1/self.lengthscale.value**2)

        # print(dK_dx1x1)
        
        # # To check correctness, compute the same formula a different way
        # x_scaled = x / self.lengthscale.value
        # y_scaled = y / self.lengthscale.value
        # diff = x_scaled - y_scaled
        # K_scaled = self.variance.value * jnp.exp(-0.5 * jnp.sum(diff**2))
        # # For pre-scaled input, the formula simplifies to:
        # dK_dx1x1_2 = K_scaled * ((diff[0]**2) - 1.0) / self.lengthscale.value**2

        # print(dK_dx1x1_2)
        

        return dK_dx1x1.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFXDY1Y1(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel for second derivatives.

    Computes the covariance between function value f(x) and second derivative d²f/dy[0]²
    for inputs $(x, y)$ with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFXDY1Y1"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x)
        y = self.slice_input(y)

        # For RBF kernel, cov(f(x), d²f/dy[0]²) = K * ((x[0]-y[0])²/lengthscale⁴ - 1/lengthscale²)
        K = self.variance.value * jnp.exp(-0.5*jnp.sum((x-y)**2)/self.lengthscale.value**2)
        dK_dy1y1 = K * ((x[0]-y[0])**2/self.lengthscale.value**4 - 1/self.lengthscale.value**2)

        return dK_dy1y1.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFDX1X1DY1Y1(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel for second derivatives.

    Computes the covariance between second derivatives d²f/dx[0]² and d²f/dy[0]²
    for inputs $(x, y)$ with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFDX1X1DY1Y1"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:

        diff = x - y
        squared_dist = jnp.sum(diff ** 2)
        K = self.variance.value * jnp.exp(-0.5 * squared_dist / self.lengthscale.value**2)

        # Covariance between second derivatives d²f/dx[0]² and d²f/dy[0]²
        d2K_dx1x1dy1y1 = K * ((diff[0]**4 / self.lengthscale.value**8) 
                              - (6.0 * diff[0]**2 / self.lengthscale.value**6) 
                              + (3.0 / self.lengthscale.value**4))
        print("1", d2K_dx1x1dy1y1)
    
        return d2K_dx1x1dy1y1.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFDX1X2Y(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel for mixed second derivatives.

    Computes the covariance between mixed second derivative d²f/(dx[0]dx[1]) and function value f(y)
    for inputs $(x, y)$ with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFDX1X2Y"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        
        diff = x - y
        squared_dist = jnp.sum(diff ** 2)
        K = self.variance.value * jnp.exp(-0.5 * squared_dist)
        
        # For RBF kernel, cov(d²f/(dx[0]dx[1]), f(y)) = K * (diff[0]*diff[1])/lengthscale^4
        # This is the mixed second derivative of the kernel with respect to x[0] and x[1]
        dK_dx1x2 = K * (diff[0] * diff[1]) / (self.lengthscale.value**2)
        
        return dK_dx1x2.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFXDY1Y2(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel for mixed second derivatives.

    Computes the covariance between function value f(x) and mixed second derivative d²f/(dy[0]dy[1])
    for inputs $(x, y)$ with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFXDY1Y2"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        diff = x - y
        squared_dist = jnp.sum(diff ** 2)
        K = self.variance.value * jnp.exp(-0.5 * squared_dist / self.lengthscale.value**2)

        # Compute the covariance
        dK_dy1y2 = K * (diff[0] * diff[1]) / self.lengthscale.value**4
        
        return dK_dy1y2.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFDX1DX2DY1Y2(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel for mixed second derivatives.

    Computes the covariance between mixed second derivative d²f/(dx[0]dx[1]) and 
    mixed second derivative d²f/(dy[0]dy[1]) for inputs $(x, y)$ with lengthscale 
    parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFDX1DX2DY1Y2"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        diff = x - y
        squared_dist = jnp.sum(diff ** 2)
        K = self.variance.value * jnp.exp(-0.5 * squared_dist / self.lengthscale.value**2)

        # Compute the covariance
        d2K_dx1x2dy1y2 = K * ((diff[0] * diff[1])**2 - 1) / self.lengthscale.value**4

        return d2K_dx1x2dy1y2.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

###
class RBFDX2X2Y(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel for second derivatives.

    Computes the covariance between second derivative d²f/dx[1]² and function value f(y)
    for inputs $(x, y)$ with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "MYRBFDX2X2Y"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:

        x = self.slice_input(x)
        y = self.slice_input(y)
        
        K = self.variance.value * jnp.exp(-0.5*jnp.sum((x-y)**2)/self.lengthscale.value**2)
        dK_dx1x1 = K * ((x[1]-y[1])**2/self.lengthscale.value**4 - 1/self.lengthscale.value**2)

        return dK_dx1x1.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFXDY2Y2(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel for second derivatives.

    Computes the covariance between function value f(x) and second derivative d²f/dy[1]²
    for inputs $(x, y)$ with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFXDY2Y2"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x)
        y = self.slice_input(y)

        # For RBF kernel, cov(f(x), d²f/dy[0]²) = K * ((x[0]-y[0])²/lengthscale⁴ - 1/lengthscale²)
        K = self.variance.value * jnp.exp(-0.5*jnp.sum((x-y)**2)/self.lengthscale.value**2)
        dK_dy1y1 = K * ((x[1]-y[1])**2/self.lengthscale.value**4 - 1/self.lengthscale.value**2)

        return dK_dy1y1.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)

class RBFDX2X2DY2Y2(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel for second derivatives.

    Computes the covariance between second derivatives d²f/dx[1]² and d²f/dy[1]²
    for inputs $(x, y)$ with lengthscale parameter $\ell$ and variance $\sigma^2$.
    """

    name: str = "RBFDX2X2DY2Y2"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:

        diff = x - y
        squared_dist = jnp.sum(diff ** 2)
        K = self.variance.value * jnp.exp(-0.5 * squared_dist / self.lengthscale.value**2)

        # Covariance between second derivatives d²f/dx[0]² and d²f/dy[0]²
        d2K_dx1x1dy1y1 = K * ((diff[1]**4 / self.lengthscale.value**8) 
                              - (6.0 * diff[1]**2 / self.lengthscale.value**6) 
                              + (3.0 / self.lengthscale.value**4))
        print("1", d2K_dx1x1dy1y1)
    
        return d2K_dx1x1dy1y1.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)
