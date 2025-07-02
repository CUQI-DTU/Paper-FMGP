import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from skimage import measure
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from tqdm import tqdm
import cuqi

def plot_1d_function_with_points(x, y, pts=None, xlim=None, ylim=None):
    x = x.flatten()
    y = y.flatten()
    if xlim is None:
        xlim = (x.min(), x.max())
    if ylim is None:
        ylim = (y.min(), y.max())
    fig, ax = plt.subplots()
    ax.plot(x, y)
    if pts is not None:
        ax.scatter(pts[0], pts[1], c="black", alpha=0.3)
    
    ax.set_xlim([xlim[0], xlim[1]])
    ax.set_ylim([ylim[0], ylim[1]])

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    return fig, ax

def plot_2d_function_with_points(x=None, y=None, z=None, pts=None, vpts=None, xlim=None, ylim=None, vmin=None, vmax=None, cmap=None, marker_size=None, colorbar=False, alpha=0.5, ticks=False, **kwargs):
    cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    if vmin is None and z is not None:
        vmin = z.min()
    if vmax is None and z is not None:
        vmax = z.max()
    if xlim is None and x is not None:
        xlim = (x.min(), x.max())
    if ylim is None and y is not None:
        ylim = (y.min(), y.max())
    if cmap is None:
        cmap = "viridis"
    if marker_size is None:
        marker_size = 8
    fig, ax = plt.subplots(**kwargs, constrained_layout=True)
    if x is not None and y is not None and z is not None:
        pcm = ax.pcolormesh(x, y, z, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        # fig.colorbar(pcm, ax=ax)
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pcm, cax=cax)
    if pts is not None:
        scatter = ax.scatter(pts[:,0], pts[:,1], marker='x', c="black", alpha=alpha)
    if vpts is not None:
        scatter = ax.scatter(vpts[:,0], vpts[:,1], marker='o', c=cols[1], alpha=alpha, s=marker_size, clip_on=False)

    ax.set_aspect('equal', 'box')

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    # plt.tight_layout()

    return fig, ax

def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return np.where(x > 0, 1.0, 0.0)

# %%
def condition_gp(mu1, mu2, Ks, x2):
    # not what we need yet
    """
    Condition a GP prior on a set of points x2.
    x1     mu1,  K11 K12
    x2     mu2,  K21 K22
    """
    mu1 = mu1.flatten()
    mu2 = mu2.flatten()
    K11 = Ks[0]
    K12 = Ks[1]
    K22 = Ks[2]

    mu2_cond = mu2 + K12 @ np.linalg.solve(K22, x2 - mu1)
    K22_cond = K22 - K12 @ np.linalg.solve(K11, K12.T)
    return mu2_cond, K22_cond

def derivative_enhanced_gp(mu_ft, mu_ds, mu_fu, k00tt, k10st, k11ss, k00ut, k01us, k00uu, f0t, f1s):
    # not what we need yet
    k22 = np.block([[k00tt, k10st.T], 
                    [k10st, k11ss]])
    k12 = np.block([[k00ut, k01us]])
    k11 = k00uu

    mu2 = np.hstack([mu_ft.flatten(), mu_ds.flatten()])

    x2 = np.hstack([f0t.flatten(), f1s.flatten()])

    return condition_gp(mu_fu, mu2, [k11, k12, k22], x2)

def draw_samples_with_derivative_enhanced_gp(K00tt, K01ts, K10st, K11ss, K00tu, K10su, K00uu, fun_noise_var, grad_noise_var, f0t_prior_mean, f1s_prior_mean, f0u_prior_mean, f0t, f1s_samples):
    fun_noise_matrix = np.eye(len(K00tt)) * fun_noise_var
    grad_noise_matrix = np.eye(len(K11ss)) * grad_noise_var
    A_full = np.block([[K00tt+fun_noise_matrix, K01ts], 
                    [K10st, K11ss+grad_noise_matrix]])
    B = np.vstack([K00tu, K10su])
    C = K00uu

    no_of_samples_per_s = 1
    f0u_samples = np.zeros((len(f0u_prior_mean), f1s_samples.shape[1]*no_of_samples_per_s))
    predict_cov = C - B.T @ np.linalg.solve(A_full, B)
    predict_cov = (predict_cov+predict_cov.T)/2
    f0t_f1s_prior_mean = np.hstack([f0t_prior_mean.flatten(), f1s_prior_mean.flatten()])
    # f0u_prior_mean = opt_posterior.prior.mean_function(u).flatten()
    chol_l = np.linalg.cholesky(predict_cov+1e-10*np.eye(predict_cov.shape[0]))
    for i in tqdm(range(f1s_samples.shape[1])):
        data = np.hstack([np.asarray(f0t.flatten()), f1s_samples[:,i]])
        predict_mean = f0u_prior_mean + B.T @ np.linalg.solve(A_full, data-f0t_f1s_prior_mean)
        xi = np.random.randn(predict_cov.shape[0])
        f0u_samples_per_s = predict_mean + chol_l @ xi
        f0u_samples[:,i*no_of_samples_per_s:(i+1)*no_of_samples_per_s] = f0u_samples_per_s.reshape(-1,no_of_samples_per_s)
    return cuqi.samples.Samples(f0u_samples)

def mean_squared_error(y_ref, y_pred):
    return np.mean((y_ref.flatten() - y_pred.flatten()) ** 2)
def mean_absolute_error(y_ref, y_pred):
    return np.mean(np.abs(y_ref.flatten() - y_pred.flatten()))
def root_mean_squared_error(y_ref, y_pred):
    return np.sqrt(np.mean((y_ref.flatten() - y_pred.flatten()) ** 2))

class ReparameterizedGaussian(cuqi.distribution.Gaussian):
    def __init__(self, mean=None, cov=None, **kwargs):
        super().__init__(mean=mean, cov=cov, **kwargs)
    def logpdf(self, x):
        sigma = np.exp(x)
        return super().logpdf(sigma) + np.sum(x)
    def gradient(self, x):
        sigma = np.exp(x)
        return super().gradient(sigma) * sigma + 1.0

def plot_1d_gp(x, mean=None, upper=None, lower=None, data=None, virtual=None, ground_truth=None, xlim=None, ylim=None, legend=False, alpha=0.5, grid_alpha=0.3, grid_linewidth=0.5, grid_on=True, **kwargs):
    cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(**kwargs)
    if x is not None:
        x = x.flatten()
    if mean is not None:
        ax.plot(x, mean, color=cols[0], label="Mean")
    if upper is not None and lower is not None:
        ax.fill_between(x, lower, upper, alpha=0.3, color=cols[0], label="95% CI")
    if data is not None:
        ax.scatter(data[0], data[1], marker="x", c="black", alpha=alpha, label="Data")
    if virtual is not None:
        ax.scatter(virtual[0], virtual[1], marker="o", s=(mpl.rcParams['lines.markersize']**2)/2, c=cols[1], alpha=alpha, label="Virtual data", clip_on=False)
    if ground_truth is not None:
        ax.plot(x, ground_truth, color=cols[3], linestyle="--", label="Ground truth")
    
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    if grid_on:
        ax.grid(color="gray", linestyle="-", alpha=grid_alpha, linewidth=grid_linewidth)

    if xlim is not None:
        ax.set_xlim([xlim[0], xlim[1]])
    if ylim is not None:
        ax.set_ylim([ylim[0], ylim[1]])

    if legend:
        ax.legend()
    
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.tight_layout()

    return fig, ax

def plot_3d_slices(data, coords, z_values=[1.0], x_value=1.0, y_value=0.15, cmap='viridis', figsize=(10, 8), vmin=0.0, vmax=1.0, xlim=None, ylim=None, zlim=None):
    """
    Plot slices of 3D data at specific z values and x, y values with customizable colormap range.
    
    Parameters:
    - data: The data values to visualize
    - coords: Tuple of (x, y, z) coordinates
    - z_values: List of z-values where to create horizontal slices
    - x_value: Value where to create vertical slice in yz plane
    - y_value: Value where to create vertical slice in xz plane
    - cmap: Colormap to use
    - figsize: Figure size
    - vmin: Minimum value for colormap
    - vmax: Maximum value for colormap
    """
    
    # Extract coordinates
    x, y, z = coords
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    
    # Create a figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a grid for slices
    res = 50  # resolution of the planes
    x_grid = np.linspace(x_min, x_max, res)
    y_grid = np.linspace(y_min, y_max, res)
    z_grid = np.linspace(z_min, z_max, res)
    X, Y = np.meshgrid(x_grid, y_grid)
    Y_vert, Z_vert = np.meshgrid(y_grid, z_grid)
    X_vert2, Z_vert2 = np.meshgrid(x_grid, z_grid)
    
    # Original data points for interpolation
    points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
    values = data.flatten()
    
    # Create color normalizer with specified range
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot horizontal slices
    for z_val in z_values:
        Z = np.ones_like(X) * z_val
        slice_points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
        slice_values = griddata(points, values, slice_points, method='linear')
        slice_values = slice_values.reshape(X.shape)
        colors = plt.cm.get_cmap(cmap)(norm(slice_values))
        ax.plot_surface(X, Y, Z, facecolors=colors, alpha=1.0, 
                       rstride=1, cstride=1)
    
    # Plot vertical slice in yz plane
    X_vert = np.ones_like(Y_vert) * x_value
    slice_points_vert = np.vstack([X_vert.flatten(), Y_vert.flatten(), Z_vert.flatten()]).T
    slice_values_vert = griddata(points, values, slice_points_vert, method='linear')
    slice_values_vert = slice_values_vert.reshape(Y_vert.shape)
    colors_vert = plt.cm.get_cmap(cmap)(norm(slice_values_vert))
    ax.plot_surface(X_vert, Y_vert, Z_vert, facecolors=colors_vert, alpha=1.0,
                   rstride=1, cstride=1)
    
    # Plot vertical slice in xz plane
    Y_vert2 = np.ones_like(X_vert2) * y_value
    slice_points_vert2 = np.vstack([X_vert2.flatten(), Y_vert2.flatten(), Z_vert2.flatten()]).T
    slice_values_vert2 = griddata(points, values, slice_points_vert2, method='linear')
    slice_values_vert2 = slice_values_vert2.reshape(X_vert2.shape)
    colors_vert2 = plt.cm.get_cmap(cmap)(norm(slice_values_vert2))
    ax.plot_surface(X_vert2, Y_vert2, Z_vert2, facecolors=colors_vert2, alpha=1.0,
                   rstride=1, cstride=1)
    
    # Add colorbar with the specified range
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax)
    
    if xlim is not None:
        ax.set_xlim([xlim[0], xlim[1]])
    if ylim is not None:
        ax.set_ylim([ylim[0], ylim[1]])
    if zlim is not None:
        ax.set_zlim([zlim[0], zlim[1]])

    plt.tight_layout()

    return fig, ax

def plot_isosurfaces(coords, values, isovalues, grid_size=20, colors=None, alphas=None, 
                     figsize=(10, 8), ax=None, xlabel='Velocity', ylabel='Time', zlabel='Space',
                     add_colorbar=False, show_data_points=False, data_points=None):
    """
    Plot multiple isosurfaces from 3D scattered data with proper color representation.
    
    Parameters remain the same as your original function
    """
    from scipy.interpolate import griddata
    from skimage import measure
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    
    # Create a regular 3D grid
    x_min, x_max = np.min(coords[:,0]), np.max(coords[:,0])
    y_min, y_max = np.min(coords[:,1]), np.max(coords[:,1])
    z_min, z_max = np.min(coords[:,2]), np.max(coords[:,2])
    
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    z = np.linspace(z_min, z_max, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Interpolate scattered data to regular grid
    grid_values = griddata(coords, values, (X, Y, Z), method='linear')
    
    # Create or get figure and axes
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Set white background for better contrast
    ax.set_facecolor('white')
    
    # Default colors and alphas if not provided
    if colors is None:
        cmap = plt.cm.get_cmap('viridis')
        colors = [cmap(i/len(isovalues)) for i in range(len(isovalues))]
    elif isinstance(colors, str):
        colors = [colors] * len(isovalues)
        
    if alphas is None:
        alphas = [0.7] * len(isovalues)
    elif isinstance(alphas, (int, float)):
        alphas = [alphas] * len(isovalues)
    
    # Create custom legend handles with the actual colors
    legend_handles = []
    
    # Plot each isosurface
    for i, isovalue in enumerate(isovalues):
        try:
            verts, faces, normals, _ = measure.marching_cubes(grid_values, level=isovalue)
            
            # Scale vertices back to original coordinate system
            verts[:, 0] = verts[:, 0] * (x_max - x_min) / (grid_size - 1) + x_min
            verts[:, 1] = verts[:, 1] * (y_max - y_min) / (grid_size - 1) + y_min
            verts[:, 2] = verts[:, 2] * (z_max - z_min) / (grid_size - 1) + z_min
            
            # Convert named colors to RGB if necessary
            if isinstance(colors[i], str):
                color_rgb = mcolors.to_rgba(colors[i])
            else:
                color_rgb = colors[i]
                
            # Plot the isosurface with enhanced lighting settings
            mesh = ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                                triangles=faces, alpha=alphas[i], color=color_rgb,
                                shade=True, lightsource=plt.matplotlib.colors.LightSource(azdeg=315, altdeg=45))
            
            # Create proper legend entry
            legend_handles.append(mpatches.Patch(color=colors[i], 
                                               label=f'{isovalue:.{1}f}'))
            
        except Exception as e:
            print(f"Could not generate isosurface for value {isovalue}: {e}")
    
    # Show data points if requested
    if show_data_points and data_points is not None:
        ax.scatter(data_points[:,0], data_points[:,1], data_points[:,2], 
                  c='black', marker='o', s=10, alpha=0.5)
        legend_handles.append(mpatches.Patch(color='black', label='Data points'))
    
    # Add custom legend with proper colors
    ax.legend(handles=legend_handles, loc='upper left', mode = "expand", ncol = len(isovalues), fontsize='small', handletextpad=0.3)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    # ax.set_title(title)

    ax.set_box_aspect(aspect=None, zoom=0.85)
    
    plt.tight_layout()
    
    return fig, ax