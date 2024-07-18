import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm

# Define spherical harmonics plotting function
def plot_spherical_harmonic(ax, l, m, colormap):
    theta = np.linspace(0, np.pi, num=360)
    phi = np.linspace(0, 2 * np.pi, num=720)
    theta, phi = np.meshgrid(theta, phi)
    
    Y_lm = sph_harm(m, l, phi, theta).real
    
    X = np.abs(Y_lm) * np.sin(theta) * np.cos(phi)
    Y = np.abs(Y_lm) * np.sin(theta) * np.sin(phi)
    Z = np.abs(Y_lm) * np.cos(theta)
    
    norm = plt.Normalize(Y_lm.min(), Y_lm.max())
    surf = ax.plot_surface(X, Y, Z, facecolors=plt.get_cmap(colormap)(norm(Y_lm)), edgecolor='none')
    ax.set_title(f'Spherical Harmonics Y_{l}^{m}', fontsize=16)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    return surf, norm

# Specify the l and m values
l = 5
m = 0

# Create figure and subplot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the spherical harmonic
colormap = 'coolwarm'
surf, norm = plot_spherical_harmonic(ax, l, m, colormap)

# Add colorbar
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax, shrink=0.5)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
plt.show()