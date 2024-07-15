import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm

# Set the values for l and m
l = 5  # Example l value
m = 0  # Example m value

# Create a meshgrid for theta and phi
theta = np.linspace(0, np.pi, 360)
phi = np.linspace(0, 2 * np.pi, 720)
theta, phi = np.meshgrid(theta, phi)

# Calculate the spherical harmonics
Y_lm = sph_harm(m, l, phi, theta)

# Calculate the real part of the spherical harmonics
Y_lm_real = np.real(Y_lm)

# Convert spherical coordinates to Cartesian coordinates for plotting
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Normalize Y_lm_real for color mapping
norm = plt.Normalize(Y_lm_real.min(), Y_lm_real.max())
colors = plt.cm.coolwarm(norm(Y_lm_real))

# Plot the spherical harmonics on the surface of a sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, antialiased=False, shade=False)

# Add a color bar which maps values to colors
mappable = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
mappable.set_array(Y_lm_real)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set aspect ratio
ax.set_box_aspect([1, 1, 1])  # aspect ratio is 1:1:1

# Set plot title
ax.set_title(f'Real part of Spherical Harmonics Y_{l}^{m}')

plt.show()