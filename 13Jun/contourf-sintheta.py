import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt

# Define the function b(theta, phi) = sin(theta)
def b(theta, phi):
    return np.sin(theta)

# Compute the a_{lm} coefficients
def compute_alm(b, L, num_points=100):
    theta = np.linspace(0, np.pi, num_points)
    phi = np.linspace(0, 2 * np.pi, num_points)
    theta, phi = np.meshgrid(theta, phi)
    alm = {}
    for l in range(L + 1):
        for m in range(-l, l + 1):
            ylm = sph_harm(m, l, phi, theta)
            integrand = b(theta, phi) * np.conj(ylm) * np.sin(theta)
            alm[(l, m)] = np.sum(integrand) * (np.pi / num_points) * (2 * np.pi / num_points)
    return alm

# Reconstruct the function from a_{lm} coefficients
def reconstruct_b(alm, theta, phi, L):
    b_reconstructed = np.zeros_like(theta, dtype=complex)
    for l in range(L + 1):
        for m in range(-l, l + 1):
            ylm = sph_harm(m, l, phi, theta)
            b_reconstructed += alm[(l, m)] * ylm
    return b_reconstructed.real

# Plot the function using contourf
def plot_function(theta, phi, b_values, title):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    c = ax.contourf(phi, theta, b_values, cmap='viridis')
    fig.colorbar(c)
    ax.set_title(title)
    plt.show()

# Define parameters
L = 10  # Maximum degree
num_points = 100  # Number of points for theta and phi

# Compute alm coefficients
alm = compute_alm(b, L, num_points)

# Create a grid for theta and phi
theta = np.linspace(0, np.pi, num_points)
phi = np.linspace(0, 2 * np.pi, num_points)
theta, phi = np.meshgrid(theta, phi)

# Reconstruct the function
b_reconstructed = reconstruct_b(alm, theta, phi, L)

# Plot the original function
plot_function(theta, phi, b(theta, phi), "Original Function: b(theta, phi) = sin(theta)")

# Plot the reconstructed function
plot_function(theta, phi, b_reconstructed, "Reconstructed Function")