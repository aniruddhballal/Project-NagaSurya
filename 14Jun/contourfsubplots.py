import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt

def b(y, x):
    return np.sin(y)
    #return np.cos(y)
    #return np.exp(y)
    #return y

def calcalm(b, lmax, n):
    y = np.linspace(0, np.pi, n)
    x = np.linspace(0, 2 * np.pi, n)
    y, x = np.meshgrid(y, x)
    alm = {}
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            ylm = sph_harm(m, l, x, y)
            integrand = b(y, x) * np.conj(ylm) * np.sin(y)
            alm[(l, m)] = np.sum(integrand) * (np.pi / n) * (2 * np.pi / n)
    return alm

def recons(alm, y, x, lmax):
    brecons = np.zeros_like(y, dtype=complex)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            ylm = sph_harm(m, l, x, y)
            brecons += alm[(l, m)] * ylm
    return brecons.real

def plot_functions(y, x, original_bvals, reconstructed_bvals, contnum=20):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    contour_levels_orig = np.linspace(np.min(original_bvals), np.max(original_bvals), contnum)
    contourf_plot_orig = axs[0].contourf(x, y, original_bvals, levels=contour_levels_orig, cmap='viridis')
    fig.colorbar(contourf_plot_orig, ax=axs[0], label='Value')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title("Original Function: b(y, x) = sin(y)")
    axs[0].set_xticks(np.linspace(0, 2 * np.pi, 5))
    axs[0].set_xticklabels([r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    axs[0].set_yticks(np.linspace(0, np.pi, 5))
    axs[0].set_yticklabels([r'0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    axs[0].grid(True)

    contour_levels_recons = np.linspace(np.min(reconstructed_bvals), np.max(reconstructed_bvals), contnum)
    contourf_plot_recons = axs[1].contourf(x, y, reconstructed_bvals, levels=contour_levels_recons, cmap='viridis')
    fig.colorbar(contourf_plot_recons, ax=axs[1], label='Value')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_title("Reconstructed Function")
    axs[1].set_xticks(np.linspace(0, 2 * np.pi, 5))
    axs[1].set_xticklabels([r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    axs[1].set_yticks(np.linspace(0, np.pi, 5))
    axs[1].set_yticklabels([r'0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# Parameters
lmax = 5
n = 100

# Compute alm coefficients
alm = calcalm(b, lmax, n)

# Create a grid for y and x
y = np.linspace(0, np.pi, n)
x = np.linspace(0, 2 * np.pi, n)
y, x = np.meshgrid(y, x)

# Reconstruct the function
brecons = recons(alm, y, x, lmax)

# Plot the original and reconstructed functions
plot_functions(y, x, b(y, x), brecons)