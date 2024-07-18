import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt

def b(y, x):
    return np.sin(y)
    # return np.cos(y)
    # return np.exp(y)
    # return y

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

def plot_functions(original_bvals, reconstructed_bvals_list, params, contnum=20):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot original function
    y_orig = np.linspace(0, np.pi, 100)
    x_orig = np.linspace(0, 2 * np.pi, 100)
    x_orig, y_orig = np.meshgrid(x_orig, y_orig)
    contour_levels_orig = np.linspace(np.min(original_bvals), np.max(original_bvals), contnum)
    contourf_plot_orig = axs[0, 0].contourf(x_orig, y_orig, original_bvals, levels=contour_levels_orig, cmap='viridis')
    fig.colorbar(contourf_plot_orig, ax=axs[0, 0], label='Value')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    axs[0, 0].set_title("Original Function: b(y, x) = sin(y)")
    axs[0, 0].set_xticks(np.linspace(0, 2 * np.pi, 5))
    axs[0, 0].set_xticklabels([r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    axs[0, 0].set_yticks(np.linspace(0, np.pi, 5))
    axs[0, 0].set_yticklabels([r'0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    axs[0, 0].grid(True)

    # Plot reconstructed functions
    for i, (brecons, (lmax, n)) in enumerate(zip(reconstructed_bvals_list, params)):
        row, col = divmod(i + 1, 2)
        y = np.linspace(0, np.pi, n)
        x = np.linspace(0, 2 * np.pi, n)
        x, y = np.meshgrid(x, y)
        contour_levels_recons = np.linspace(np.min(brecons), np.max(brecons), contnum)
        contourf_plot_recons = axs[row, col].contourf(x, y, brecons, levels=contour_levels_recons, cmap='viridis')
        fig.colorbar(contourf_plot_recons, ax=axs[row, col], label='Value')
        axs[row, col].set_xlabel('x')
        axs[row, col].set_ylabel('y')
        axs[row, col].set_title(f"Reconstructed (lmax={lmax}, n={n})")
        axs[row, col].set_xticks(np.linspace(0, 2 * np.pi, 5))
        axs[row, col].set_xticklabels([r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        axs[row, col].set_yticks(np.linspace(0, np.pi, 5))
        axs[row, col].set_yticklabels([r'0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
        axs[row, col].grid(True)

    plt.tight_layout()
    plt.show()

# Parameters
lmax_values = [1, 15, 5]
n_values = [1000, 1000, 1000]
params = list(zip(lmax_values, n_values))

# Compute the original function
y_orig = np.linspace(0, np.pi, 100)
x_orig = np.linspace(0, 2 * np.pi, 100)
y_orig, x_orig = np.meshgrid(y_orig, x_orig)
original_bvals = b(y_orig, x_orig)

# Compute alm coefficients and reconstruct functions
reconstructed_bvals_list = []
for lmax, n in params:
    alm = calcalm(b, lmax, n)
    y = np.linspace(0, np.pi, n)
    x = np.linspace(0, 2 * np.pi, n)
    y, x = np.meshgrid(y, x)
    brecons = recons(alm, y, x, lmax)
    reconstructed_bvals_list.append(brecons)

# Plot the original and reconstructed functions
plot_functions(original_bvals, reconstructed_bvals_list, params)
