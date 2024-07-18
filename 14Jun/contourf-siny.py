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

def plot_function(y, x, bvals, name, contnum=20):
    plt.figure(figsize=(10, 5))
    contour_levels = np.linspace(np.min(bvals), np.max(bvals), contnum)
    contourf_plot = plt.contourf(x, y, bvals, levels=contour_levels, cmap='viridis')
    plt.colorbar(contourf_plot, label='Value')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(name)
    plt.xticks(np.linspace(0, 2 * np.pi, 5), [r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    plt.yticks(np.linspace(0, np.pi, 5), [r'0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    plt.grid(True)
    plt.show()

lmax = 5
n = 100

alm = calcalm(b, lmax, n)

y = np.linspace(0, np.pi, n)
x = np.linspace(0, 2 * np.pi, n)
y, x = np.meshgrid(y, x)

brecons = recons(alm, y, x, lmax)

plot_function(y, x, b(y, x), "Original Function: b(y, x) = sin(y)")
plot_function(y, x, brecons, "Reconstructed Function")