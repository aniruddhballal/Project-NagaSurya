import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt

def b(y, x):
    return np.sin(y)

def calcalm(b, lmax, n):
    x = np.linspace(0, 2 * np.pi, n)  # phi
    y = np.linspace(0, np.pi, n)      # theta
    y, x = np.meshgrid(y, x)
    alm = {}
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            ylm = sph_harm(m, l, x, y)
            integrand = b(y, x) * np.conj(ylm) * np.sin(y)
            alm[(l, m)] = np.sum(integrand) * (np.pi / n) * (2 * np.pi / n)
    return alm

def recons(alm, x, y, lmax):
    sizex = x.shape
    brecons = np.zeros(sizex, dtype=complex)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            ylm = sph_harm(m, l, x, y)
            brecons += alm[(l, m)] * ylm
    return brecons.real

def plotty(ax, x, y, bvals, type, lmax, num_points, contnum=100):
    clevels = np.linspace(0, 1, contnum)
    contourf_plot = ax.contourf(x, y, bvals, levels=clevels, cmap='viridis', vmin=0, vmax=1)
    num_ticks = min(10, contnum)
    ticks = np.linspace(0, 1, num_ticks)
    cbar = plt.colorbar(contourf_plot, ax=ax, label='Value', ticks=ticks)
    ax.set_xlabel('x or phi')
    ax.set_ylabel('y or theta')
    name = ''
    if type == 1:
        name = 'og function (sin theta)'
    else:
        name = 'recons (lmax: ' + str(lmax) + ', num_points: ' + str(num_points) + ')'
    ax.set_title(name)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels([r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.set_yticks(np.linspace(0, np.pi, 5))
    ax.set_yticklabels([r'0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    # ax.grid(True)

lmax = 5
num_points = 100
alm = calcalm(b, lmax, num_points)
x = np.linspace(0, 2 * np.pi, num_points)
y = np.linspace(0, np.pi, num_points)
y, x = np.meshgrid(y, x)

b_reconstructed = recons(alm, x, y, lmax)

fig, axs = plt.subplots(1, 2, figsize=(1, 4))
plotty(axs[0], x, y, b(y, x), 1, lmax, num_points)
plotty(axs[1], x, y, b_reconstructed, 2, lmax, num_points)
plt.tight_layout()
plt.show()