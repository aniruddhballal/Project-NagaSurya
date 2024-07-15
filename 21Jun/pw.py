import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
import csv
import time

#8.67 seconds for lmax=30
#18.02 seconds for lmax=40

def b(y, x):
    return np.sin(y)
    #return np.sin(x)
    #return np.cos(y)
    #return np.cos(x)

def calcalm(b, lmax, num_points_theta, num_points_phi):
    x = np.linspace(0, 2 * np.pi, num_points_phi)  # phi
    y = np.linspace(0, np.pi, num_points_theta)    # theta
    y, x = np.meshgrid(y, x)
    alm = {}
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            ylm = sph_harm(m, l, x, y)
            integrand = b(y, x) * np.conj(ylm) * np.sin(y)
            alm[(l, m)] = np.sum(integrand) * (np.pi / num_points_theta) * (2 * np.pi / num_points_phi)
    
    # Write alm values to a CSV file
    with open('alm_values_sin(y).csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['l', 'm', 'alm'])
        for (l, m), value in alm.items():
            writer.writerow([l, m, value])
    
    return alm

def recons(alm, x, y, lmax):
    sizex = x.shape
    brecons = np.zeros(sizex, dtype=complex)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            ylm = sph_harm(m, l, x, y)
            brecons += alm[(l, m)] * ylm
    return brecons.real

def plottyminmax(ax, x, y, bvals, type, lmax, num_points_theta, num_points_phi, contnum):
    vmin = np.min(bvals)
    vmax = np.max(bvals)
    clevels = np.linspace(vmin, vmax, contnum + 1)
    contourf_plot = ax.contourf(x, y, bvals, levels=clevels, cmap='viridis', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(contourf_plot, ax=ax, label='Value')
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f'{vmin:.2f}', f'{vmax:.2f}'])
    ax.set_xlabel('x or phi')
    ax.set_ylabel('y or theta')
    name = ''
    if type == 1:
        name = 'og function (sin theta)'
    elif type == 2:
        name = f'recons (lmax: {lmax}, num_points_theta: {num_points_theta}, num_points_phi: {num_points_phi})'
    elif type == 3:
        name = 'Difference'
    ax.set_title(name)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels([r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.set_yticks(np.linspace(0, np.pi, 5))
    ax.set_yticklabels([r'0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])

def plotty01(ax, x, y, bvals, type, lmax, num_points_theta, num_points_phi, contnum):
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
    elif type == 2:
        name = f'recons (lmax: {lmax}, num_points_theta: {num_points_theta}, num_points_phi: {num_points_phi})'
    elif type == 3:
        name = 'Difference'
    ax.set_title(name)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels([r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.set_yticks(np.linspace(0, np.pi, 5))
    ax.set_yticklabels([r'0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])

# Start stopwatch when Enter is pressed
input("Press Enter to start the stopwatch...")
start_time = time.time()

fig, axs = plt.subplots(2, 2, figsize=(15, 12))
axs = axs.flatten()

num_points_theta = 200
num_points_phi = 400
contnum = 100

x = np.linspace(0, 2 * np.pi, num_points_phi)
y = np.linspace(0, np.pi, num_points_theta)
y, x = np.meshgrid(y, x)
plottyminmax(axs[0], x, y, b(y, x), 1, 5, num_points_theta, num_points_phi, contnum)

lmax = 60
alm = calcalm(b, lmax, num_points_theta, num_points_phi)
x = np.linspace(0, 2 * np.pi, num_points_phi)
y = np.linspace(0, np.pi, num_points_theta)
y, x = np.meshgrid(y, x)
b_reconstructed = recons(alm, x, y, lmax)
plotty01(axs[1], x, y, b_reconstructed, 2, lmax, num_points_theta, num_points_phi, contnum)

# Calculate and plot the difference
difference = b(y, x) - b_reconstructed
plottyminmax(axs[2], x, y, difference, 3, lmax, num_points_theta, num_points_phi, contnum)

axs[3].axis('off')  # Turn off the unused subplot

# Stop stopwatch and calculate runtime
end_time = time.time()
runtime = end_time - start_time

# Display runtime on the plot
plt.figtext(0.95, 0.05, f"Runtime: {runtime:.2f} seconds", va='bottom', ha='right')

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
plt.show()
