import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
import csv
import os
import time
from datetime import datetime, timedelta, timezone
import scipy.integrate
import winsound
from tqdm import tqdm

# IST is UTC + 5:30
IST = timezone(timedelta(hours=5, minutes=30))

def b(y, x):
    # return np.sin(y)
    return np.sin(x)
    # return np.cos(y)
    # return np.cos(x)

# Ensure the 'csvs' directory exists
csv_folder = 'csvs'
os.makedirs(csv_folder, exist_ok=True)

def read_alm_from_csv(csv_filename):
    alm = {}
    if os.path.exists(csv_filename):
        with open(csv_filename, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                l = int(row['l'])
                m = int(row['m'])
                alm[(l, m)] = complex(row['alm'])  # string to complex
    return alm

def write_alm_to_csv(csv_filename, alm):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['l', 'm', 'alm'])
        for (l, m), value in alm.items():
            writer.writerow([l, m, value])

def adaptive_calcalm(b, lmax, num_points_theta, num_points_phi):
    alm = {}

    csv_filename = os.path.join(csv_folder, f'theta{num_points_theta}, phi{num_points_phi}, lmax{lmax}, sinx.csv')
    alm.update(read_alm_from_csv(csv_filename))

    def integrand_real(theta, phi, l, m):
        ylm = sph_harm(m, l, phi, theta)
        return (b(theta, phi) * np.conj(ylm) * np.sin(theta)).real

    def integrand_imag(theta, phi, l, m):
        ylm = sph_harm(m, l, phi, theta)
        return (b(theta, phi) * np.conj(ylm) * np.sin(theta)).imag

    new_calculations_needed = any((l, m) not in alm for l in range(lmax + 1) for m in range(-l, l + 1))

    if new_calculations_needed:
        total_calculations = (lmax + 1) * (lmax + 1)
        progress_bar = tqdm(total=total_calculations, desc="Calculating coefficients")
    
    for l in range(lmax + 1):
        if (l, 0) not in alm:
            for m in range(-l, l + 1):
                real_result = scipy.integrate.dblquad(
                    integrand_real, 0, 2 * np.pi, lambda x: 0, lambda x: np.pi,
                    args=(l, m)
                )
                imag_result = scipy.integrate.dblquad(
                    integrand_imag, 0, 2 * np.pi, lambda x: 0, lambda x: np.pi,
                    args=(l, m)
                )
                alm[(l, m)] = real_result[0] + 1j * imag_result[0]
                if new_calculations_needed:
                    progress_bar.update(1)

    if new_calculations_needed:
        progress_bar.close()
        write_alm_to_csv(csv_filename, alm)
    return alm

def recons(alm, x, y, lmax):
    sizex = x.shape
    brecons = np.zeros(sizex, dtype=complex)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            if np.isnan(alm[(l, m)]):
                continue  # Skip if alm is NaN
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
        name = 'og function (sin x)'
    elif type == 2:
        name = f'recons (lmax: {lmax}, num_points_theta: {num_points_theta}, num_points_phi: {num_points_phi})'
    elif type == 3:
        name = 'Delta'
    ax.set_title(name)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels([r'0', r'$\pi/2$', r'$2\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.set_yticks(np.linspace(0, np.pi, 5))
    ax.set_yticklabels([r'0', r'$\pi/4$', r'$pi/2$', r'$3\pi/4$', r'$pi$'])

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
        name = 'og function (sin x)'
    elif type == 2:
        name = f'recons (lmax: {lmax}, theta: {num_points_theta}, phi: {num_points_phi}, b: sin(x))'
    elif type == 3:
        name = 'Delta'
    ax.set_title(name)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels([r'0', r'$\pi/2$', r'$2\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.set_yticks(np.linspace(0, np.pi, 5))
    ax.set_yticklabels([r'0', r'$\pi/4$', r'$pi/2$', r'$3\pi/4$', r'$pi$'])

def beepbeep(f,d):
    winsound.Beep(f,d)

fig, axs = plt.subplots(2, 2, figsize=(15, 12))
axs = axs.flatten()

num_points_theta = 20
num_points_phi = 40
contnum = 100
lmax = 30 # Example of higher lmax

print("\n------------------------------------------------------------------------------------------------------------------------")
print("Imported all necessary libraries, compiled successfully.")
print("Current values:\n\tnum_points_theta\t-\t" + str(num_points_theta) + "\n\tnum_points_phi\t\t-\t" + str(num_points_phi) + "\n\tlmax\t\t\t-\t" + str(lmax))

input("Press Enter to confirm these values and start the process:")

start_time = time.time()
start_ist = datetime.now(IST)
start_ist_str = start_ist.strftime("%Y-%m-%d %H:%M:%S %Z")

beepbeep(500,1300)

print("Process started...\nIST now: " + start_ist_str)

x = np.linspace(0, 2 * np.pi, num_points_phi)
y = np.linspace(0, np.pi, num_points_theta)
y, x = np.meshgrid(y, x)
plottyminmax(axs[0], x, y, b(y, x), 1, 5, num_points_theta, num_points_phi, contnum)

alm = adaptive_calcalm(b, lmax, num_points_theta, num_points_phi)
x = np.linspace(0, 2 * np.pi, num_points_phi)
y = np.linspace(0, np.pi, num_points_theta)
y, x = np.meshgrid(y, x)
b_recons = recons(alm, x, y, lmax)
plottyminmax(axs[1], x, y, b_recons, 2, lmax, num_points_theta, num_points_phi, contnum)

# Calculate and plot the difference
delta = b(y, x) - b_recons
plottyminmax(axs[2], x, y, delta, 3, lmax, num_points_theta, num_points_phi, contnum)
abs_tot_sum = np.sum(np.abs(delta))

# Write the total absolute error to a text file inside the specified folder
error_folder = 'total abs errors - lmax 20 sinx'
os.makedirs(error_folder, exist_ok=True)
error_filename = os.path.join(error_folder, f'TotAbsErr_theta{num_points_theta}_phi{num_points_phi}.txt')
with open(error_filename, 'w') as file:
    file.write(f'{abs_tot_sum:.4f}')

axs[3].axis('off')  # Turn off the unused subplot

end_time = time.time()
end_ist = datetime.now(IST)
runtime = end_time - start_time

# Time in IST
end_ist_str = end_ist.strftime("%Y-%m-%d %H:%M:%S %Z")

csv_filename = os.path.join(csv_folder, f'theta{num_points_theta}, phi{num_points_phi}, lmax{lmax}, sinx.csv')
plt.figtext(0.95, 0.05, f"CSV used: {csv_filename}\nSum(abs(delta)): {abs_tot_sum:.4f}\n\n\nRuntime: {runtime:.4f} seconds\nStart Time (IST): {start_ist_str}\nEnd Time (IST): {end_ist_str}", va='bottom', ha='right')

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)

beepbeep(600,600)

plt.show()