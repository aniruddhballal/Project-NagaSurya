import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
import scipy.integrate
from tqdm import tqdm
import csv

def b(y, x):
    return np.sin(y)

def read_alm_from_csv(csv_filename):
    alm = {}
    try:
        with open(csv_filename, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                l = int(row['l'])
                m = int(row['m'])
                alm[(l, m)] = complex(row['alm'])
    except FileNotFoundError:
        pass
    return alm

def write_alm_to_csv(csv_filename, alm):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['l', 'm', 'alm'])
        for (l, m), value in alm.items():
            writer.writerow([l, m, value])

def adaptive_calcalm(b, lmax, num_points_theta, num_points_phi):
    alm = {}

    csv_filename = f'theta{num_points_theta}, phi{num_points_phi}, lmax{lmax}, siny.csv'
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

# Parameters
theta_range = range(10, 101, 10)
phi_range = range(20, 201, 20)
lmax = 10
abs_tot_sum_values = []

for num_points_theta in theta_range:
    for num_points_phi in phi_range:
        x = np.linspace(0, 2 * np.pi, num_points_phi)
        y = np.linspace(0, np.pi, num_points_theta)
        y, x = np.meshgrid(y, x)

        alm = adaptive_calcalm(b, lmax, num_points_theta, num_points_phi)
        b_recons = recons(alm, x, y, lmax)
        delta = b(y, x) - b_recons
        abs_tot_sum = np.sum(np.abs(delta))
        abs_tot_sum_values.append((num_points_theta, num_points_phi, abs_tot_sum))

# Plotting
num_points_theta_values, num_points_phi_values, abs_tot_sum_values = zip(*abs_tot_sum_values)
theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing='ij')
abs_tot_sum_grid = np.array(abs_tot_sum_values).reshape(theta_grid.shape)

plt.figure(figsize=(10, 8))
contour = plt.contourf(theta_grid, phi_grid, abs_tot_sum_grid, levels=20, cmap='viridis')
plt.colorbar(contour, label='Sum of abs(delta)')
plt.xlabel('num_points_theta')
plt.ylabel('num_points_phi')
plt.title('Sum of abs(delta) for different (num_points_theta, num_points_phi) with lmax = 10')
plt.show()
