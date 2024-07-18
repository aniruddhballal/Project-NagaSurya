import numpy as np
from scipy.special import sph_harm
import scipy.integrate
import matplotlib.pyplot as plt

def b(y, x):
    return np.sin(y)

def adaptive_calcalm(b, lmax, num_points_theta, num_points_phi):
    alm = {}

    def integrand_real(theta, phi, l, m):
        ylm = sph_harm(m, l, phi, theta)
        return (b(theta, phi) * np.conj(ylm) * np.sin(theta)).real

    def integrand_imag(theta, phi, l, m):
        ylm = sph_harm(m, l, phi, theta)
        return (b(theta, phi) * np.conj(ylm) * np.sin(theta)).imag

    for l in range(lmax + 1):
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

def compute_total_abs_error(b, num_points_theta, num_points_phi, lmax):
    x = np.linspace(0, 2 * np.pi, num_points_phi)
    y = np.linspace(0, np.pi, num_points_theta)
    y, x = np.meshgrid(y, x)

    alm = adaptive_calcalm(b, lmax, num_points_theta, num_points_phi)
    b_recons = recons(alm, x, y, lmax)
    
    delta = b(y, x) - b_recons
    abs_tot_sum = np.sum(np.abs(delta))
    return abs_tot_sum

num_points_theta_list = range(10, 30, 10)
errors = []

for num_points_theta in num_points_theta_list:
    num_points_phi = 2 * num_points_theta  # Example relation
    lmax = 20
    error = compute_total_abs_error(b, num_points_theta, num_points_phi, lmax)
    errors.append(error)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(num_points_theta_list, errors, marker='o', linestyle='-')
plt.xlabel('Number of Theta Points')
plt.ylabel('Total Absolute Error')
plt.title('Total Absolute Error vs. Number of Theta Points')
plt.grid(True)
plt.show()
