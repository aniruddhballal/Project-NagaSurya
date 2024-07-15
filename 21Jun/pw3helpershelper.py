import numpy as np
from scipy.special import sph_harm
import csv

def b(y, x):
    return np.sin(y)
    #return np.sin(x)
    #return np.cos(y)
    #return np.cos(x)

def write_alm_to_csv(csv_filename, alm):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['l', 'm', 'alm'])
        for (l, m), value in alm.items():
            writer.writerow([l, m, value])

def calcalm(b, lmax, num_points_theta, num_points_phi):
    alm = {}

    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            x = np.linspace(0, 2 * np.pi, num_points_phi)  # phi
            y = np.linspace(0, np.pi, num_points_theta)    # theta
            y, x = np.meshgrid(y, x)
            ylm = sph_harm(m, l, x, y)
            integrand = b(y, x) * np.conj(ylm) * np.sin(y)
            alm[(l, m)] = np.sum(integrand) * (np.pi / num_points_theta) * (2 * np.pi / num_points_phi)

    return alm

# Interactive input
num_points_theta = int(input("Enter the number of points for theta: "))
num_points_phi = int(input("Enter the number of points for phi: "))
lmax = int(input("Enter the value of lmax: "))

# Calculate alm values
alm = calcalm(b, lmax, num_points_theta, num_points_phi)

# Save alm values to CSV file
csv_filename = f'alm_values_theta_{num_points_theta}_phi_{num_points_phi}_lmax_{lmax}.csv'
write_alm_to_csv(csv_filename, alm)

print(f'alm values saved to {csv_filename}')
