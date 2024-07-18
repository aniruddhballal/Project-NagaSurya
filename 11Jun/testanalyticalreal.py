import numpy as np
from scipy.special import sph_harm

# Define the dummy function B(theta, phi)
def B(theta, phi):
    return 4 * np.sin(theta)**2 * np.cos(phi)**3

# Function to calculate spherical harmonics coefficients numerically
def calculate_alm(data, l_max, theta, phi):
    n_lat, n_lon = data.shape
    alm = np.zeros((l_max + 1, 2 * l_max + 1), dtype=np.complex128)

    phi_grid, theta_grid = np.meshgrid(phi, theta)

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi_grid, theta_grid)
            integrand = data * Y_lm.conj() * np.sin(theta_grid)
            alm[l, m + l_max] = np.sum(integrand) * (2 * np.pi / n_lon) * (np.pi / n_lat)
            print(f"Calculated a_lm for l={l}, m={m}: {alm[l, m + l_max]}")

    return alm

# Function to calculate expected spherical harmonics coefficients analytically
def expected_alm(l, m):
    if l == 2 and m == 0:
        return 4 * np.sqrt(5 / (4 * np.pi)) * (2 / 3)
    if l == 4 and abs(m) == 3:
        return 4 * np.sqrt(9 / (64 * np.pi)) * (4 / 15) * (1j if m != 0 else 1)
    return 0

# Parameters
l_max = 4
n_lat = 180
n_lon = 360

# Generate theta and phi grids
theta = np.linspace(0, np.pi, n_lat)
phi = np.linspace(0, 2 * np.pi, n_lon)
phi_grid, theta_grid = np.meshgrid(phi, theta)

# Evaluate the dummy function B on the grid
data = B(theta_grid, phi_grid)

# Calculate spherical harmonics coefficients numerically
alm_calculated = calculate_alm(data, l_max, theta, phi)

# Compare calculated and expected coefficients
tolerance = 1e-5  # Set a tolerance level for comparison
for l in range(l_max + 1):
    for m in range(-l, l + 1):
        alm_exp = expected_alm(l, m)
        alm_calc = alm_calculated[l, m + l_max]
        if np.allclose(alm_calc, alm_exp, atol=tolerance):
            print(f"l={l}, m={m}: Calculated and Expected a_lm values match within tolerance.")
        else:
            print(f"l={l}, m={m}: Discrepancy detected! Calculated a_lm={alm_calc}, Expected a_lm={alm_exp}")

# Save the results to an ASCII file
output_file_path = "alm_values_test_analytical_real.txt"
with open(output_file_path, "w") as file:
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            a_lm = alm_calculated[l, m + l_max]
            file.write(f"{l} {m} {a_lm.real} {a_lm.imag}\n")
            print(f"Saved a_lm for l={l}, m={m}: {a_lm.real} {a_lm.imag}")
