import numpy as np
import sunpy.map
import matplotlib.pyplot as plt
from scipy.special import sph_harm

# Function to calculate B(theta, phi)
def B(theta, phi):
    return 3 * np.sin(theta)**2 * np.cos(phi)**4

# Function to calculate spherical harmonics coefficients
def calculate_alm(l_max):
    alm_expected = np.zeros((l_max + 1, 2 * l_max + 1), dtype=np.complex128)

    # Calculate expected alm values analytically for the chosen function
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            alm_expected[l, m + l_max] = calculate_expected_alm(l, m)

    return alm_expected

# Function to calculate expected alm analytically for the chosen function
def calculate_expected_alm(l, m):
    # Integration over theta and phi for the chosen function B(theta, phi)
    alm = 0
    for i_theta in range(n_theta):
        for j_phi in range(n_phi):
            alm += B(theta[i_theta], phi[j_phi]) * \
                   sph_harm(m, l, phi[j_phi], theta[i_theta]) * \
                   np.sin(theta[i_theta]) * d_theta * d_phi
    return alm

# Step 1: Define theta and phi grids
n_theta = 100  # Number of points for theta grid
n_phi = 100    # Number of points for phi grid
theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2 * np.pi, n_phi)
d_theta = theta[1] - theta[0]
d_phi = phi[1] - phi[0]

# Step 2: Calculate spherical harmonics coefficients
l_max = 5  # Use smaller l_max for initial testing
alm_calculated = calculate_alm(l_max)

# Step 3: Compare calculated alm values with expected alm values
for l in range(l_max + 1):
    for m in range(-l, l + 1):
        alm_expected = calculate_expected_alm(l, m)
        alm_calculated_value = alm_calculated[l, m + l_max]
        print(f"l={l}, m={m}: Calculated alm={alm_calculated_value}, Expected alm={alm_expected}")

# Step 4: Save the results to an ASCII file
output_file_path = "alm_values_test.txt"

with open(output_file_path, "w") as file:
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            a_lm = alm_calculated[l, m + l_max]
            file.write(f"{l} {m} {a_lm.real} {a_lm.imag}\n")
