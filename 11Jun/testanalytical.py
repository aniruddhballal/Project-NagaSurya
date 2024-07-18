import numpy as np
import sunpy.map
import matplotlib.pyplot as plt
from scipy.special import sph_harm

# Function to calculate B(theta, phi)
def B(theta, phi):
    return 3 * np.sin(theta)**2 * np.cos(phi)**4

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

# Function to calculate spherical harmonics coefficients
def calculate_alm(l_max):
    n_lat, n_lon = data.shape
    alm = np.zeros((l_max + 1, 2 * l_max + 1), dtype=np.complex128)
    
    # Generate theta and phi grids
    theta = np.linspace(0, np.pi, n_lat)  # theta from 0 to pi
    phi = np.linspace(0, 2 * np.pi, n_lon)  # phi from 0 to 2*pi
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    
    # Calculate coefficients for each l and m
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi_grid, theta_grid)
            integrand = data * Y_lm.conj() * np.sin(theta_grid)
            alm[l, m + l_max] = np.sum(integrand) * (2 * np.pi / n_lon) * (np.pi / n_lat)
    
    return alm

# Step 1: Define theta and phi grids
n_theta = 100  # Number of points for theta grid
n_phi = 100    # Number of points for phi grid
theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2 * np.pi, n_phi)
d_theta = theta[1] - theta[0]
d_phi = phi[1] - phi[0]

# Step 2: Load the FITS file
file_path = "C:/Users/aniru/OneDrive/Desktop/we goin solar/hmi.Synoptic_Mr_small.2110.fits"
syn_map = sunpy.map.Map(file_path)

# Step 3: Replace NaNs with 0s
data = syn_map.data
data = np.nan_to_num(data, nan=0.0)

# Step 4: Calculate expected alm values analytically
l_max = 1  # Use smaller l_max for initial testing
alm_expected = np.zeros((l_max + 1, 2 * l_max + 1), dtype=np.complex128)
for l in range(l_max + 1):
    for m in range(-l, l + 1):
        alm_expected[l, m + l_max] = calculate_expected_alm(l, m)

# Step 5: Calculate spherical harmonics coefficients
alm_calculated = calculate_alm(l_max)

# Step 6: Compare calculated alm values with expected alm values
for l in range(l_max + 1):
    for m in range(-l, l + 1):
        alm_calculated_value = alm_calculated[l, m + l_max]
        alm_expected_value = alm_expected[l, m + l_max]
        print(f"l={l}, m={m}: Calculated alm={alm_calculated_value}, Expected alm={alm_expected_value}")

# Step 7: Save the results to an ASCII file
output_file_path = "alm_values_test_analytical.txt"
with open(output_file_path, "w") as file:
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            a_lm = alm_calculated[l, m + l_max]
            file.write(f"{l} {m} {a_lm.real} {a_lm.imag}\n")
