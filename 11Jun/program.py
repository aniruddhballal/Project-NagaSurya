import numpy as np
import sunpy.map
import matplotlib.pyplot as plt
from scipy.special import sph_harm

# Function to calculate spherical harmonics coefficients
def calculate_alm(data, l_max):
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

# Step 1: Load the FITS file
file_path = "C:/Users/aniru/OneDrive/Desktop/we goin solar/hmi.Synoptic_Mr_small.2110.fits"
syn_map = sunpy.map.Map(file_path)

# Plot the map to visualize
fig = plt.figure(figsize=(12, 5))
ax = plt.subplot(projection=syn_map)
im = syn_map.plot(axes=ax)
ax.coords[0].set_axislabel("Carrington Longitude [deg]")
ax.coords[1].set_axislabel("Latitude [deg]")
ax.coords.grid(color='black', alpha=0.6, linestyle='dotted', linewidth=0.5)
cb = plt.colorbar(im, fraction=0.019, pad=0.1)
cb.set_label(f"Radial magnetic field [{syn_map.unit}]")
ax.set_ylim(bottom=0)
ax.set_title(f"{syn_map.meta['content']},\nCarrington rotation {syn_map.meta['CAR_ROT']}")
plt.show()

# Step 2: Replace NaNs with 0s
data = syn_map.data
data = np.nan_to_num(data, nan=0.0)

# Step 3: Calculate the spherical harmonics coefficients
l_max = 10  # Use smaller l_max for initial testing
alm = calculate_alm(data, l_max)

# Step 4: Save the results to an ASCII file
output_file_path = "alm_values.txt"

with open(output_file_path, "w") as file:
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            a_lm = alm[l, m + l_max]
            file.write(f"{l} {m} {a_lm.real} {a_lm.imag}\n")
