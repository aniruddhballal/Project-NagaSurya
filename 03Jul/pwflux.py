import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from datetime import datetime
from sunpy.coordinates.sun import carrington_rotation_time
import pandas as pd
import csv

R_SUN_CM = 6.96e10
PIXEL_LATITUDE_COUNT = 360
PIXEL_LONGITUDE_COUNT = 720
FITS_DIRECTORY = 'C:/Users/aniru/pyproj/my_env1/we goin solar/03Jul/gecko/gecko_fits/'
OUTPUT_DIRECTORY = 'plots/raw magnetic field plots'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

carrington_numbers = []
total_magnetic_fluxes = []
dates = []

def crtodate(carrington_number):
    rotation_time = carrington_rotation_time(carrington_number)
    return rotation_time.to_datetime()

for carrington_number in range(2096, 2286):
    fits_file_path = os.path.join(FITS_DIRECTORY, f'hmi.Synoptic_Mr_small.{carrington_number}.fits')
    
    if not os.path.exists(fits_file_path):
        print(f"File {fits_file_path} does not exist. Skipping...")
        continue
    
    try:
        fits_data = fits.open(fits_file_path)
        magnetic_field_data = fits_data[0].data
    except Exception as e:
        print(f"Error reading {fits_file_path}: {e}")
        continue

    # Take the absolute value of the magnetic field data
    magnetic_field_data = np.abs(np.nan_to_num(magnetic_field_data, nan=0.0, posinf=0.0, neginf=0.0))

    # Area represented by each pixel calc
    latitude_centers = np.linspace(-np.pi / 2, np.pi / 2, PIXEL_LATITUDE_COUNT)
    longitude_centers = np.linspace(0, 2 * np.pi, PIXEL_LONGITUDE_COUNT)
    latitude_diff = np.abs(latitude_centers[1] - latitude_centers[0])
    longitude_diff = np.abs(longitude_centers[1] - longitude_centers[0])

    # Area of each pixel
    pixel_area = R_SUN_CM**2 * np.outer(np.cos(latitude_centers), np.ones(PIXEL_LONGITUDE_COUNT)) * latitude_diff * longitude_diff

    # Gauss to Maxwell by multiplying with pixel area
    magnetic_flux_data_maxwell = magnetic_field_data * pixel_area

    # Sum the magnetic field to get the total magnetic flux
    total_magnetic_flux = np.sum(magnetic_flux_data_maxwell)

    carrington_numbers.append(carrington_number)
    total_magnetic_fluxes.append(total_magnetic_flux)
    
    dt_obj = crtodate(carrington_number)
    date = dt_obj.strftime('%Y-%m')
    dates.append(date)

# Create a DataFrame for plotting
plot_df = pd.DataFrame({
    "CR Map Number": carrington_numbers,
    "Date": dates,
    "Total Magnetic Flux (Maxwell)": total_magnetic_fluxes
})

# Sort the DataFrame by CR Map Number
plot_df.sort_values("CR Map Number", inplace=True)

# Save the DataFrame to CSV
csv_path = os.path.join(OUTPUT_DIRECTORY, 'total_magnetic_flux_values.csv')
plot_df.to_csv(csv_path, index=False)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(plot_df["CR Map Number"], plot_df["Total Magnetic Flux (Maxwell)"], marker='o', linestyle='-', color='b')
plt.xlabel('Carrington Rotation Map Number', fontsize=14)
plt.ylabel('Total Magnetic Flux (Maxwell)', fontsize=14)
plt.title('Total Magnetic Flux over Carrington Rotation Maps', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save the plot
plot_path = os.path.join(OUTPUT_DIRECTORY, 'total_magnetic_flux_variation.png')
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"Plot saved to {plot_path}")
print(f"Magnetic flux values saved to {csv_path}")