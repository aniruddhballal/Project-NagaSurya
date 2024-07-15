import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from datetime import datetime
from sunpy.coordinates.sun import carrington_rotation_time

# Constants
R_SUN_CM = 6.96e10  # Radius of the Sun in cm
PIXEL_LATITUDE_COUNT = 360
PIXEL_LONGITUDE_COUNT = 720
FITS_DIRECTORY = 'C:/Users/aniru/pyproj/my_env1/we goin solar/03Jul/gecko/gecko_fits/'
OUTPUT_DIRECTORY = 'raw magnetic field plots'

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

carrington_numbers = []
total_magnetic_fluxes = []
dates = []

def crtodate(carrington_number):
    # Calculate the date associated with the Carrington number
    rotation_time = carrington_rotation_time(carrington_number)
    rotation_time_str = str(rotation_time)  # Convert Time object to string
    return datetime.strptime(rotation_time_str, '%Y-%m-%d %H:%M:%S.%f')

for carrington_number in range(2096, 2286):
    fits_file_path = os.path.join(FITS_DIRECTORY, f'hmi.Synoptic_Mr_small.{carrington_number}.fits')
    
    if not os.path.exists(fits_file_path):
        print(f"File {fits_file_path} does not exist. Skipping...")
        continue
    
    # Load the FITS file
    fits_data = fits.open(fits_file_path)
    magnetic_field_data = fits_data[0].data  # Assuming the magnetic field data is in the primary HDU

    # Handle invalid values
    magnetic_field_data = np.nan_to_num(magnetic_field_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Calculate the area represented by each pixel
    latitude_centers = np.linspace(-np.pi / 2, np.pi / 2, PIXEL_LATITUDE_COUNT)
    longitude_centers = np.linspace(0, 2 * np.pi, PIXEL_LONGITUDE_COUNT)
    latitude_diff = np.abs(latitude_centers[1] - latitude_centers[0])
    longitude_diff = np.abs(longitude_centers[1] - longitude_centers[0])

    # Compute the area of each pixel
    pixel_area = R_SUN_CM**2 * np.outer(np.cos(latitude_centers), np.ones(PIXEL_LONGITUDE_COUNT)) * latitude_diff * longitude_diff

    # Convert magnetic field data from Gauss to Maxwell by multiplying with pixel area
    magnetic_flux_data_maxwell = magnetic_field_data * pixel_area

    # Sum the magnetic field to get the total magnetic flux
    total_magnetic_flux = np.sum(magnetic_flux_data_maxwell)

    # Store the results
    carrington_numbers.append(carrington_number)
    total_magnetic_fluxes.append(total_magnetic_flux)
    
    # Get date from Carrington number
    dt_obj = crtodate(carrington_number)
    date = dt_obj.strftime('%B %Y')
    dates.append(date)

# Plot the variation of the total magnetic fields with dates
plt.figure(figsize=(10, 6))
plt.plot(dates, total_magnetic_fluxes, marker='o', linestyle='-', color='b')
plt.xlabel('Date')
plt.ylabel('Total Magnetic Flux (Maxwell)')
plt.title('Variation of Total Magnetic Flux over Time')
plt.xticks(np.arange(0, len(dates), step=12), rotation=45)  # Show ticks for each year
plt.grid(True)

# Save the plot
plot_path = os.path.join(OUTPUT_DIRECTORY, 'total_magnetic_flux_variation.png')
plt.tight_layout()
plt.savefig(plot_path)
plt.show()

print(f"Plot saved to {plot_path}")