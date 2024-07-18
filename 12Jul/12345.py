import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.time import Time
import matplotlib.dates as mdates
from sunpy.coordinates.sun import carrington_rotation_time

# Radius of the Sun in kilometers and conversion to cm
radius_sun_cm = 6.96e5 * 1e5  # in cm
surface_area_sun_cm2 = 4 * np.pi * radius_sun_cm**2  # in cm^2

# Function to get month-year from Carrington map number
def get_month_year(carrington_number):
    date = carrington_rotation_time(carrington_number).to_datetime()
    return date.strftime("%B-%Y")

# Define the Carrington map number range
carrington_numbers = range(2096, 2286)

# Define the l and m values to be plotted
l_values = [1, 2, 3, 4, 5]
m_value = 0

# Prepare a dictionary to store the magnitudes of alm for each (l, m) combination
alm_magnitudes = {l: [] for l in l_values}
time_axis = []
magnetic_flux_values = []

# Loop through the Carrington map numbers and read the corresponding CSV files
for carrington_number in carrington_numbers:
    alm_file_path = f"alm values/values_{carrington_number}.csv"
    flux_file_path = f"flux values/flux_{carrington_number}.csv"
    
    if os.path.exists(alm_file_path) and os.path.exists(flux_file_path):
        alm_data = pd.read_csv(alm_file_path)
        flux_data = pd.read_csv(flux_file_path)
        
        # Convert alm column to complex numbers
        alm_data['alm'] = alm_data['alm'].apply(lambda x: complex(x.replace('(','').replace(')','')))
        
        time_axis.append(get_month_year(carrington_number))
        b_avg = flux_data['B Avg'].values[0]
        
        # Calculate the total magnetic flux in Weber
        total_magnetic_flux_mx = b_avg * surface_area_sun_cm2
        total_magnetic_flux_wb = total_magnetic_flux_mx / 1e8  # in Weber
        magnetic_flux_values.append(total_magnetic_flux_wb)
        
        for l in l_values:
            alm = alm_data[(alm_data['l'] == l) & (alm_data['m'] == m_value)]
            if not alm.empty:
                # Calculate magnitude of complex number
                alm_magnitudes[l].append(np.abs(alm['alm'].values[0]))
            else:
                alm_magnitudes[l].append(None)
    else:
        print(f"File {alm_file_path} or {flux_file_path} not found. Skipping.")

# Convert time_axis to datetime objects
time_axis = [carrington_rotation_time(cn).to_datetime() for cn in carrington_numbers if os.path.exists(f"alm values/values_{cn}.csv")]

# Plot the data
fig, axs = plt.subplots(len(l_values), 1, sharex=True, figsize=(12, 15))
fig.suptitle('alm Magnitudes Over Time')

for i, l in enumerate(l_values):
    ax1 = axs[i]
    ax2 = ax1.twinx()
    
    ax1.plot(time_axis, alm_magnitudes[l], label=f'l={l}, m={m_value}', color='tab:blue')
    ax1.set_ylabel(f'Magnitude of alm')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    ax2.plot(time_axis, magnetic_flux_values, label='Magnetic Flux (Wb)', color='tab:red')
    ax2.set_ylabel('Magnetic Flux (Wb)')
    ax2.legend(loc='upper right')

# Set annual ticks
axs[-1].xaxis.set_major_locator(mdates.YearLocator())
axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Rotate x-axis labels for better readability
plt.setp(axs[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')

axs[-1].set_xlabel('Time (Month-Year)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Define the directory to save the plot
output_dir = "12345 plots"

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the plot in the specified directory
output_file = os.path.join(output_dir, "plot.png")
plt.savefig(output_file)