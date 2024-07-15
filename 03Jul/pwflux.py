import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sunpy.coordinates.sun import carrington_rotation_time

# Radius of the Sun in kilometers and conversion to cm
radius_sun_cm = 6.96e5 * 1e5  # in cm
surface_area_sun_cm2 = 4 * np.pi * radius_sun_cm**2  # in cm^2

# Directory containing the CSV files
directory = "C:/Users/aniru/pyproj/my_env1/we goin solar/03Jul/flux/values/ungaussian"
# directory = "C:/Users/aniru/pyproj/my_env1/we goin solar/03Jul/flux/values/pre-gaussian"

# Lists to store results
dates = []
flux_values = []
northern_polar_flux_values = []
southern_polar_flux_values = []
northern_hemisphere_avg_values = []
southern_hemisphere_avg_values = []

# Process each CSV file in the directory
for filename in os.listdir(directory):
    if filename.startswith("flux_") and filename.endswith(".csv"):
        carrington_number = int(filename.split('_')[1].split('.')[0])
        filepath = os.path.join(directory, filename)
        
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Extract values from the CSV file
        b_avg = df.iloc[0, 0]
        northern_hemisphere_avg = df.iloc[0, 1]
        southern_hemisphere_avg = df.iloc[0, 2]
        northern_polar_flux = df.iloc[0, 3]
        southern_polar_flux = df.iloc[0, 4]
        
        # Calculate the total magnetic flux in Weber
        total_magnetic_flux_mx = b_avg * surface_area_sun_cm2
        total_magnetic_flux_wb = total_magnetic_flux_mx / 1e8  # in Weber
        
        # Append the date and flux values
        dt_obj = datetime.strptime(str(carrington_rotation_time(carrington_number)), '%Y-%m-%d %H:%M:%S.%f')
        date = dt_obj.strftime('%B %Y')
        
        dates.append(date)
        flux_values.append(total_magnetic_flux_wb)
        northern_polar_flux_values.append(northern_polar_flux)
        southern_polar_flux_values.append(southern_polar_flux)
        northern_hemisphere_avg_values.append(northern_hemisphere_avg)
        southern_hemisphere_avg_values.append(southern_hemisphere_avg)

# Create DataFrames for plotting
plot_df = pd.DataFrame({
    "Date": dates,
    "Total Magnetic Flux (Wb)": flux_values
})
polar_flux_df = pd.DataFrame({
    "Date": dates,
    "Northern Polar Flux (Wb)": northern_polar_flux_values,
    "Southern Polar Flux (Wb)": southern_polar_flux_values
})
hemispherical_flux_df = pd.DataFrame({
    "Date": dates,
    "Northern Hemisphere Avg (Wb)": northern_hemisphere_avg_values,
    "Southern Hemisphere Avg (Wb)": southern_hemisphere_avg_values
})

# Sort the DataFrames by date
plot_df["Date"] = pd.to_datetime(plot_df["Date"], format='%B %Y')
plot_df.sort_values("Date", inplace=True)
polar_flux_df["Date"] = pd.to_datetime(polar_flux_df["Date"], format='%B %Y')
polar_flux_df.sort_values("Date", inplace=True)
hemispherical_flux_df["Date"] = pd.to_datetime(hemispherical_flux_df["Date"], format='%B %Y')
hemispherical_flux_df.sort_values("Date", inplace=True)

# Create output directories if they don't exist
overall_output_dir = "plots/overall"
if not os.path.exists(overall_output_dir):
    os.makedirs(overall_output_dir)
polar_output_dir = "plots/polar"
if not os.path.exists(polar_output_dir):
    os.makedirs(polar_output_dir)
hemispherical_output_dir = "plots/hemispherical"
if not os.path.exists(hemispherical_output_dir):
    os.makedirs(hemispherical_output_dir)

# Plot Total Magnetic Flux vs Time
plt.figure(figsize=(14, 7))
plt.plot(plot_df["Date"], plot_df["Total Magnetic Flux (Wb)"], marker='o', linestyle='-', color='b')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Total Magnetic Flux (Wb)', fontsize=14)
plt.title('Magnitude of the Solar Flux vs Time', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plot_path = os.path.join(overall_output_dir, "solar_flux_vs_time_ungaussian.png")
plt.savefig(plot_path, dpi=300)

# Plot Polar Flux vs Time
plt.figure(figsize=(14, 7))
plt.plot(polar_flux_df["Date"], polar_flux_df["Northern Polar Flux (Wb)"], marker='o', linestyle='-', color='r', label='Northern Polar Flux')
plt.plot(polar_flux_df["Date"], polar_flux_df["Southern Polar Flux (Wb)"], marker='o', linestyle='-', color='g', label='Southern Polar Flux')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Polar Flux (Wb)', fontsize=14)
plt.title('Polar Flux vs Time', fontsize=16)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
polar_plot_path = os.path.join(polar_output_dir, "polar_flux_vs_time_ungaussian.png")
plt.savefig(polar_plot_path, dpi=300)

# Plot Hemispherical Flux vs Time
plt.figure(figsize=(14, 7))
plt.plot(hemispherical_flux_df["Date"], hemispherical_flux_df["Northern Hemisphere Avg (Wb)"], marker='o', linestyle='-', color='r', label='Northern Hemisphere Avg')
plt.plot(hemispherical_flux_df["Date"], hemispherical_flux_df["Southern Hemisphere Avg (Wb)"], marker='o', linestyle='-', color='g', label='Southern Hemisphere Avg')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Hemispherical Flux (Wb)', fontsize=14)
plt.title('Hemispherical Flux vs Time', fontsize=16)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
hemispherical_plot_path = os.path.join(hemispherical_output_dir, "hemispherical_flux_vs_time_ungaussian.png")
plt.savefig(hemispherical_plot_path, dpi=300)