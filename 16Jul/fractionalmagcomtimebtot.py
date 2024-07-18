import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Radius of the Sun in kilometers and conversion to cm
radius_sun_cm = 6.96e5 * 1e5  # in cm
surface_area_sun_cm2 = 4 * np.pi * radius_sun_cm**2  # in cm^2


def returncom(l_values, m_values, alm_magnitudes, map_number, month_year, threshold, max_alm):
    # Filter alm values greater than threshold and m values greater than or equal to 0
    mask = (alm_magnitudes >= threshold) & (m_values >= 0)
    alm_magnitudes_filtered = alm_magnitudes[mask]
    num_elements_above_threshold = len(alm_magnitudes_filtered)
    
    # Calculate center of mass components
    total_mass = alm_magnitudes_filtered.sum()
    
    # Calculate the magnitude of the center of mass
    center_of_mass_magnitude = total_mass / num_elements_above_threshold
    
    return center_of_mass_magnitude

# Function to convert Carrington map number to date
def get_date_from_map_number(map_number):
    base_date = datetime(1853, 11, 9)  # Base date for Carrington rotation 1
    rotation_period_days = 27.2753  # Average Carrington rotation period in days
    days_since_base = (int(map_number) - 1) * rotation_period_days
    map_date = base_date + timedelta(days=days_since_base)
    return map_date

# Main function to process all CSV files in the "alm values" folder
def process_all_csv_files():
    alm_folder = 'E:/SheshAditya/alm values'
    total_magnetic_flux_file = 'total_magnetic_flux_values.csv'
    
    center_of_mass_magnitude_list = []
    date_list = []
    magnetic_flux_values = []
    
    # Read total magnetic flux data from CSV
    flux_df = pd.read_csv(total_magnetic_flux_file)
    
    for index, row in flux_df.iterrows():
        map_number = row['CR Map Number']
        total_magnetic_flux = row['Total Magnetic Flux (Maxwell)']
        
        # Load alm data for current map number
        alm_filename = os.path.join(alm_folder, f'values_{map_number}.csv')
        
        if not os.path.isfile(alm_filename):
            print(f"Alm file not found: {alm_filename}")
            continue
        
        df = pd.read_csv(alm_filename)
        l_values = df['l'].values
        m_values = df['m'].values
        alm_values = df['alm'].apply(lambda x: complex(x.strip('()'))).values
        
        # Determine the maximum values of l and m
        l_max = np.max(l_values)
        m_max = np.max(m_values)

        # Initialize alm array with zeros
        alm = np.zeros((l_max + 1, m_max + 1), dtype=complex)

        # Fill alm array with corresponding alm_values
        for l, m, alm_value in zip(l_values, m_values, alm_values):
            alm[l, m] = alm_value

        # Calculate magnitudes of alm
        alm_magnitudes = np.abs(alm_values)
        
        # Set the threshold to the lowest of alm magnitudes
        threshold = np.percentile(alm_magnitudes, 0)
        max_alm = np.percentile(alm_magnitudes, 100)
        
        # Get date from Carrington map number
        map_date = get_date_from_map_number(map_number)
        
        # Calculate center of mass and append to lists
        center_of_mass_magnitude = returncom(l_values, m_values, alm_magnitudes, map_number, map_date, threshold, max_alm)
        
        center_of_mass_magnitude_list.append(center_of_mass_magnitude)
        date_list.append(map_date)
        magnetic_flux_values.append(total_magnetic_flux)
    
    # Convert date_list to numpy array for easier manipulation
    date_list = np.array(date_list)
    
    # Smooth the center of mass data
    sigma = 2  # Adjust sigma as needed for desired smoothing effect
    smoothed_center_of_mass_magnitude = gaussian_filter1d(center_of_mass_magnitude_list, sigma)
    
    #convert all of it into a range of (0,1)
    maxval1 = max(smoothed_center_of_mass_magnitude)
    smoothed_center_of_mass_magnitude = np.array(smoothed_center_of_mass_magnitude) / maxval1
    smoothed_center_of_mass_magnitude = smoothed_center_of_mass_magnitude.tolist()

    maxval2 = max(magnetic_flux_values)
    magnetic_flux_values = np.array(magnetic_flux_values) / maxval2
    magnetic_flux_values = magnetic_flux_values.tolist()

    # Plot center of mass values and magnetic flux over Carrington map numbers
    fig, ax1 = plt.subplots(figsize=(15, 10))

    ax1.plot(date_list, smoothed_center_of_mass_magnitude, label='(0-1) Smoothed COM Mag', marker='o', color='b')
    ax1.set_xlabel('Year', fontsize=20)
    ax1.set_ylabel('(0-1) Smoothed COM Magnitude', fontsize=20, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.xaxis.set_major_locator(mdates.YearLocator(1))  # Major ticks every year
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Year and month
    ax1.grid(True)
    
    # Create a second y-axis to plot magnetic flux
    ax2 = ax1.twinx()
    ax2.plot(date_list, magnetic_flux_values, label='(0-1) Tot Mag Flux', marker='x', color='r')
    ax2.set_ylabel('(0-1) Tot Mag Flux', fontsize=20, color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.suptitle('(0-1) Smoothed COM Mag and (0-1) Tot Flux over Time', fontsize=18)
    fig.legend(loc='upper left', fontsize=15)
    fig.tight_layout()  # Adjust layout to make room for the legend

    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

    pltfolder = 'plots/0-1'
    os.makedirs(pltfolder, exist_ok=True)
    plot_filename = os.path.join(pltfolder, 'avgalm-frac-smooth.png')
    plt.savefig(plot_filename)
    plt.close()

# Process all CSV files with the calculated threshold
process_all_csv_files()