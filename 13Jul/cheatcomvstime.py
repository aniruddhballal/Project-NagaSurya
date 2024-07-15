import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d
from datetime import datetime, timedelta
import matplotlib.dates as mdates

def plot_and_save(l_values, m_values, alm_magnitudes, map_number, month_year, threshold, max_alm):
    # Filter alm values greater than threshold and m values greater than or equal to 0
    mask = (alm_magnitudes >= threshold) & (m_values >= 0)
    l_values_filtered = l_values[mask]
    m_values_filtered = m_values[mask]
    alm_magnitudes_filtered = alm_magnitudes[mask]
    
    # Calculate center of mass
    total_mass = alm_magnitudes_filtered.sum()
    center_of_mass_l = (l_values_filtered * alm_magnitudes_filtered).sum() / total_mass
    center_of_mass_m = (m_values_filtered * alm_magnitudes_filtered).sum() / total_mass
    
    return center_of_mass_l, center_of_mass_m

# Function to convert Carrington map number to date
def get_date_from_map_number(map_number):
    base_date = datetime(1853, 11, 9)  # Base date for Carrington rotation 1
    rotation_period_days = 27.2753  # Average Carrington rotation period in days
    days_since_base = (int(map_number) - 1) * rotation_period_days
    map_date = base_date + timedelta(days=days_since_base)
    return map_date

# Main function to process all CSV files in the "alm values" folder
def process_all_csv_files():
    folder_name = 'alm values'
    
    center_of_mass_l_list = []
    center_of_mass_m_list = []
    date_list = []
    
    for csv_file in os.listdir(folder_name):
        if csv_file.endswith('.csv'):
            map_number = csv_file.split('_')[1].split('.')[0]
            csv_filename = os.path.join(folder_name, csv_file)
            
            # Load CSV file
            df = pd.read_csv(csv_filename)
            
            # Extract l, m, and alm values
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
            
            # Set the threshold to the highest 0.5% of alm magnitudes
            threshold = np.percentile(alm_magnitudes, 99.5)
            max_alm = np.percentile(alm_magnitudes, 100)
            
            # Get date from Carrington map number
            map_date = get_date_from_map_number(map_number)
            
            # Calculate center of mass and append to lists
            center_of_mass_l, center_of_mass_m = plot_and_save(l_values, m_values, alm_magnitudes, map_number, map_date, threshold, max_alm)
            
            center_of_mass_l_list.append(center_of_mass_l)
            center_of_mass_m_list.append(center_of_mass_m)
            date_list.append(map_date)
    
    # Convert date_list to numpy array for easier manipulation
    date_list = np.array(date_list)
    
    # Smooth the center of mass data
    sigma = 5  # Adjust sigma as needed for desired smoothing effect
    smoothed_center_of_mass_l = gaussian_filter1d(center_of_mass_l_list, sigma)
    smoothed_center_of_mass_m = gaussian_filter1d(center_of_mass_m_list, sigma)
    
    # Plot center of mass values over Carrington map numbers
    fig, ax = plt.subplots(figsize=(15, 10))

    ax.plot(date_list, smoothed_center_of_mass_l, label='Smoothed Center of Mass l', marker='o')
    ax.plot(date_list, smoothed_center_of_mass_m, label='Smoothed Center of Mass m', marker='x')

    # Format x-axis as dates
    ax.xaxis.set_major_locator(mdates.YearLocator(1))  # Major ticks every year
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Year and month

    ax.set_xlabel('Year', fontsize=20)
    ax.set_ylabel('Center of Mass', fontsize=20)
    ax.set_title('Smoothed Center of Mass over Time', fontsize=20)
    ax.grid(True)
    ax.legend(loc='upper left', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)

    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

    pltfolder = 'cheatcomvstime plots'
    os.makedirs(pltfolder, exist_ok=True)
    plot_filename = os.path.join(pltfolder, 'center_of_mass_over_time_smoothedsigma5_14Jul.png')
    plt.savefig(plot_filename)
    plt.close()

# Process all CSV files with the calculated threshold
process_all_csv_files()