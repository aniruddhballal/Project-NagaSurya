import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter1d

def return_lmax(l_values, m_values, alm_magnitudes, threshold):
    mask = (alm_magnitudes >= threshold) & (m_values >= 0)
    l_values_filtered = l_values[mask]
    alm_magnitudes_filtered = alm_magnitudes[mask]
    max_index = np.argmax(alm_magnitudes_filtered)
    l_max = l_values_filtered[max_index]
    return l_max

# Function to convert Carrington map number to month-year string
def get_month_year_from_map_number(map_number):
    base_date = datetime(1853, 11, 9)  # Base date for Carrington rotation 1
    rotation_period_days = 27.2753  # Average Carrington rotation period in days
    days_since_base = (int(map_number) - 1) * rotation_period_days
    map_date = base_date + timedelta(days=days_since_base)
    return map_date

# Main function to process all CSV files in the "alm values" folder
def process_all_csv_files():
    folder_name = 'alm values'
    lmax_list = []
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
            
            # Calculate magnitudes of alm
            alm_magnitudes = np.abs(alm_values)
            
            # Set the threshold to the highest 0.5% of alm magnitudes
            threshold = np.percentile(alm_magnitudes, 99.5)
            
            # Get l_max
            l_max = return_lmax(l_values, m_values, alm_magnitudes, threshold)
            
            # Get month-year and exact date from Carrington map number
            map_date = get_month_year_from_map_number(map_number)
            
            # Append the l_max and date to the lists
            lmax_list.append(l_max)
            date_list.append(map_date)
    
    # Convert date_list to numpy array for easier manipulation
    date_list = np.array(date_list)
    
    # Apply Gaussian filter to lmax_list
    sigma = 2  # Standard deviation for Gaussian kernel
    lmax_smoothed = gaussian_filter1d(lmax_list, sigma=sigma)
    
    # Plot the variation of l_max over time
    plt.figure(figsize=(12, 6))
    plt.plot(date_list, lmax_list, marker='o', label='Original')
    plt.plot(date_list, lmax_smoothed, marker='', linestyle='-', linewidth=2, label='Smoothed')
    plt.xlabel('Year')
    plt.ylabel('l_max')
    plt.title('Variation of l_max over the years')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    plt.savefig('l_max_vs_time.png')
    plt.show()

# Process all CSV files
process_all_csv_files()