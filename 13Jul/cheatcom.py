import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, timedelta
from rich.console import Console

console = Console()

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
    
    # Find the magnitude at the center of mass
    center_of_mass_magnitude = total_mass / len(alm_magnitudes_filtered)  # Assuming the average magnitude for simplicity
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot the center of mass
    ax.scatter(center_of_mass_l, center_of_mass_m, c='red', marker='x', s=200, label=f'Center of Mass: ({center_of_mass_l:.2f}, {center_of_mass_m:.2f})\nMagnitude: {center_of_mass_magnitude:.2f}')
    ax.set_xlabel('l', fontsize=20)
    ax.set_ylabel('m', fontsize=20)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_title(f'Center of Mass of alm Values\nCarrington map {map_number} ({month_year})', fontsize=20)
    ax.grid(True)

    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Add legend
    legend = ax.legend(loc='upper left', fontsize=20)
    
    pltfolder = 'cheatcom plots'
    os.makedirs(pltfolder, exist_ok=True)
    plot_filename = os.path.join(pltfolder, f'plot_{map_number}.png')
    plt.savefig(plot_filename)
    plt.close()

    return center_of_mass_l, center_of_mass_m

# Function to convert Carrington map number to month-year string
def get_month_year_from_map_number(map_number):
    base_date = datetime(1853, 11, 9)  # Base date for Carrington rotation 1
    rotation_period_days = 27.2753  # Average Carrington rotation period in days
    days_since_base = (int(map_number) - 1) * rotation_period_days
    map_date = base_date + timedelta(days=days_since_base)
    return map_date.strftime("%B %Y")

# Main function to process all CSV files in the "alm values" folder
def process_all_csv_files():
    folder_name = 'alm values'
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
            mmax = np.max(m_values)

            # Initialize alm array with zeros
            alm = np.zeros((l_max + 1, mmax + 1), dtype=complex)

            # Fill alm array with corresponding alm_values
            for l, m, alm_value in zip(l_values, m_values, alm_values):
                alm[l, m] = alm_value

            # Calculate magnitudes of alm
            alm_magnitudes = np.abs(alm_values)
            
            # Set the threshold to the highest 0.5% of alm magnitudes
            threshold = np.percentile(alm_magnitudes, 99.5)
            max_alm = np.percentile(alm_magnitudes, 100)
            
            # Get month-year from Carrington map number
            month_year = get_month_year_from_map_number(map_number)
            
            # Create and save the plot
            plot_and_save(l_values, m_values, alm_magnitudes, map_number, month_year, threshold, max_alm)

# Process all CSV files with the calculated threshold
process_all_csv_files()
plt.close()