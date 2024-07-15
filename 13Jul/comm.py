import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, timedelta

# Function to plot the center of mass of alm magnitudes and save the plot
def plot_and_save_center_of_mass(l_values, m_values, alm_magnitudes, map_number, month_year):
    # Filter m values greater than or equal to 0
    mask = (m_values >= 0)
    l_values_filtered = l_values[mask]
    m_values_filtered = m_values[mask]
    alm_magnitudes_filtered = alm_magnitudes[mask]
    
    # Calculate the center of mass
    total_mass = np.sum(alm_magnitudes_filtered)
    com_l = np.sum(l_values_filtered * alm_magnitudes_filtered) / total_mass
    com_m = np.sum(m_values_filtered * alm_magnitudes_filtered) / total_mass

    # Find the closest alm magnitude to the center of mass coordinates
    distance = np.sqrt((l_values_filtered - com_l)**2 + (m_values_filtered - com_m)**2)
    closest_index = np.argmin(distance)
    com_magnitude = alm_magnitudes_filtered[closest_index]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the center of mass with color based on magnitude
    scatter = ax.scatter(com_l, com_m, c=com_magnitude, cmap='viridis', s=100, label=f'Center of Mass (Magnitude: {com_magnitude:.2f})', vmin=0, vmax=3.5)
    
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_xlim(0, 85)
    ax.set_ylim(0, 85)
    ax.set_title(f'Carrington map {map_number} ({month_year})')
    ax.grid(True)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Magnitude of alm')
    cbar.set_ticks([0, 1, 2, 3, 3.5])  # Add ticks for better readability
    
    # Add legend
    ax.legend()
    
    # Create folder if it doesn't exist
    folder_name = 'com plots'
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the plot with map number in filename
    plot_filename = os.path.join(folder_name, f'plot_{map_number}.png')
    plt.savefig(plot_filename)
    plt.close()

# Function to convert Carrington map number to month-year string
def get_month_year_from_map_number(map_number):
    base_date = datetime(1853, 11, 9)  # Base date for Carrington rotation 1
    rotation_period_days = 27.2753  # Average Carrington rotation period in days
    days_since_base = (int(map_number) - 1) * rotation_period_days
    map_date = base_date + timedelta(days=days_since_base)
    return map_date.strftime("%B %Y")

# Main function to process all CSV files in the "alm values smol" folder
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
            alm_values = df['alm'].apply(lambda x: complex(x.replace('(', '').replace(')', ''))).values
            
            # Calculate magnitudes of alm
            alm_magnitudes = np.abs(alm_values)
                        
            # Get month-year from Carrington map number
            month_year = get_month_year_from_map_number(map_number)
            
            # Plot and save the center of mass plot
            plot_and_save_center_of_mass(l_values, m_values, alm_magnitudes, map_number, month_year)

# Process all CSV files
process_all_csv_files()