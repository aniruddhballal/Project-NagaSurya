import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, timedelta

# Function to plot alm magnitudes greater than a threshold and save the plot
def plot_and_save(l_values, m_values, alm_magnitudes, map_number, month_year, threshold, max_alm, ax):
    # Filter alm values greater than threshold and m values greater than or equal to 0
    mask = (alm_magnitudes >= threshold) & (m_values >= 0)
    l_values_filtered = l_values[mask]
    m_values_filtered = m_values[mask]
    alm_magnitudes_filtered = alm_magnitudes[mask]
    
    avg_alm = alm_magnitudes_filtered.mean()
    
    scatter = ax.scatter(l_values_filtered, m_values_filtered, c=alm_magnitudes_filtered, cmap='viridis', s=25, alpha=0.75)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Magnitude of alm')
    scatter.set_clim(0.2, 14.7658)  # Set colorbar limits on the scatter plot
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 60)
    ax.set_title(f'{threshold:.4f} <= alm values <= {max_alm:.4f}\nCarrington map {map_number} ({month_year})')
    ax.grid(True)
    
    # Highlight the highest and lowest alm values
    max_index = np.argmax(alm_magnitudes_filtered)
    min_index = np.argmin(alm_magnitudes_filtered)
    min_alm = alm_magnitudes_filtered.min()
    
    ax.scatter(l_values_filtered[max_index], m_values_filtered[max_index], c='red', marker='s', s=50, label='Max alm')
    ax.scatter(l_values_filtered[min_index], m_values_filtered[min_index], c='blue', marker='s', s=50, label='Min alm')
    
    ax.legend()
    
    # Mark the average alm value on the colorbar
    cbar.ax.axhline(avg_alm, color='white', linewidth=2)
    cbar.ax.axhline(max_alm, color='white', linewidth=2)
    cbar.ax.axhline(min_alm, color='white', linewidth=2)

    # Add a new tick for the average value on the colorbar
    cbar.set_ticks([avg_alm])

    # Create folder if it doesn't exist
    folder_name = 'plots/trial'
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the plot with threshold in filename
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
            alm_values = df['alm'].apply(lambda x: complex(x.replace('(', '').replace(')', ''))).values
            
            # Calculate magnitudes of alm
            alm_magnitudes = np.abs(alm_values)
            
            # Set the threshold to the highest 0.5% of alm magnitudes
            threshold = np.percentile(alm_magnitudes, 99.5)
            max_alm = np.percentile(alm_magnitudes, 100)
            
            # Get month-year from Carrington map number
            month_year = get_month_year_from_map_number(map_number)
            
            # Create subplots
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot sin(x) in subplot 1
            x = np.linspace(0, 2 * np.pi, 100)
            axs[0, 0].plot(x, np.sin(x), label='sin(x)')
            axs[0, 0].set_xlabel('x')
            axs[0, 0].set_ylabel('sin(x)')
            axs[0, 0].set_title('Plot of sin(x)')
            axs[0, 0].legend()
            
            # Plot alm values in subplot 2
            plot_and_save(l_values, m_values, alm_magnitudes, map_number, month_year, threshold, max_alm, axs[0, 1])
            
            # Leave subplots 3 and 4 empty
            
            plt.tight_layout()

# Process all CSV files with the calculated threshold
process_all_csv_files()
