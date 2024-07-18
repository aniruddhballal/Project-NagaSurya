import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, timedelta
from scipy.special import sph_harm

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
    max_alm_value = alm_magnitudes_filtered[max_index]
    min_alm_value = alm_magnitudes_filtered[min_index]
    l_max = l_values_filtered[max_index]
    m_max = m_values_filtered[max_index]
    l_min = l_values_filtered[min_index]
    m_min = m_values_filtered[min_index]
    
    ax.scatter(l_max, m_max, c='red', marker='s', s=50, label='Max alm')
    ax.scatter(l_min, m_min, c='blue', marker='s', s=50, label='Min alm')
    
    ax.legend()
    
    # Mark the average alm value on the colorbar
    cbar.ax.axhline(avg_alm, color='white', linewidth=2)
    cbar.ax.axhline(max_alm_value, color='white', linewidth=2)
    cbar.ax.axhline(min_alm_value, color='white', linewidth=2)

    # Add a new tick for the average value on the colorbar
    cbar.set_ticks([avg_alm])
    
    return l_max, m_max

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
            
            # Calculate magnitudes of alm
            alm_magnitudes = np.abs(alm_values)
            
            # Set the threshold to the highest 0.5% of alm magnitudes
            threshold = np.percentile(alm_magnitudes, 99.5)
            max_alm = np.max(alm_magnitudes)
            
            # Get month-year from Carrington map number
            month_year = get_month_year_from_map_number(map_number)
            
            # Create subplots
            fig, axs = plt.subplots(1, 2, figsize=(15, 7))
            
            # Plot alm values in subplot 1
            l_max, m_max = plot_and_save(l_values, m_values, alm_magnitudes, map_number, month_year, threshold, max_alm, axs[0])
            
            # Plot spherical harmonics in subplot 2
            ax = fig.add_subplot(122, projection='3d')
            theta = np.linspace(0, np.pi, num=100)
            phi = np.linspace(0, 2*np.pi, num=100)
            theta, phi = np.meshgrid(theta, phi)
            
            # Calculate spherical harmonics for the maximum l_max and m_max values
            Y_lm = sph_harm(m_max, l_max, phi, theta).real
            
            # Convert spherical to Cartesian coordinates
            X = np.abs(Y_lm) * np.sin(theta) * np.cos(phi)
            Y = np.abs(Y_lm) * np.sin(theta) * np.sin(phi)
            Z = np.abs(Y_lm) * np.cos(theta)
            
            # Plot spherical harmonics
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
            ax.set_title(f'Spherical Harmonics Y_{l_max}^{m_max}', fontsize=16)
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            ax.set_zlabel('Z Axis')
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            plt.tight_layout()
            
            # Save the figure
            plt_folder = 'combo2 plots'
            os.makedirs(plt_folder, exist_ok=True)
            plot_filename = os.path.join(plt_folder, f'ylm_plot_{map_number}_{datetime.now().strftime("%Y%m%d%H%M%S")}.png')
            plt.savefig(plot_filename)
            plt.close()

# Process all CSV files with the calculated threshold
process_all_csv_files()