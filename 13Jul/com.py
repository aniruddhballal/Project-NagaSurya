import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter1d
import matplotlib.dates as mdates

# Global lists to accumulate l and m values
l_l = []
m_m = []
dates = []

# Function to plot all alm points and save the plot
def plotty(l_values, m_values, alm_magnitudes, map_number):
    global l_l, m_m, dates
    
    mask = (m_values >= 0)
    l_values_filtered = l_values[mask]
    m_values_filtered = m_values[mask]
    alm_magnitudes_filtered = alm_magnitudes[mask]
    
    alm_sum = np.sum(alm_magnitudes_filtered)
    l_mean = np.sum(alm_magnitudes_filtered * l_values_filtered) / alm_sum
    m_mean = np.sum(alm_magnitudes_filtered * m_values_filtered) / alm_sum
    
    # Print the map number, alm sum, and mean (l, m)
    print(f"{map_number}, sum: {alm_sum}, (l,m) = ({l_mean}, {m_mean})")
    
    # Accumulate l, m, and dates
    l_l.append(l_mean)
    m_m.append(m_mean)
    dates.append(get_date_from_map_number(map_number))

# Function to convert Carrington map number to datetime
def get_date_from_map_number(map_number):
    base_date = datetime(1853, 11, 9)  # Base date for Carrington rotation 1
    rotation_period_days = 27.2753  # Average Carrington rotation period in days
    days_since_base = (int(map_number) - 1) * rotation_period_days
    map_date = base_date + timedelta(days=days_since_base)
    return map_date

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
                        
            # Plot and save the plot with all alm points
            plotty(l_values, m_values, alm_magnitudes, map_number)

# Process all CSV files
process_all_csv_files()

# Apply Gaussian smoothing to l_l and m_m
sigma = 1  # Standard deviation for Gaussian kernel
l_l_smoothed = gaussian_filter1d(l_l, sigma=sigma)
m_m_smoothed = gaussian_filter1d(m_m, sigma=sigma)

# Plot l_l vs dates with connected dots
fig, ax1 = plt.subplots(figsize=(10, 8))

color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('l_l', color=color)
ax1.plot(dates, l_l_smoothed, marker='o', color=color, alpha=0.5, label='l_l')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 50)  # Set y-axis limits for l_l

# Create a second y-axis for m_m
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('m_m', color=color)
ax2.plot(dates, m_m_smoothed, marker='o', color=color, alpha=0.5, label='m_m')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 50)  # Set y-axis limits for m_m

# Format the x-axis to show one tick per year
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Add grid and title
fig.tight_layout()
plt.title('l_l and m_m vs Date')
plt.grid(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show plot
plt.show()

# Optionally, you can save the plot
# plt.savefig('l_l_and_m_m_vs_date.png')