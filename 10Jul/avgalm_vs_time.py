import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter

def get_month_year_from_map_number(map_number):
    base_date = datetime(1853, 11, 9)  # Base date for Carrington rotation 1
    rotation_period_days = 27.2753  # Average Carrington rotation period in days
    days_since_base = (int(map_number) - 1) * rotation_period_days
    map_date = base_date + timedelta(days=days_since_base)
    return map_date.strftime("%B %Y")

def process_all_csv_files():
    folder_name = 'E:/SheshAditya/alm values'
    avg_values = []
    months_years = []
    
    for csv_file in os.listdir(folder_name):
        if csv_file.endswith('.csv'):
            map_number = csv_file.split('_')[1].split('.')[0]
            csv_filename = os.path.join(folder_name, csv_file)
            
            df = pd.read_csv(csv_filename)
            
            alm_values = df['alm'].apply(lambda x: complex(x.replace('(', '').replace(')', ''))).values
            
            alm_magnitudes = np.abs(alm_values)
            
            perc = 99.5
            threshold = np.percentile(alm_magnitudes, perc)
            
            # Filter values between threshold and maximum
            values_above_threshold = alm_magnitudes[alm_magnitudes > threshold]
            
            # Calculate the average of values between threshold and maximum
            avg_value = np.mean(values_above_threshold)
            avg_values.append(avg_value)
            
            month_year = get_month_year_from_map_number(map_number)
            months_years.append(month_year)
                
    # Convert month-year strings to datetime objects
    months_years_dt = [datetime.strptime(m, "%B %Y") for m in months_years]
    
    # Extract years for x-tick labels
    years = [dt.year for dt in months_years_dt]
    
    sig = 2
    avg_values = gaussian_filter(avg_values, sigma = sig)

    # Plot month-year vs averaged values
    plt.figure(figsize=(12, 6))
    plt.plot(months_years_dt, avg_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Year')
    plt.ylabel('Average alm Magnitude')
    plt.title(f'Avg alm Mag (gaussian) Above Threshold vs. Year (top {perc}%ile)')
    plt.xticks(ticks=[datetime(year, 1, 1) for year in sorted(set(years))], labels=sorted(set(years)))
    plt.grid(True)
    
    # Create folder if it doesn't exist
    plot_folder = 'plots'
    os.makedirs(plot_folder, exist_ok=True)
    
    # Save the plot
    plot_filename = os.path.join(plot_folder, f'avg_alm_magnitude_vs_year_{perc}%ile_gaussian.png')
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    print(f'Plot saved as {plot_filename}')

# Process all CSV files
process_all_csv_files()