import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, timedelta

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
    avg_values = []
    months_years = []
    
    for csv_file in os.listdir(folder_name):
        if csv_file.endswith('.csv'):
            map_number = csv_file.split('_')[1].split('.')[0]
            csv_filename = os.path.join(folder_name, csv_file)
            
            # Load CSV file
            df = pd.read_csv(csv_filename)
            
            # Extract alm values
            alm_values = df['alm'].apply(lambda x: complex(x.replace('(', '').replace(')', ''))).values
            
            # Calculate magnitudes of alm
            alm_magnitudes = np.abs(alm_values)
            
            # Set the threshold to the highest 5% of alm magnitudes
            threshold = np.percentile(alm_magnitudes, 95)
            
            # Filter values between threshold and maximum
            values_above_threshold = alm_magnitudes[alm_magnitudes > threshold]
            
            # Calculate the average of values between threshold and maximum
            avg_value = np.mean(values_above_threshold)
            avg_values.append(avg_value)
            
            # Get month-year from Carrington map number
            month_year = get_month_year_from_map_number(map_number)
            months_years.append(month_year)
                
    # Convert month-year strings to datetime objects
    months_years_dt = [datetime.strptime(m, "%B %Y") for m in months_years]
    
    # Extract years for x-tick labels
    years = [dt.year for dt in months_years_dt]
    
    # Plot month-year vs averaged values
    plt.figure(figsize=(12, 6))
    plt.plot(months_years_dt, avg_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Year')
    plt.ylabel('Average alm Magnitude')
    plt.title('Average alm Magnitude Above Threshold vs. Year')
    plt.xticks(ticks=[datetime(year, 1, 1) for year in sorted(set(years))], labels=sorted(set(years)))
    plt.grid(True)
    
    # Create folder if it doesn't exist
    plot_folder = 'plots'
    os.makedirs(plot_folder, exist_ok=True)
    
    # Save the plot
    plot_filename = os.path.join(plot_folder, 'avg_alm_magnitude_vs_year.png')
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    print(f'Plot saved as {plot_filename}')

# Process all CSV files
process_all_csv_files()