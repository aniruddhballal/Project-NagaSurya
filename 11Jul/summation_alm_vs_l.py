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

# Function to process each CSV file and create the plot
def process_and_plot_csv(map_number):
    folder_name = 'alm values'
    csv_filename = os.path.join(folder_name, f'values_{map_number}.csv')
    
    if not os.path.exists(csv_filename):
        print(f"File {csv_filename} does not exist.")
        return
    
    # Load CSV file
    df = pd.read_csv(csv_filename)
    
    # Extract l, m, and alm values
    l_values = df['l'].values
    alm_values = df['alm'].apply(lambda x: complex(x.strip('()'))).values
    
    # Calculate magnitudes of alm
    alm_magnitudes = np.abs(alm_values)

    # Create an array to store the summation results
    summation_values = np.zeros(86)
    
    # Calculate summation of alm values for each l and divide by (2l + 1)
    for l in range(86):
        alm_sum = np.sum(alm_magnitudes[l_values == l])
        summation_values[l] = alm_sum / (2 * l + 1)
    
    # Get month-year from Carrington map number
    month_year = get_month_year_from_map_number(map_number)
    
    # Plot the summation values
    plt.figure(figsize=(10, 5))
    plt.plot(range(86), summation_values, marker='o')
    plt.xlabel('l')
    plt.ylabel('Summation of alm values / (2l + 1)')
    plt.title(f'Summation of alm values for Carrington map {map_number} ({month_year})')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    output_folder = 'summation alm plots'
    os.makedirs(output_folder, exist_ok=True)
    output_filename = os.path.join(output_folder, f'plot_{map_number}.png')
    plt.savefig(output_filename)
    plt.close()

# Main function to process all CSV files in the "alm values" folder
def process_all_csv_files():
    folder_name = 'alm values'
    
    for csv_file in os.listdir(folder_name):
        if csv_file.endswith('.csv'):
            map_number = csv_file.split('_')[1].split('.')[0]
            process_and_plot_csv(map_number)

# Process all CSV files
process_all_csv_files()