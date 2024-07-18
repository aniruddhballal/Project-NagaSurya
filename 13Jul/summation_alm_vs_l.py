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
    m_values = df['m'].values
    alm_values = df['alm'].apply(lambda x: complex(x.strip('()'))).values
    
   
    # Calculate magnitudes of alm
    df['alm_magnitude'] = np.abs(alm_values)

    avg_magnitudes = []

    for l_value, group in df.groupby('l'):
        # Take the last 7 values
        last_7_magnitudes = group['alm_magnitude'].tail(5)
        avg_magnitude = last_7_magnitudes.mean()
        avg_magnitudes.append((l_value, avg_magnitude))
    
    # Convert to DataFrame for easier plotting
    avg_magnitudes_df = pd.DataFrame(avg_magnitudes, columns=['l', 'avg_magnitude'])

    # Get month-year from Carrington map number
    month_year = get_month_year_from_map_number(map_number)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(avg_magnitudes_df['l'], avg_magnitudes_df['avg_magnitude'], marker='o')
    plt.xlabel('l value')
    plt.ylabel('Average Magnitude of Last 5 alm values')
    plt.title(f'Carrington map {map_number} ({month_year})')
    plt.ylim(0, 9)
    plt.grid(True)
    
    # Save the plot
    output_folder = 'summation alm plots - abs(l-m) - 5'
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