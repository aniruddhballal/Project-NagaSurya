import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sunpy.coordinates.sun import carrington_rotation_time

# Create the output directory if it doesn't exist
output_dir = 'plots_alm_mag'
os.makedirs(output_dir, exist_ok=True)

# Range of Carrington map numbers
start_map = 2096
end_map = 2285

# Function to get month-year from Carrington map number
def get_month_year(carrington_number):
    date = carrington_rotation_time(carrington_number).to_datetime()
    return date.strftime("%B-%Y")

# Loop through Carrington map numbers
for carrington_number in range(start_map, end_map + 1):
    # Define file paths
    input_file = f'alm values/values_{carrington_number}.csv'
    
    # Check if the input file exists
    if os.path.exists(input_file):
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Assuming the CSV has columns 'l', 'm', 'alm'
        l_values = df['l']
        m_values = df['m']
        alm_values = df['alm'].apply(lambda x: complex(x.replace('i', 'j')) if 'i' in x else complex(x))
        
        # Filter out negative m values
        mask = m_values > 0
        l_values = l_values[mask]
        m_values = m_values[mask]
        alm_values = alm_values[mask]
        
        # Calculate the magnitude of alm values
        alm_magnitude = np.abs(alm_values)
        
        # Get the month-year string
        month_year = get_month_year(carrington_number)
        
        # Create a plot
        plt.figure(figsize=(10, 6))
        sc = plt.scatter(l_values, m_values, c=alm_magnitude, cmap='viridis', marker='o', vmin=0, vmax=7)
        cbar = plt.colorbar(sc)
        cbar.set_label('Magnitude of alm')
        plt.xlabel('l')
        plt.ylabel('m')
        plt.title(f'Magnitude of alm values for Carrington Map {carrington_number} ({month_year})')
        plt.grid(True)
        
        # Display magnitudes next to the points with larger font size
        for l, m, mag in zip(l_values, m_values, alm_magnitude):
            plt.text(l, m, f'{mag:.2f}', fontsize=10, ha='right')

        # Save the plot
        output_file = os.path.join(output_dir, f'plot_{carrington_number}.png')
        plt.savefig(output_file)
        plt.close()
        
        print(f'Saved plot for Carrington Map {carrington_number} as {output_file}')
    else:
        print(f'File {input_file} does not exist.')

print('Plotting completed.')