import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from astropy.io import fits
from datetime import datetime
from sunpy.coordinates.sun import carrington_rotation_time

# Function to extract Carrington map number (xxxx) from file name
def extract_carrington_map_number(filename):
    base = os.path.basename(filename)
    carrington_map_number = os.path.splitext(base)[0].split('.')[2]
    return carrington_map_number

# Function to plot the Carrington map with specified colors using plotty function
def plot_carrington_map(fits_file, carrington_map_number):
    with fits.open(fits_file) as hdul:
        data = hdul[0].data
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Define x, y for longitude and latitude
        x = np.linspace(0, 2 * np.pi, data.shape[1])
        y = np.linspace(0, np.pi, data.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Use plotty function for custom plotting
        plotty(ax, X, Y, data, carrington_map_number)
        ax[3].axis('off')  # Turn off the unused subplot
        
        # Leave other subplots empty
        for i in range(2, 5):
            fig.add_subplot(2, 2, i)
        
        plt.tight_layout()
        plt.show()

# Function plotty with custom colormap and formatting
def plotty(ax, x, y, bvals, carrington_map_number):
    font_size = 12
    contnum = 50

    # Replace NaN values with -9999
    bvals = np.where(np.isnan(bvals), -9999, bvals)

    vmin = np.nanmin(bvals)
    vmax = np.nanmax(bvals)
    absmax = max(abs(vmin), abs(vmax))
    clevels = np.linspace(-absmax, absmax, contnum + 1)
    
    # Create custom colormap
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    
    contourf_plot = ax.contourf(x, y, bvals, levels=clevels, cmap=cmap, vmin=-absmax, vmax=absmax)

    cbar = plt.colorbar(contourf_plot, ax=ax, label='Magnetic Field Strength')
    cbar.set_ticks([-absmax, absmax])
    cbar.set_ticklabels([f'{-absmax:.2f}', f'{absmax:.2f}'], fontsize=font_size)
    ax.set_xlabel(r'Longitude ($\phi$)', fontsize=font_size)
    ax.set_ylabel(r'Latitude ($\theta$)', fontsize=font_size)

    dt_obj = datetime.strptime(str(carrington_rotation_time(carrington_map_number)), '%Y-%m-%d %H:%M:%S.%f')
    month_year = dt_obj.strftime('%B %Y')

    name = f'{month_year} - CR map: {carrington_map_number}'  # Fixed formatting
    ax.set_title(name, fontsize=font_size)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels([r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=font_size)
    ax.set_yticks(np.linspace(0, np.pi, 5))
    ax.set_yticklabels([r'0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.grid(True)

# Paths and filenames
fits_folder = 'fits_files'
alm_folder = 'alm values'

# Example usage
fits_file = os.path.join(fits_folder, 'hmi.Synoptic_Mr_small.2096.fits')  # Replace 'xxxx' with actual Carrington map number
carrington_map_number = extract_carrington_map_number(fits_file)

# Plot Carrington map and alm values
plot_carrington_map(fits_file, carrington_map_number)