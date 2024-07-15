import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.special import sph_harm
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn

console = Console()

def recons(alm, x, y, lmax, map_number, verbose=True):
    sizex = x.shape
    brecons = np.zeros(sizex, dtype=complex)

    total_calculations = (lmax + 1) * (lmax + 1)
    if verbose:
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn()
        )
        task = progress.add_task("[red]Reconstructing data", total=total_calculations)
        progress.start()

    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            if np.isnan(alm[(l, m)]):
                continue  # Skip if alm is NaN
            ylm = sph_harm(m, l, x, y)
            brecons += alm[(l, m)] * ylm
            if verbose:
                progress.update(task, description=f"[green]Reconstructing {map_number}: ", advance=1)

    if verbose:
        progress.stop()

    return brecons.real

def plot_and_save(l_values, m_values, alm_magnitudes, map_number, month_year, threshold, max_alm, ax):
    # Filter alm values greater than threshold and m values greater than or equal to 0
    mask = (alm_magnitudes >= threshold) & (m_values >= 0)
    l_values_filtered = l_values[mask]
    m_values_filtered = m_values[mask]
    alm_magnitudes_filtered = alm_magnitudes[mask]
    
    avg_alm = alm_magnitudes_filtered.mean()
    
    scatter = ax.scatter(l_values_filtered, m_values_filtered, c=alm_magnitudes_filtered, cmap='viridis', s=50, alpha=0.75)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'Magnitude of alm', fontsize=20)
    scatter.set_clim(0.2, 14.7658)  # Set colorbar limits on the scatter plot
    ax.set_xlabel('l', fontsize=20)
    ax.set_ylabel('m', fontsize=20)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_title(f'ALM Range of |alm|: {threshold:.2f} to {max_alm:.2f}\nCarrington map {map_number} ({month_year})', fontsize=20)
    ax.grid(True)

    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=20)
    cbar.ax.tick_params(labelsize=20)
    
    # Find indices of max and second max alm values
    sorted_indices = np.argsort(alm_magnitudes_filtered)[::-1]  # Sort indices in descending order of alm_magnitudes_filtered
    max_index = sorted_indices[0]
    second_max_index = sorted_indices[1]

    l_max = l_values_filtered[max_index]
    m_max = m_values_filtered[max_index] 
    l_second_max = l_values_filtered[second_max_index]
    m_second_max = m_values_filtered[second_max_index]

    ax.scatter(l_max, m_max, c='red', marker='s', s=100, label=f'Max alm: {alm_magnitudes_filtered[max_index]:.2f}')
    legend = ax.legend(loc='upper left', fontsize=20)
    ax.scatter(l_second_max, m_second_max, c='blue', marker='s', s=100, label=f'Second Max alm: {alm_magnitudes_filtered[second_max_index]:.2f}')
    legend = ax.legend(loc='upper left', fontsize=20)
    
    # Mark the average alm value on the colorbar
    cbar.ax.axhline(avg_alm, color='white', linewidth=2)

    # Set colorbar ticks
    cbar.set_ticks([0.2, avg_alm, 14.7658])
    cbar.set_ticklabels([f'{0.2:.4f}', f'{avg_alm:.4f}', f'{14.7658:.4f}'])

    return l_max, m_max, l_second_max, m_second_max

# Function to convert Carrington map number to month-year string
def get_month_year_from_map_number(map_number):
    base_date = datetime(1853, 11, 9)  # Base date for Carrington rotation 1
    rotation_period_days = 27.2753  # Average Carrington rotation period in days
    days_since_base = (int(map_number) - 1) * rotation_period_days
    map_date = base_date + timedelta(days=days_since_base)
    return map_date.strftime("%B %Y")

def plotty(ax, x, y, bvals, carrington_map_number, lmax):
    font_size = 20
    contnum = 50

    vmin = np.min(bvals)
    vmax = np.max(bvals)
    absmax = max(abs(vmin), abs(vmax))
    clevels = np.linspace(-absmax, absmax, contnum + 1)
    
    # Create custom colormap
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    
    contourf_plot = ax.contourf(x, y, bvals, levels=clevels, cmap=cmap, vmin=-absmax, vmax=absmax)

    cbar = plt.colorbar(contourf_plot, ax=ax)
    cbar.set_label('Gauss (G)', fontsize=20)  # Increase the font size here    
    cbar.set_ticks([-absmax, absmax])
    cbar.set_ticklabels([f'{-absmax:.2f}', f'{absmax:.2f}'], fontsize=font_size)
    ax.set_xlabel(r'$\phi$', fontsize=font_size)
    ax.set_ylabel(r'$\theta$', fontsize=font_size)

    month_year = get_month_year_from_map_number(carrington_map_number)

    name = f'{month_year} - CR: {carrington_map_number}, RECONSTRUCTED WITH lmax: {lmax}'
    ax.set_title(name, fontsize=font_size)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels([r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=font_size)
    ax.set_yticks(np.linspace(0, np.pi, 5))
    ax.set_yticklabels([r'0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.grid(True)

def plotty200(ax, x, y, bvals):
    font_size = 20
    contnum = 50

    vmin = np.min(bvals)
    vmax = np.max(bvals)

    clevels = np.linspace(-200, 200, contnum + 1)
    
    # Create custom colormap
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    
    bvals = np.clip(bvals, -200, 200)

    contourf_plot = ax.contourf(x, y, bvals, levels=clevels, cmap=cmap, vmin=-200, vmax=200)

    cbar = plt.colorbar(contourf_plot, ax=ax)
    cbar.set_label('Gauss (G)', fontsize=20)  # Increase the font size here 
    cbar.set_ticks([-200, 200])
    cbar.set_ticklabels([f'{-200}', f'{200}'], fontsize=font_size)
    ax.set_xlabel(r'$\phi$', fontsize=font_size)
    ax.set_ylabel(r'$\theta$', fontsize=font_size)

    name = f'saturated to +/-200 (og range: {vmin:.2f} to {vmax:.2f})'
    
    ax.set_title(name, fontsize=font_size)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels([r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=font_size)
    ax.set_yticks(np.linspace(0, np.pi, 5))
    ax.set_yticklabels([r'0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.grid(True)

# Define spherical harmonics plotting function
def plot_atomic(ax, l, m, colormap):
    theta = np.linspace(0, np.pi, num=360)
    phi = np.linspace(0, 2 * np.pi, num=720)
    theta, phi = np.meshgrid(theta, phi)
    
    Y_lm = sph_harm(m, l, phi, theta).real
    
    X = np.abs(Y_lm) * np.sin(theta) * np.cos(phi)
    Y = np.abs(Y_lm) * np.sin(theta) * np.sin(phi)
    Z = np.abs(Y_lm) * np.cos(theta)
    
    norm = plt.Normalize(Y_lm.min(), Y_lm.max())
    ax.plot_surface(X, Y, Z, facecolors=plt.get_cmap(colormap)(norm(Y_lm)), edgecolor='none')
        
    # Remove the grid and background
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.axis('off')

# Define spherical harmonics plotting function
def plot_sphere(ax, l, m, colormap='coolwarm'):
    # Create a meshgrid for theta and phi
    theta = np.linspace(0, np.pi, 360)
    phi = np.linspace(0, 2 * np.pi, 720)
    theta, phi = np.meshgrid(theta, phi)
    
    # Calculate the spherical harmonics
    Y_lm = sph_harm(m, l, phi, theta)
    
    # Calculate the real part of the spherical harmonics
    Y_lm_real = np.real(Y_lm)
    
    # Convert spherical coordinates to Cartesian coordinates for plotting
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # Normalize Y_lm_real for color mapping
    norm = plt.Normalize(Y_lm_real.min(), Y_lm_real.max())
    colors = plt.get_cmap(colormap)(norm(Y_lm_real))
    
    # Plot the spherical harmonics on the surface of a sphere
    surface = ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, antialiased=False, shade=False)
    
    # Add a color bar which maps values to colors
    mappable = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    mappable.set_array(Y_lm_real)
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set aspect ratio
    ax.set_box_aspect([1, 1, 1])  # aspect ratio is 1:1:1
    
    # Set plot title
    ax.set_title(r'$Y_{%d}^{%d}$' % (l, m), fontsize = 24, loc = 'left', pad = 15, x = 0.05)
    
    # Remove the grid and background
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.axis('off')

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
            
            # Determine the maximum values of l and m
            l_max = np.max(l_values)
            mmax = np.max(m_values)

            # Initialize alm array with zeros
            alm = np.zeros((l_max + 1, mmax + 1), dtype=complex)

            # Fill alm array with corresponding alm_values
            for l, m, alm_value in zip(l_values, m_values, alm_values):
                alm[l, m] = alm_value

            # Calculate magnitudes of alm
            alm_magnitudes = np.abs(alm_values)
            
            # Set the threshold to the highest 0.5% of alm magnitudes
            threshold = np.percentile(alm_magnitudes, 99.5)
            max_alm = np.percentile(alm_magnitudes, 100)
            
            # Get month-year from Carrington map number
            month_year = get_month_year_from_map_number(map_number)
            
            # Update FITS file path based on the Carrington map number
            fits_file = f'C:/Users/aniru/pyproj/my_env1/we goin solar/12Jul/fits_files/hmi.Synoptic_Mr_small.{map_number}.fits'

            # Open the FITS file and access the data
            with fits.open(fits_file) as hdul:
                b = hdul[0].data

            # Ensure there are no NaN or infinite values in the data
            b = np.nan_to_num(b, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

            # Apply Gaussian smoothing
            sig = 2
            b = gaussian_filter(b, sigma=sig)  # Adjust sigma as needed for smoothing

            # Prepare theta and phi arrays
            num_points_theta = b.shape[0]
            num_points_phi = b.shape[1]

            y = np.linspace(0, np.pi, num_points_theta)
            x = np.linspace(0, 2 * np.pi, num_points_phi)
            y, x = np.meshgrid(y, x, indexing='ij')
                        
            # Create subplots
            fig, axs = plt.subplots(2, 2, figsize=(30, 15))

            # Plot alm values in subplot 2
            lmax, mmax, l2max, m2max = plot_and_save(l_values, m_values, alm_magnitudes, map_number, month_year, threshold, max_alm, axs[0, 1])
            
            ls = [lmax, l2max]
            ms = [mmax, m2max]

            maxlmax = max(lmax, l2max)

            b_reconstructed = recons(alm, x, y, maxlmax, map_number, verbose=True)  # Pass verbose=True to show progress bar
            print(f"Done reconstructing Carrington map number {map_number}")

            # Plotting plots 1, 2, and 3
            plotty(axs[0, 0], x, y, b_reconstructed, map_number, maxlmax)
            plotty200(axs[1, 0], x, y, b)

            # Plotting in the 4th subplot
            ax4 = fig.add_subplot(224, projection='3d')

            # Plot the first ylm on the left side
            ax_left = fig.add_axes([0.415, 0.1, 0.3, 0.3], projection='3d')  # Adjust the left subplot size and position
            plot_atomic(ax_left, ls[0], ms[0], 'coolwarm_r')

            # Plot the second ylm on the right side
            ax_right = fig.add_axes([0.54, 0.1, 0.3, 0.3], projection='3d')  # Adjust the right subplot size and position
            plot_sphere(ax_right, ls[0], ms[0], 'coolwarm_r')

            # Plot the first ylm on the left side
            ax_left = fig.add_axes([0.665, 0.1, 0.3, 0.3], projection='3d')  # Adjust the left subplot size and position
            plot_atomic(ax_left, ls[1], ms[1], 'coolwarm_r')

            # Plot the second ylm on the right side
            ax_right = fig.add_axes([0.79, 0.1, 0.3, 0.3], projection='3d')  # Adjust the right subplot size and position
            plot_sphere(ax_right, ls[1], ms[1], 'coolwarm_r')


            # Adjust subplot properties
            ax4.axis('off')  # Turn off the main subplot's axis

            # Remove previous empty subplot
            fig.delaxes(axs[1, 1])

            plt.tight_layout()

            # Save the figure
            pltfolder = 'pwbetter plots'
            os.makedirs(pltfolder, exist_ok=True)
            plot_filename = os.path.join(pltfolder, f'plot_{map_number}.png')
            plt.savefig(plot_filename)
            plt.close()

# Process all CSV files with the calculated threshold
process_all_csv_files()
plt.close()