import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
import csv
import os
import time
from datetime import datetime, timedelta, timezone
import winsound
from tqdm import tqdm  # Import tqdm for progress bar
from astropy.io import fits
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.integrate import simpson

# IST is UTC + 5:30
IST = timezone(timedelta(hours=5, minutes=30))

def get_folder_paths(lmax, num_points_theta, num_points_phi, carrington_map_number):
    today = datetime.now().strftime("%d%b")
    fits_folder = os.path.join(today, "fits_data")
    folder_path = os.path.join(fits_folder, str(carrington_map_number))
    lmax_folder = os.path.join(folder_path, f"lmax{lmax}")
    theta_phi_folder = os.path.join(lmax_folder, f"theta{num_points_theta}_phi{num_points_phi}")
    os.makedirs(theta_phi_folder, exist_ok=True)
    return theta_phi_folder

def read_alm_from_csv(csv_filename):
    alm = {}
    if os.path.exists(csv_filename):
        with open(csv_filename, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                l = int(row['l'])
                m = int(row['m'])
                alm[(l, m)] = complex(row['alm'])  # string to complex
    return alm

def write_alm_to_csv(csv_filename, alm):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['l', 'm', 'alm'])
        for (l, m), value in alm.items():
            writer.writerow([l, m, value])

def calcalm(b_func, lmax, num_points_theta, num_points_phi, carrington_map_number, verbose=True):
    alm = {}

    folder_path = get_folder_paths(lmax, num_points_theta, num_points_phi, carrington_map_number)
    csv_filename = os.path.join(folder_path, f'simpvalues.csv')
    alm.update(read_alm_from_csv(csv_filename))

    def integrand(theta, phi, l, m):
        ylm = sph_harm(m, l, phi, theta)
        return b_func(theta, phi) * np.conj(ylm) * np.sin(theta)

    total_calculations = (lmax + 1) * (lmax + 1)
    progress_bar = tqdm(total=total_calculations, desc=f"Calculating alm (Carrington map {carrington_map_number})") if verbose else None

    theta = np.linspace(0, np.pi, num_points_theta)
    phi = np.linspace(0, 2 * np.pi, num_points_phi)
    dtheta = np.pi / (num_points_theta - 1)
    dphi = 2 * np.pi / (num_points_phi - 1)

    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            if (l, m) not in alm:
                theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
                integrand_values = integrand(theta_grid, phi_grid, l, m)
                real_result = simpson(simpson(integrand_values.real, dx=dphi, axis=1), dx=dtheta, axis=0)
                imag_result = simpson(simpson(integrand_values.imag, dx=dphi, axis=1), dx=dtheta, axis=0)
                alm[(l, m)] = real_result + 1j * imag_result

                # Write the current state of alm to CSV
                write_alm_to_csv(csv_filename, alm)
            if verbose:
                progress_bar.update(1)

    if verbose:
        progress_bar.close()
    return alm

def recons(alm, x, y, lmax, verbose=True):
    sizex = x.shape
    brecons = np.zeros(sizex, dtype=complex)

    total_calculations = (lmax + 1) * (lmax + 1)
    progress_bar = tqdm(total=total_calculations, desc="Reconstructing data") if verbose else None

    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            if np.isnan(alm[(l, m)]):
                continue  # Skip if alm is NaN
            ylm = sph_harm(m, l, x, y)
            brecons += alm[(l, m)] * ylm
            if verbose:
                progress_bar.update(1)

    if verbose:
        progress_bar.close()

    return brecons.real

def plotty(ax, x, y, bvals, type, lmax, contnum, carrington_map_number, sig):
    vmin = np.min(bvals)
    vmax = np.max(bvals)
    clevels = np.linspace(vmin, vmax, contnum + 1)
    
    # Create custom colormap
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    
    contourf_plot = ax.contourf(x, y, bvals, levels=clevels, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(contourf_plot, ax=ax, label='Value')
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f'{vmin:.2f}', f'{vmax:.2f}'])
    ax.set_xlabel('x or phi')
    ax.set_ylabel('y or theta')
    name = ''
    if type == 1:
        name = f'CR map: {carrington_map_number} with gaussian smoothin (sigma: {sig})'
    elif type == 2:
        name = f'Reconstructed (simpsons) (lmax: {lmax}, contour num: {contnum})'
    elif type == 3:
        name = 'Delta/Error'
    ax.set_title(name)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels([r'0', r'$\pi/2$', r'$2\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.set_yticks(np.linspace(0, np.pi, 5))
    ax.set_yticklabels([r'0', r'$\pi/4$', r'$pi/2$', r'$3\pi/4$', r'$pi$'])

def process_carrington_map(carrington_map_number, lmax, contnum):
    # Update FITS file path based on the Carrington map number
    fits_file = f'C:/Users/aniru/pyproj/my_env1/we goin solar/hmi.Synoptic_Mr_small.{carrington_map_number}.fits'

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

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()

    function_start_time = time.time()
    start_ist = datetime.now(IST)
    start_ist_str = start_ist.strftime("%Y-%m-%d %H:%M:%S %Z")

    print(f"\nProcess started for Carrington map number {carrington_map_number}...\nIST now: {start_ist_str}")

    plotty(axs[0], x, y, b, 1, lmax, contnum, carrington_map_number, sig)

    alm = calcalm(lambda theta, phi: b[np.floor(theta * (num_points_theta - 1) / np.pi).astype(int),
                                       np.floor(phi * (num_points_phi - 1) / (2 * np.pi)).astype(int)], 
                  lmax, num_points_theta, num_points_phi, carrington_map_number)
    
    beepbeep(1000,600)

    b_reconstructed = recons(alm, x, y, lmax, verbose=True)  # Pass verbose=True to show progress bar
    print(f"Done reconstructing for Carrington map number {carrington_map_number}")

    plotty(axs[1], x, y, b_reconstructed, 2, lmax, contnum, carrington_map_number, sig)

    delta = b - b_reconstructed
    plotty(axs[2], x, y, delta, 3, lmax, contnum, carrington_map_number, sig)

    # Calculate and round error metrics
    max_delta = round(np.max(delta), 4)
    min_delta = round(np.min(delta), 4)
    avg_abs_delta = round(np.mean(np.abs(delta)), 4)
    recons_avg = round(np.mean(np.abs(b_reconstructed)), 4)
    b_avg = round(np.mean(np.abs(b)), 4)
    ndiff = round(np.abs(recons_avg - b_avg), 4)

    # Write the error metrics to a CSV file inside the specified folder
    folder_path = get_folder_paths(lmax, num_points_theta, num_points_phi, carrington_map_number)
    error_filename = os.path.join(folder_path, 'simperrors.csv')
    with open(error_filename, 'w', newline='') as btxt:
        writer = csv.writer(btxt)
        writer.writerow(['Max Delta', 'Min Delta', 'Avg Abs Delta', 'B Avg', 'Recons Avg', 'Net Diff'])
        writer.writerow([max_delta, min_delta, avg_abs_delta, b_avg, recons_avg, ndiff])

    axs[3].axis('off')  # Turn off the unused subplot

    function_end_time = time.time()
    function_runtime = function_end_time - function_start_time

    # Time in IST
    end_ist = datetime.now(IST)
    end_ist_str = end_ist.strftime("%Y-%m-%d %H:%M:%S %Z")

    csv_filename = os.path.join(folder_path, f'simpvalues.csv')
    plt.figtext(0.95, 0.05, f"CSV used: {csv_filename}\nB Avg: {b_avg:.4f}\nRecons Avg: {recons_avg}\nNet Diff: {ndiff}\n\nRuntime: {function_runtime:.4f} seconds\nStart Time (IST): {start_ist_str}\nEnd Time (IST): {end_ist_str}", va='bottom', ha='right')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2, wspace=0.3, hspace=0.6)

    # Save the plot to a file inside the specified folder with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(folder_path, f'simpplot_{timestamp}_sigma{sig}.png')
    plt.savefig(plot_filename)

    print("Saved the plot.")
    #plt.show()

    beepbeep(600,600)

def beepbeep(f, d):
    winsound.Beep(f, d)

carrington_map_numbers = [2149, 2110, 2225, 2096]

lmax = 80
contour_num = 50
method = "simpsons 1/3 rule"

print("\n------------------------------------------------------------------------------------------------------------------------")
print("Imported all necessary libraries, compiled successfully.")
print(f"lmax\t\t\t:\t{lmax}")
print("Method of integration\t:\t" + method)
print("Carrington Map Numbers\t:\t" + str(carrington_map_numbers))
input("Press Enter to confirm these values and start the process:")

beepbeep(500, 1300)

for carrington_map_number in carrington_map_numbers:
    process_carrington_map(carrington_map_number, lmax, contour_num)