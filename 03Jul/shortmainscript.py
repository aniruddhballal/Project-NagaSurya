import numpy as np
from scipy.special import sph_harm
import csv
import os
from datetime import datetime, timedelta, timezone
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.console import Console
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.integrate import simpson

# IST is UTC + 5:30
IST = timezone(timedelta(hours=5, minutes=30))

console = Console()

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

    values_folder = "alm values"
    if not os.path.exists(values_folder):
        os.makedirs(values_folder)
    values_filename = os.path.join(values_folder, f'values_{carrington_map_number}.csv')
    alm.update(read_alm_from_csv(values_filename))

    def integrand(theta, phi, l, m):
        ylm = sph_harm(m, l, phi, theta)
        return b_func(theta, phi) * np.conj(ylm) * np.sin(theta)

    total_calculations = (lmax + 1) * (lmax + 1)

    if verbose:
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn()
        )
        task = progress.add_task(f"[red]Calculating alm (Carrington map {carrington_map_number})", total=total_calculations)
        progress.start()

    theta = np.linspace(0, np.pi, num_points_theta)
    phi = np.linspace(0, 2 * np.pi, num_points_phi)
    dtheta = np.pi / (num_points_theta - 1)
    dphi = 2 * np.pi / (num_points_phi - 1)

    alm_to_write = {}

    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            if (l, m) not in alm:
                theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
                integrand_values = integrand(theta_grid, phi_grid, l, m)
                real_result = simpson(simpson(integrand_values.real, dx=dphi, axis=1), dx=dtheta, axis=0)
                imag_result = simpson(simpson(integrand_values.imag, dx=dphi, axis=1), dx=dtheta, axis=0)
                alm[(l, m)] = real_result + 1j * imag_result

            alm_to_write[(l, m)] = alm[(l, m)]

            if verbose:
                progress.update(task, advance=1)

    # Write all alm values to CSV at once
    write_alm_to_csv(values_filename, alm_to_write)

    if verbose:
        progress.stop()

    return alm

def process_carrington_map(carrington_map_number, lmax):
    # Update FITS file path based on the Carrington map number
    fits_file = f'C:/Users/aniru/pyproj/my_env1/we goin solar/03Jul/fits_files/hmi.Synoptic_Mr_small.{carrington_map_number}.fits'

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

    start_ist = datetime.now(IST)
    start_ist_str = start_ist.strftime("%Y-%m-%d %H:%M:%S %Z")

    print(f"\nProcess (only alm calc) started for Carrington map number {carrington_map_number}...\nIST now: {start_ist_str}")

    alm = calcalm(lambda theta, phi: b[np.floor(theta * (num_points_theta - 1) / np.pi).astype(int),
                                       np.floor(phi * (num_points_phi - 1) / (2 * np.pi)).astype(int)], 
                  lmax, num_points_theta, num_points_phi, carrington_map_number)
    
    end_ist = datetime.now(IST)
    end_ist_str = end_ist.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"Process completed for Carrington map number {carrington_map_number}.\nIST now: {end_ist_str}")

if __name__ == "__main__":

    lmax = 85
    method = "Simpson's 1/3 Rule"

    print("\n------------------------------------------------------------------------------------------------------------------------")
    print("\nImported all necessary libraries, compiled successfully.")
    print(f"lmax\t\t\t:\t{lmax}")
    print("Method of integration\t:\t" + method)
    print("Values are autoconfirmed and process is initialised:")

    with open('crmaps.txt', 'r') as f:
        carrington_map_numbers = f.readlines()

    processed_maps_file = 'processed_maps.csv'
    if os.path.exists(processed_maps_file):
        with open(processed_maps_file, 'r') as f:
            processed_maps = [line.strip() for line in f.readlines()]
    else:
        processed_maps = []

    for line in carrington_map_numbers:
        carrington_map_number = line.strip()
        if carrington_map_number not in processed_maps:
            process_carrington_map(carrington_map_number, lmax)  # adjust lmax as needed

            with open(processed_maps_file, 'a') as f:
                f.write(f"{carrington_map_number}\n")

            break  # Exit after processing one map to allow restart