import os
from astropy.io import fits
import numpy as np
# from scipy.ndimage import gaussian_filter
import csv
from rich.progress import Progress

def process_carrington_map(carrington_map_number):

    fits_file = f'C:/Users/aniru/pyproj/my_env1/we goin solar/03Jul/gecko/gecko_fits/hmi.Synoptic_Mr_small.{carrington_map_number}.fits'
    with fits.open(fits_file) as hdul:
        b = hdul[0].data

    b = np.nan_to_num(b, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

    """
    # Apply Gaussian smoothing
    sig = 2
    b = gaussian_filter(b, sigma=sig)  # Adjust sigma as needed for smoothing
    """
    #un-comment the above code and set the flux_folder value to "values/pre-gaussian" if you want to apply gaussian filter to input data

    b_avg = round(np.mean(np.abs(b)), 4)
    northern_hemisphere_avg = round(np.mean(np.abs(b[b.shape[0]//2:, :])), 4)
    southern_hemisphere_avg = round(np.mean(np.abs(b[:b.shape[0]//2, :])), 4)
    northern_polar_flux = round(np.mean(np.abs(b[-25:, :])), 4)  # theta varies from 160 to 180 degrees
    southern_polar_flux = round(np.mean(np.abs(b[:25, :])), 4)   # theta varies from 0 to 25 degrees

    flux_folder = "values/ungaussian"
    if not os.path.exists(flux_folder):
        os.makedirs(flux_folder)
    error_filename = os.path.join(flux_folder, f'flux_{carrington_map_number}.csv')
    with open(error_filename, 'w', newline='') as btxt:
        writer = csv.writer(btxt)
        writer.writerow(['B Avg', 'Northern Hemisphere Avg', 'Southern Hemisphere Avg', 'Northern Polar Flux', 'Southern Polar Flux'])
        writer.writerow([b_avg, northern_hemisphere_avg, southern_hemisphere_avg, northern_polar_flux, southern_polar_flux])

fits_folder = 'C:/Users/aniru/pyproj/my_env1/we goin solar/03Jul/gecko/gecko_fits'
fits_files = os.listdir(fits_folder)

with Progress() as progress:
    task = progress.add_task("Processing FITS files...", total=len(fits_files))

    for fits_filename in fits_files:
        # Extract the CR map number
        carrington_map_number = int(fits_filename.split('.')[2])

        fits_file = os.path.join(fits_folder, fits_filename)

        try:
            with fits.open(fits_file) as hdul:
                b = hdul[0].data

            b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

            process_carrington_map(carrington_map_number)

        except Exception as e:
            print(f"Error processing Carrington map number {carrington_map_number}: {e}")
            continue
        
        progress.update(task, advance=1)