import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime, timedelta, timezone
import winsound
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.console import Console
from astropy.io import fits
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.integrate import simpson
import shutil
from sunpy.coordinates.sun import carrington_rotation_time

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

    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            if (l, m) not in alm:
                theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
                integrand_values = integrand(theta_grid, phi_grid, l, m)
                real_result = simpson(simpson(integrand_values.real, dx=dphi, axis=1), dx=dtheta, axis=0)
                imag_result = simpson(simpson(integrand_values.imag, dx=dphi, axis=1), dx=dtheta, axis=0)
                alm[(l, m)] = real_result + 1j * imag_result

                # Write the current state of alm to CSV
                write_alm_to_csv(values_filename, alm)
            if verbose:
                progress.update(task, advance=1)

    if verbose:
        progress.stop()
    return alm

def recons(alm, x, y, verbose=True):
    sizex = x.shape
    brecons = np.zeros(sizex, dtype=complex)

    total_calculations = (5 + 1) * (5 + 1)
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

    for l in range(5 + 1):
        for m in range(-l, l + 1):
            if np.isnan(alm[(l, m)]):
                continue
            ylm = sph_harm(m, l, x, y)
            brecons += alm[(l, m)] * ylm
            if verbose:
                progress.update(task, description=f"[green]Reconstructing: ", advance=1)

    if verbose:
        progress.stop()

    return brecons.real

def plotty(ax, x, y, bvals, carrington_map_number):
    font_size = 12
    contnum = 50

    vmin = np.min(bvals)
    vmax = np.max(bvals)
    absmax = 15
    clevels = np.linspace(-absmax, absmax, contnum + 1)
    
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    
    bvals = np.clip(bvals, -15, 15)

    contourf_plot = ax.contourf(x, y, bvals, levels=clevels, cmap=cmap, vmin=-absmax, vmax=absmax)

    cbar = plt.colorbar(contourf_plot, ax=ax, label='Gauss (G)')
    cbar.set_ticks([-absmax, absmax])
    cbar.set_ticklabels([f'{-absmax:.2f}', f'{absmax:.2f}'], fontsize=font_size)
    ax.set_xlabel(r'$\phi$', fontsize=font_size)
    ax.set_ylabel(r'$\theta$', fontsize=font_size)

    dt_obj = datetime.strptime(str(carrington_rotation_time(carrington_map_number)), '%Y-%m-%d %H:%M:%S.%f')
    month_year = dt_obj.strftime('%B %Y')

    name = f'large scale str ({vmax:.2f} to {vmin:.2f}) - CR map: {carrington_map_number} {month_year}'
    ax.set_title(name, fontsize=font_size)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels([r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=font_size)
    ax.set_yticks(np.linspace(0, np.pi, 5))
    ax.set_yticklabels([r'0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.grid(True)

def process_carrington_map(carrington_map_number, lmax):
    fits_file = f'E:/my_env1/we goin solar/03Jul/motion/fits_files/hmi.Synoptic_Mr_small.{carrington_map_number}.fits'

    with fits.open(fits_file) as hdul:
        b = hdul[0].data

    b = np.nan_to_num(b, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

    sig = 2
    b = gaussian_filter(b, sigma=sig)

    num_points_theta = b.shape[0]
    num_points_phi = b.shape[1]

    y = np.linspace(0, np.pi, num_points_theta)
    x = np.linspace(0, 2 * np.pi, num_points_phi)
    y, x = np.meshgrid(y, x, indexing='ij')

    alm = calcalm(lambda theta, phi: b[np.floor(theta * (num_points_theta - 1) / np.pi).astype(int),
                                       np.floor(phi * (num_points_phi - 1) / (2 * np.pi)).astype(int)], 
                  lmax, num_points_theta, num_points_phi, carrington_map_number)
    
    b_reconstructed = recons(alm, x, y, verbose=True)

    fig, ax = plt.subplots(figsize=(12, 6)) #aspect tatio
    plotty(ax, x, y, b_reconstructed, carrington_map_number) 

    plots_folder = "plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
    plot_filename = os.path.join(plots_folder, f'plot_{carrington_map_number}.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()

    print(f"Saved the plot: {plot_filename}")

    beepbeep(1000,600)

def beepbeep(f, d):
    winsound.Beep(f, d)

lmax = 85
method = "Simpson's 1/3 Rule"

print("\n------------------------------------------------------------------------------------------------------------------------")
print("Imported all necessary libraries, compiled successfully.")
print(f"lmax\t\t\t:\t{lmax}")
print("Method of integration\t:\t" + method)
input("Press Enter to confirm these values and start the process:")

beepbeep(500, 1300)

fits_folder = 'E:/my_env1/we goin solar/03Jul/motion/fits_files'
fits_files = os.listdir(fits_folder)

done_folder = os.path.join(fits_folder, 'done')
if not os.path.exists(done_folder):
    os.makedirs(done_folder)

for fits_filename in fits_files:
    if fits_filename.startswith('hmi.Synoptic_Mr_small.') and fits_filename.endswith('.fits'):
        carrington_map_number = int(fits_filename.split('.')[2])

        fits_file = os.path.join(fits_folder, fits_filename)

        try:
            with fits.open(fits_file) as hdul:
                b = hdul[0].data

            b = np.nan_to_num(b, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

            process_carrington_map(carrington_map_number, lmax)

            shutil.move(fits_file, os.path.join(done_folder, fits_filename))

        except Exception as e:
            print(f"Error processing Carrington map number {carrington_map_number}: {e}")
            continue