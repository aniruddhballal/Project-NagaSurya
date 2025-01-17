import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
import csv
import os
import time
from datetime import datetime, timedelta, timezone
import scipy.integrate
import winsound
from tqdm import tqdm

# IST is UTC + 5:30
IST = timezone(timedelta(hours=5, minutes=30))

def read_function_definitions(file_path):
    functions = {}
    with open(file_path, 'r') as file:
        for line in file:
            lhs, rhs = line.split('=')
            lhs = lhs.strip()
            rhs = rhs.strip()
            functions[lhs] = rhs
    return functions

# Read the function definitions from b.txt
file_path = 'b.txt'
function_definitions = read_function_definitions(file_path)

def b(y, x):
    if b_function_name in function_definitions:
        rhs = function_definitions[b_function_name]
        return eval(rhs)
    else:
        raise ValueError(f"Function {b_function_name} is not defined in {file_path}")

def get_folder_paths(b_function_name, lmax, num_points_theta, num_points_phi):
    today = datetime.now().strftime("%d%b")
    base_folder = today
    func_folder = os.path.join(base_folder, b_function_name)
    lmax_folder = os.path.join(func_folder, f"lmax{lmax}")
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

def adaptive_calcalm(b, lmax, num_points_theta, num_points_phi, b_function_name):
    alm = {}

    folder_path = get_folder_paths(b_function_name, lmax, num_points_theta, num_points_phi)
    csv_filename = os.path.join(folder_path, f'values.csv')
    alm.update(read_alm_from_csv(csv_filename))

    def integrand_real(theta, phi, l, m):
        ylm = sph_harm(m, l, phi, theta)
        return (b(theta, phi) * np.conj(ylm) * np.sin(theta)).real

    def integrand_imag(theta, phi, l, m):
        ylm = sph_harm(m, l, phi, theta)
        return (b(theta, phi) * np.conj(ylm) * np.sin(theta)).imag

    new_calculations_needed = any((l, m) not in alm for l in range(lmax + 1) for m in range(-l, l + 1))

    if new_calculations_needed:
        total_calculations = (lmax + 1) * (lmax + 1)
        progress_bar = tqdm(total=total_calculations, desc="Calculating coefficients")
    
    for l in range(lmax + 1):
        if (l, 0) not in alm:
            for m in range(-l, l + 1):
                real_result = scipy.integrate.dblquad(
                    integrand_real, 0, 2 * np.pi, lambda x: 0, lambda x: np.pi,
                    args=(l, m)
                )
                imag_result = scipy.integrate.dblquad(
                    integrand_imag, 0, 2 * np.pi, lambda x: 0, lambda x: np.pi,
                    args=(l, m)
                )
                alm[(l, m)] = real_result[0] + 1j * imag_result[0]
                if new_calculations_needed:
                    progress_bar.update(1)

    if new_calculations_needed:
        progress_bar.close()
        write_alm_to_csv(csv_filename, alm)
    return alm

def recons(alm, x, y, lmax):
    sizex = x.shape
    brecons = np.zeros(sizex, dtype=complex)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            if np.isnan(alm[(l, m)]):
                continue  # Skip if alm is NaN
            ylm = sph_harm(m, l, x, y)
            brecons += alm[(l, m)] * ylm
    return brecons.real

def plotty(ax, x, y, bvals, type, lmax, num_points_theta, num_points_phi, contnum, bname):
    vmin = np.min(bvals)
    vmax = np.max(bvals)
    clevels = np.linspace(vmin, vmax, contnum + 1)
    contourf_plot = ax.contourf(x, y, bvals, levels=clevels, cmap='viridis', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(contourf_plot, ax=ax, label='Value')
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f'{vmin:.2f}', f'{vmax:.2f}'])
    ax.set_xlabel('x or phi')
    ax.set_ylabel('y or theta')
    name = ''
    if type == 1:
        name = f'og function: {bname}'
    elif type == 2:
        name = f'recons (lmax: {lmax}, theta: {num_points_theta}, phi: {num_points_phi})'
    elif type == 3:
        name = 'Delta'
    ax.set_title(name)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels([r'0', r'$\pi/2$', r'$2\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.set_yticks(np.linspace(0, np.pi, 5))
    ax.set_yticklabels([r'0', r'$\pi/4$', r'$pi/2$', r'$3\pi/4$', r'$pi$'])

def beepbeep(f,d):
    winsound.Beep(f,d)

fig, axs = plt.subplots(2, 2, figsize=(15, 12))
axs = axs.flatten()

num_points_theta = 50
num_points_phi = 100
contnum = 100
lmax = 20
b_function_name = "sin(y)"

print("\n------------------------------------------------------------------------------------------------------------------------")
print("Imported all necessary libraries, compiled successfully.")
print("Current values:\n\tb\t\t\t-\t" + b_function_name + "\n\tlmax\t\t\t-\t" + str(lmax) + "\n\tnum_points_theta\t-\t" + str(num_points_theta) + "\n\tnum_points_phi\t\t-\t" + str(num_points_phi))

input("Press Enter to confirm these values and start the process:")

start_time = time.time()
start_ist = datetime.now(IST)
start_ist_str = start_ist.strftime("%Y-%m-%d %H:%M:%S %Z")

beepbeep(500,1300)

print("Process started...\nIST now: " + start_ist_str)

x = np.linspace(0, 2 * np.pi, num_points_phi)
y = np.linspace(0, np.pi, num_points_theta)
y, x = np.meshgrid(y, x)
plotty(axs[0], x, y, b(y, x), 1, 5, num_points_theta, num_points_phi, contnum, b_function_name)

alm = adaptive_calcalm(b, lmax, num_points_theta, num_points_phi, b_function_name)
x = np.linspace(0, 2 * np.pi, num_points_phi)
y = np.linspace(0, np.pi, num_points_theta)
y, x = np.meshgrid(y, x)
b_reconstructed = recons(alm, x, y, lmax)
plotty(axs[1], x, y, b_reconstructed, 2, lmax, num_points_theta, num_points_phi, contnum, b_function_name)

delta = b(y, x) - b_reconstructed
plotty(axs[2], x, y, delta, 3, lmax, num_points_theta, num_points_phi, contnum, b_function_name)

# Calculate and round error metrics
delta = b(y, x) - b_reconstructed
abs_tot_sum = round(np.sum(np.abs(delta)), 4)
max_delta = round(np.max(delta), 4)
min_delta = round(np.min(delta), 4)
avg_abs_delta = round(np.mean(np.abs(delta)), 4)
range_b = np.max(b(y, x)) - np.min(b(y, x))
percentage_error = round((avg_abs_delta / range_b) * 100, 4)
sum_error_div_nsquare = round(abs_tot_sum / (num_points_theta * num_points_phi), 4)

# Write the error metrics to a CSV file inside the specified folder
folder_path = get_folder_paths(b_function_name, lmax, num_points_theta, num_points_phi)
error_filename = os.path.join(folder_path, 'errors.csv')
with open(error_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Total Abs Delta', 'Max Delta', 'Min Delta', 'Avg Abs Delta', 'Percentage Error', 'Sum Error Div Nsquared'])
    writer.writerow([abs_tot_sum, max_delta, min_delta, avg_abs_delta, percentage_error, sum_error_div_nsquare])

axs[3].axis('off')  # Turn off the unused subplot

end_time = time.time()
end_ist = datetime.now(IST)
runtime = end_time - start_time

# Time in IST
end_ist_str = end_ist.strftime("%Y-%m-%d %H:%M:%S %Z")

csv_filename = os.path.join(folder_path, f'values.csv')
plt.figtext(0.95, 0.05, f"CSV used: {csv_filename}\nSum(abs(delta)): {abs_tot_sum}\n\n\nRuntime: {runtime:.4f} seconds\nStart Time (IST): {start_ist_str}\nEnd Time (IST): {end_ist_str}", va='bottom', ha='right')

plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2, wspace=0.3, hspace=0.6)

# Save the plot to a file inside the specified folder with a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_filename = os.path.join(folder_path, f'plot_{timestamp}.png')
plt.savefig(plot_filename)

beepbeep(600,600)

plt.show()