import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
import csv
import os
import time
from datetime import datetime, timedelta, timezone

# IST is UTC + 5:30
IST = timezone(timedelta(hours=5, minutes=30))

def b(y, x):
    return np.sin(y)
    # return np.sin(x)
    # return np.cos(y)
    # return np.cos(x)

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

def recons(alm, x, y, lmax):
    sizex = x.shape
    brecons = np.zeros(sizex, dtype=complex)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            ylm = sph_harm(m, l, x, y)
            brecons += alm[(l, m)] * ylm
    return brecons.real

def calculate_total_sum_delta(lmax, num_points_theta_list):
    total_sum_deltas = []

    for num_points_theta in num_points_theta_list:
        num_points_phi = 2 * num_points_theta
        csv_filename = f'alm_values_theta_{num_points_theta}_phi_{num_points_phi}_lmax_{lmax}.csv'
        alm = read_alm_from_csv(csv_filename)

        x = np.linspace(0, 2 * np.pi, num_points_phi)
        y = np.linspace(0, np.pi, num_points_theta)
        y, x = np.meshgrid(y, x)
        b_recons = recons(alm, x, y, lmax)
        delta = b(y, x) - b_recons
        total_sum = np.sum(np.abs(delta))
        total_sum_deltas.append(total_sum)

    return total_sum_deltas

input("Press Enter to start the process:")
start_time = time.time()
start_ist = datetime.now(IST)
start_ist_str = start_ist.strftime("%Y-%m-%d %H:%M:%S %Z")
print("Process started...\nIST now: " + start_ist_str)

lmax = 60
num_points_theta_list = []
num_points_theta = 100

while num_points_theta <= 250:
    num_points_theta_list.append(num_points_theta)
    num_points_theta += 10

total_sum_deltas = calculate_total_sum_delta(lmax, num_points_theta_list)

plt.figure(figsize=(10, 6))
plt.plot(num_points_theta_list, total_sum_deltas, marker='o')
plt.xlabel('Number of Points Theta')
plt.ylabel('Total Sum of Delta')
plt.title(f'Total Sum of Delta vs. Number of Points Theta for lmax = {lmax}')
plt.grid(True)

end_time = time.time()
end_ist = datetime.now(IST)
runtime = end_time - start_time

# Time in IST
end_ist_str = end_ist.strftime("%Y-%m-%d %H:%M:%S %Z")

# Display times and runtime
plt.figtext(0.15, 0.15, f"Runtime: {runtime:.4f} seconds\nStart Time (IST): {start_ist_str}\nEnd Time (IST): {end_ist_str}", va='bottom', ha='left')

plt.show()
