import matplotlib.pyplot as plt
import os
import glob
import sunpy.map

# Directory containing FITS files
fits_dir = r'E:\SheshAditya\fits_files'

# Create output directory if it doesn't exist
output_dir = r'E:\SheshAditya\18Jul\plots\CR Maps'
os.makedirs(output_dir, exist_ok=True)

# Pattern to match FITS files
fits_pattern = os.path.join(fits_dir, 'hmi.Synoptic_Mr_small.*.fits')

# Iterate over all FITS files matching the pattern
for fits_file in glob.glob(fits_pattern):
    # Load each FITS file into a Map
    syn_map = sunpy.map.Map(fits_file)
    
    # Plot setup
    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot(projection=syn_map)
    im = syn_map.plot(axes=ax)
    
    ax.coords[0].set_axislabel("Carrington Longitude [deg]")
    ax.coords[1].set_axislabel("Latitude [deg]")
    ax.coords.grid(color='black', alpha=0.6, linestyle='dotted', linewidth=0.5)
    
    cb = plt.colorbar(im, fraction=0.019, pad=0.1)
    cb.set_label(f"Radial magnetic field [{syn_map.unit}]")
    
    ax.set_ylim(bottom=0)
    ax.set_title(f"{syn_map.meta['content']},\n"
                 f"Carrington rotation {syn_map.meta['CAR_ROT']}")
    
    # Save the plot with a unique name based on the Carrington Map number
    map_number = os.path.basename(fits_file).split('.')[2]  # Extract xxxx from filename
    output_file = os.path.join(output_dir, f"CRmap_{map_number}.png")
    plt.savefig(output_file)
    plt.close(fig)  # Close the figure to free up memory
    print(f"CR Map number: {map_number} - Processed successfully")

print("All plots saved successfully.")