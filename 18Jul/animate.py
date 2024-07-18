import cv2
import os
from rich.progress import Progress

# Path to the directory containing the images
image_folder = 'E:/SheshAditya/18Jul/plots/CR Maps'

# Get the list of image files in the directory
images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png") or img.endswith(".jpg")]

# Parameters for the video
frame_rate = 1 / 0.05  # Frame rate to keep each image for 0.05 seconds
output_video_path = 'animations/animated18JulplotsCR Maps.mp4'  # Change the output file extension to .mp4

# Read the first image to get the size (assuming all images are the same size)
first_image_path = os.path.join(image_folder, images[0])
first_image = cv2.imread(first_image_path)
height, width, layers = first_image.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use the mp4v codec for MP4 files
video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

# Use rich progress bar
with Progress() as progress:
    task = progress.add_task("Processing images", total=len(images))
    
    # Loop through all the images and write them to the video with a progress bar
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)
        progress.advance(task)

# Release the video writer object
video.release()

print(f"Video saved as {output_video_path}")