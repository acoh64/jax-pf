import pygame
import os
import datetime
import pandas as pd

########################################
########################################
############## USER INPUT ##############
########################################
########################################

# File name to save path to
file_name = "path.txt"
# File name to save the processed path to
processed_file_name = "processed_path.csv"
# Directory to save path to
save_path = "drawn_paths"

# Final time in seconds of the path
final_time = 1.0  #seconds

# Length of the x and y axes in meters
Lx = 150e-6 #meters
Ly = 150e-6 #meters

# Size of window display for drawing path
width, height = 1000, 1000

########################################
########################################
########################################
########################################
########################################

# Set up output ranges
x_range = (-Lx/2, Lx/2)  
y_range = (-Ly/2, Ly/2) 

x_orig = (0.0, width-1)
y_orig = (0.0, height-1)

# Initialize Pygame
pygame.init()

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Draw and Save Path")

# Set up colors
white = (255, 255, 255)
black = (0, 0, 0)

# Set up clock
clock = pygame.time.Clock()

# Initialize variables
drawing = False
points = []
timestamps = []
start_time = None

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            points = [event.pos]  # Initialize with the first point
            timestamps = [datetime.datetime.now()]  # Initialize with the first timestamp
            start_time = datetime.datetime.now()
        elif event.type == pygame.MOUSEMOTION and drawing:
            points.append(event.pos)
            timestamps.append(datetime.datetime.now())
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            end_time = datetime.datetime.now()
            elapsed_time = end_time - start_time

            # Save the path and timestamps with millisecond precision
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, file_name), 'w') as file:
                file.write(f"Elapsed Time: {elapsed_time.total_seconds()} seconds\n")
                for timestamp, point in zip(timestamps, points):
                    # Format timestamp with millisecond precision
                    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    file.write(f"{timestamp_str},{point[0]},{point[1]}\n")

            # Print out the path
            print("Path saved to:", os.path.join(save_path, file_name))

            # Close the window after drawing is done
            running = False

    # Draw the path on the screen
    screen.fill(white)
    if drawing and len(points) > 1:
        pygame.draw.lines(screen, black, False, points, 2)

    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()

# Set up the file path
file_path = os.path.join(save_path, file_name)  

# Read the CSV file into a DataFrame, skip the first row, and assign column names if needed
# If your CSV file has no header, set header=None
# If your CSV file has column names, you can skip the 'header=None' part
df = pd.read_csv(file_path, header=None, skiprows=1, names=['timestamp', 'x_position', 'y_position'])

df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) / 10**9  # Convert to seconds

# Normalize timestamps to be between 0 and final_time
df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / (df['timestamp'].max() - df['timestamp'].min()) * final_time

# Normalize x positions to be between x_range[0] and x_range[1]
df['x_position'] = (df['x_position'] - x_orig[0]) / (x_orig[1] - x_orig[0]) * (x_range[1] - x_range[0]) + x_range[0]

# Normalize y positions to be between y_range[0] and y_range[1]
df['y_position'] = (df['y_position'] - y_orig[0]) / (y_orig[1] - y_orig[0]) * (y_range[1] - y_range[0]) + y_range[0]

# Save the processed DataFrame to a new CSV file
output_file_path = os.path.join(save_path, processed_file_name) 
df.to_csv(output_file_path, index=False)

# Print out the path of the processed data file
print("Processed data saved to:", output_file_path)
