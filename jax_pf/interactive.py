import numpy as np
import pandas as pd
import os

import jax.numpy as jnp

from typing import Callable, Tuple

from IPython.display import HTML, Image, display, Javascript
from google.colab.output import eval_js
from base64 import b64decode, b64encode

from datetime import datetime

Array = jnp.array

def get_path_function(file_path: str, x_s: float, t_s: float) -> Callable:
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, header=None, skiprows=1, names=['timestamp', 'x_position', 'y_position'])

    # Extract time, x, and y as NumPy arrays
    time_values = df['timestamp'].values / t_s
    x_values = df['x_position'].values / x_s
    y_values = df['y_position'].values / x_s
    
    # Define a function for linear interpolation
    def interpolate_coordinates(time_point: float) -> Tuple[float, float]:
        x_interp = jnp.interp(time_point, time_values, x_values)
        y_interp = jnp.interp(time_point, time_values, y_values)
        return x_interp, y_interp
    
    return interpolate_coordinates


canvas_html = """
<canvas width=%d height=%d style="background-color: white; border: 1px solid black;"></canvas>
<button id="finish-button">Finish</button>
<script>
var canvas = document.querySelector('canvas');
var ctx = canvas.getContext('2d');
ctx.lineWidth = %d;
var button = document.querySelector('#finish-button');
var mouse = {x: 0, y: 0};
var points = [];
var timestamps = [];

canvas.addEventListener('mousemove', function(e) {
  mouse.x = e.pageX - this.offsetLeft;
  mouse.y = e.pageY - this.offsetTop;
});

canvas.onmousedown = () => {
  ctx.beginPath();
  ctx.moveTo(mouse.x, mouse.y);
  points.push({ x: mouse.x, y: mouse.y });
  timestamps.push(new Date().toISOString());
  canvas.addEventListener('mousemove', onPaint);
};

canvas.onmouseup = () => {
  canvas.removeEventListener('mousemove', onPaint);
};

var onPaint = () => {
  ctx.lineTo(mouse.x, mouse.y);
  ctx.stroke();
  points.push({ x: mouse.x, y: mouse.y });
  timestamps.push(new Date().toISOString());
};

var data = new Promise(resolve => {
  button.onclick = () => {
    resolve({ image: canvas.toDataURL('image/png'), points: points, timestamps: timestamps });
  };
});
</script>
"""


def draw_and_save(filename=file_name, w=400, h=200, line_width=1):

    print('Use your mouse to draw a path for the light source in the white box. Press finish when done.')

    display(HTML(canvas_html % (w, h, line_width)))
    drawing_data = eval_js("data")
    
    # Save image
    binary = b64decode(drawing_data['image'].split(',')[1])
    with open(filename.replace('.csv', '.png'), 'wb') as f:
        f.write(binary)

    # Save drawing data to CSV
    df = pd.DataFrame({'Timestamp': drawing_data['timestamps'], 'X': [point['x'] for point in drawing_data['points']],
                       'Y': [point['y'] for point in drawing_data['points']]})
    df.to_csv(filename, index=False)
    print(f'Drawing data saved to {filename}')

def process_path(file_name: str, processed_file_name: str, final_time: float, x_range: Tuple[float, float], y_range: Tuple[float, float], x_orig: Tuple[float, float], y_orig: Tuple[float, float]):
    # Read the CSV file into a DataFrame, skip the first row, and assign column names if needed
    # If your CSV file has no header, set header=None
    # If your CSV file has column names, you can skip the 'header=None' part
    df = pd.read_csv(file_name, header=None, skiprows=1, names=['timestamp', 'x_position', 'y_position'])

    df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) / 10**9  # Convert to seconds

    # Normalize timestamps to be between 0 and final_time
    df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / (df['timestamp'].max() - df['timestamp'].min()) * final_time

    # Normalize x positions to be between x_range[0] and x_range[1]
    df['x_position'] = (df['x_position'] - x_orig[0]) / (x_orig[1] - x_orig[0]) * (x_range[1] - x_range[0]) + x_range[0]

    # Normalize y positions to be between y_range[0] and y_range[1]
    df['y_position'] = (df['y_position'] - y_orig[0]) / (y_orig[1] - y_orig[0]) * (y_range[1] - y_range[0]) + y_range[0]

    # Save the processed DataFrame to a new CSV file
    df.to_csv(processed_file_name, index=False)

    # Print out the path of the processed data file
    print("Processed data saved to:", processed_file_name)

def visualize_simulation(traj: Array, atoms: float, x_s: float, animation_filename: str, density: Callable, fps: int = 5):
    # Set up the figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    def update(ind):
        # Clear previous frames
        axs[0].cla()
        axs[1].cla()

        # Plot the updated frames
        axs[0].imshow(atoms * density(jnp.transpose(traj[ind]) / x_s))
        axs[0].set_title(f'Density {ind}')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].grid(False)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_aspect('equal')

        axs[1].imshow(np.angle(jnp.transpose(traj[ind])))
        axs[1].set_title(f'Phase {ind}')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Y')
        axs[1].grid(False)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_aspect('equal')

    # Set the total number of frames (adjust as needed)
    total_frames = len(traj)

    # Create the animation
    animation = FuncAnimation(fig, update, frames=total_frames, interval=200, repeat=False)

    # Save the animation as an MP4 file
    animation.save(animation_filename, writer='ffmpeg', fps=fps)

def show_video(filename='output.mp4', size=300):
  # Allows us to view mp4 videos inline in the notebook
  mp4 = open(filename,'rb').read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  return HTML(f"""
  <video width={size:d} controls>
        <source src="{data_url}" type="video/mp4">
  </video>
  """)