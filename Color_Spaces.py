import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# --- 1. Create a Sample RGB Image ---
# Creating a simple 50x50 RGB image with distinct colors for visualization
img_rgb = cv2.cvtColor(cv2.imread("Example.tif"),cv2.COLOR_BGR2RGB)

# --- 2. Define Conversion Flags and Storage ---
# Curated list of common RGB to X conversions
target_flags = [
    'COLOR_RGB2GRAY',
    'COLOR_RGB2HSV',
    'COLOR_RGB2HLS',
    'COLOR_RGB2LAB',
    'COLOR_RGB2YCrCb',
    'COLOR_RGB2XYZ',
]

# Dictionary to store all converted channels: {Title: (image_data, colormap)}
converted_channels = {}

# --- 3. Perform Conversions and Store Channels ---

# Store the original RGB image and its individual channels first
converted_channels['Original RGB'] = (img_rgb, None)
converted_channels['RGB_Red'] = (img_rgb[:,:,0], 'Reds')
converted_channels['RGB_Green'] = (img_rgb[:,:,1], 'Greens')
converted_channels['RGB_Blue'] = (img_rgb[:,:,2], 'Blues')

for flag_name in target_flags:
    flag = getattr(cv2, flag_name)
    try:
        converted_img = cv2.cvtColor(img_rgb, flag)
        
        # Define channel names based on the color space for better titles
        channel_names = ['CH0', 'CH1', 'CH2']
        if 'HSV' in flag_name:
            channel_names = ['HUE', 'SATURATION', 'VALUE']
        elif 'HLS' in flag_name:
            channel_names = ['HUE', 'LIGHTNESS', 'SATURATION']
        elif 'LAB' in flag_name:
            channel_names = ['LIGHTNESS', 'A (Green-Red)', 'B (Blue-Yellow)']
        elif 'YCrCb' in flag_name:
            channel_names = ['Y (Luma)', 'Cr (Red Diff)', 'Cb (Blue Diff)']
        elif 'XYZ' in flag_name:
            channel_names = ['X', 'Y', 'Z']
        
        
        if converted_img.ndim == 2:
            # Grayscale image (1 channel)
            converted_channels[f'{flag_name.replace("COLOR_RGB2", "")}_GRAY'] = (converted_img, 'gray')
        elif converted_img.shape[2] == 3:
            # 3-channel image. Split and store.
            channels = cv2.split(converted_img)
            
            for i, channel in enumerate(channels):
                # All split channels are treated as grayscale maps
                converted_channels[f'{flag_name.replace("COLOR_RGB2", "")}_{channel_names[i]}'] = (channel, 'gray')

    except Exception as e:
        # Simple error handling for robustness
        print(f"Failed to convert using {flag_name}: {e}")


# --- 4. Plotting Setup ---
num_plots = len(converted_channels)
plot_names = list(converted_channels.keys())

# Determine the grid size (nearly square, e.g., 20 plots -> 5x5 grid)
side = math.ceil(math.sqrt(num_plots))

# Create the figure
fig, axes = plt.subplots(side, side, figsize=(side * 3, side * 3))
axes = axes.flatten()

# --- 5. Plot all channels ---
for i, ax in enumerate(axes):
    if i < num_plots:
        img_to_plot, cmap_name = converted_channels[plot_names[i]]
        title = plot_names[i].replace('_', ' ')
        
        # Display logic: color for Original RGB, grayscale for channels
        if img_to_plot.ndim == 3: 
            ax.imshow(img_to_plot)
        else:
            # Use the specified colormap (e.g., 'Reds', 'gray')
            ax.imshow(img_to_plot, cmap=cmap_name) 

        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # Hide any extra subplots if the grid isn't perfectly full
        fig.delaxes(ax)

# --- 6. Save the figure as PDF with tight layout ---
fig.tight_layout(pad=1.5) 
output_filename = "All_Color_Spaces.pdf"
plt.savefig(output_filename, format='pdf', bbox_inches="tight")

print(f"\nFigure containing {num_plots} channels saved to {output_filename}")