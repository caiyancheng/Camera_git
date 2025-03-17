import numpy as np
import json
import matplotlib.pyplot as plt

from Color_correction.Color_space_Transform import cm_rgb2xyz
from Color_space_Transform import *
from Color_space_Transform import cm_xyz2rgb
from display_encoding import display_encode

# Initialize the display encoding tool
max_luminance = 400
display_encode_tool = display_encode(max_luminance)

# Load the data
with open('color_correction_display_color_rects.json', 'r') as f:
    data = json.load(f)

Display_RGB_all = np.array(data['All_input_RGB'])
plot_scale = 'Log'
# plot_scale = 'Linear'
# Identity matrix for transformation
M = np.eye(3)

# Initialize lists for the RGB values
Display_RGB_Linear_N_list_R = []
Display_RGB_Linear_N_list_G = []
Display_RGB_Linear_N_list_B = []
Camera_RGB_Linear_N_list_R = []
Camera_RGB_Linear_N_list_G = []
Camera_RGB_Linear_N_list_B = []

symbol_list = []
symbol_colors = {0: 'b', 1: 'g', 2: 'r'}
# Iterate through all the colors in the input list
for index in range(len(Display_RGB_all)):
    Display_RGB = Display_RGB_all[index]
    if Display_RGB[0] > 0:
        symbol_list.append(0)
    elif Display_RGB[1] > 0:
        symbol_list.append(1)
    elif Display_RGB[2] > 0:
        symbol_list.append(2)

    real_XYZ_value = data[f'r{Display_RGB[0]}, g{Display_RGB[1]}, b{Display_RGB[2]}']['Camera Mean XYZ Linear']

    # Convert Display RGB to linear sRGB, then to P3, and apply transformation
    Display_RGB_normal = Display_RGB / 255
    Display_RGB_Linear_709 = np.maximum(display_encode_tool.C2L_sRGB(Display_RGB_normal), 0)
    Display_RGB_Linear_P3 = np.maximum(cm_xyz2rgb(cm_rgb2xyz(Display_RGB_Linear_709, rgb_space='sRGB'), rgb_space='P3'),
                                       0)
    Display_RGB_Linear_N = np.maximum(np.dot(Display_RGB_Linear_P3, M.T), 0)

    # Append to the respective lists for display
    Display_RGB_Linear_N_list_R.append(Display_RGB_Linear_N[0])
    Display_RGB_Linear_N_list_G.append(Display_RGB_Linear_N[1])
    Display_RGB_Linear_N_list_B.append(Display_RGB_Linear_N[2])

    # Convert Camera RGB to linear P3 and apply inverse transformation
    Camera_RGB_Linear_P3 = np.maximum(cm_xyz2rgb(real_XYZ_value, rgb_space='P3'), 0)
    Camera_RGB_Linear_N = np.maximum(np.dot(Camera_RGB_Linear_P3, np.linalg.inv(M).T), 0)

    # Append to the respective lists for camera
    Camera_RGB_Linear_N_list_R.append(Camera_RGB_Linear_N[0])
    Camera_RGB_Linear_N_list_G.append(Camera_RGB_Linear_N[1])
    Camera_RGB_Linear_N_list_B.append(Camera_RGB_Linear_N[2])

# Plot the results
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Plot for Red channel
# axs[0].scatter(Display_RGB_Linear_N_list_R, Camera_RGB_Linear_N_list_R, c='r', alpha=0.5)
for index in range(len(Display_RGB_Linear_N_list_R)):
    symbol = symbol_list[index]
    color = symbol_colors.get(symbol, 'k')  # Default to black if symbol is not 0, 1, or 2
    axs[0].scatter(Display_RGB_Linear_N_list_R[index], Camera_RGB_Linear_N_list_R[index], edgecolor=color,
                   facecolor='r', alpha=0.5)
if plot_scale == 'Log':
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlim(0.1, max_luminance)  # Set x-axis limits
    axs[0].set_ylim(0.1, max_luminance)  # Set y-axis limits
elif plot_scale == 'Linear':
    axs[0].set_xlim(0, max_luminance)  # Set x-axis limits
    axs[0].set_ylim(0, max_luminance)  # Set y-axis limits
axs[0].set_xlabel('Display RGB Linear R (log scale)')
axs[0].set_ylabel('Camera RGB Linear R (log scale)')
axs[0].set_title('Red Channel')

# Plot for Green channel
# axs[1].scatter(Display_RGB_Linear_N_list_G, Camera_RGB_Linear_N_list_G, c='g', alpha=0.5)
for index in range(len(Display_RGB_Linear_N_list_G)):
    symbol = symbol_list[index]
    color = symbol_colors.get(symbol, 'k')  # Default to black if symbol is not 0, 1, or 2
    axs[1].scatter(Display_RGB_Linear_N_list_G[index], Camera_RGB_Linear_N_list_G[index], edgecolor=color,
                   facecolor='g', alpha=0.5)
if plot_scale == 'Log':
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlim(0.1, max_luminance)  # Set x-axis limits
    axs[1].set_ylim(0.1, max_luminance)  # Set y-axis limits
elif plot_scale == 'Linear':
    axs[1].set_xlim(0, max_luminance)  # Set x-axis limits
    axs[1].set_ylim(0, max_luminance)  # Set y-axis limits
axs[1].set_xlabel('Display RGB Linear G (log scale)')
axs[1].set_ylabel('Camera RGB Linear G (log scale)')
axs[1].set_title('Green Channel')

# Plot for Blue channel
# axs[2].scatter(Display_RGB_Linear_N_list_B, Camera_RGB_Linear_N_list_B, c='b', alpha=0.5)
for index in range(len(Display_RGB_Linear_N_list_B)):
    symbol = symbol_list[index]
    color = symbol_colors.get(symbol, 'k')  # Default to black if symbol is not 0, 1, or 2
    axs[2].scatter(Display_RGB_Linear_N_list_B[index], Camera_RGB_Linear_N_list_B[index], edgecolor=color,
                   facecolor='b', alpha=0.5)
if plot_scale == 'Log':
    axs[2].set_xscale('log')
    axs[2].set_yscale('log')
    axs[2].set_xlim(0.1, max_luminance)  # Set x-axis limits
    axs[2].set_ylim(0.1, max_luminance)  # Set y-axis limits
elif plot_scale == 'Linear':
    axs[2].set_xlim(0, max_luminance)  # Set x-axis limits
    axs[2].set_ylim(0, max_luminance)  # Set y-axis limits
axs[2].set_xlabel('Display RGB Linear B (log scale)')
axs[2].set_ylabel('Camera RGB Linear B (log scale)')
axs[2].set_title('Blue Channel')

# Adjust layout
plt.tight_layout()
plt.show()
