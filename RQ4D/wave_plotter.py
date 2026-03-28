# wave_plotter.py
# Copyright RomanAILabs - Daniel Harding
# Visually renders the Cl(4,0) Spacetime output from the RQ4D Engine

import matplotlib.pyplot as plt
import numpy as np
import os

print("\n--- ROMAN AI LABS SPACETIME VISUALIZER ---")
file_path = "spacetime_wave_data.csv"

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
    exit()

# Load the CSV data exported by Go
print("Loading Manifold Data...")
data = np.loadtxt(file_path, delimiter=',')

# Create a heatmap of the wave diffusion
plt.figure(figsize=(10, 6))
plt.imshow(data, aspect='auto', cmap='magma', interpolation='nearest')
plt.colorbar(label='e12 Bivector Energy Density')
plt.title('RQ4D Cl(4,0) Wave Mechanics: Spacetime Diffusion Heatmap', fontsize=14)
plt.xlabel('Lattice Node (Relative to Epicenter)', fontsize=12)
plt.ylabel('Time Step (T)', fontsize=12)

# Save and show
out_img = "Spacetime_Diffusion_Plot.png"
plt.savefig(out_img, dpi=300, bbox_inches='tight')
print(f"Visualization saved to {out_img}")
plt.show()
