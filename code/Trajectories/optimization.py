import sys
import os
import platform
import subprocess
import tkinter as tk
from tkinter import filedialog
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec

# --- FILE SELECTION FUNCTION ---
def select_file(title):
    system = platform.system() 
    if system == 'Darwin': 
        apple_script = f'''
        try
            set theFile to choose file with prompt "{title}"
            POSIX path of theFile
        on error
            return ""
        end try
        '''
        result = subprocess.run(['osascript', '-e', apple_script], capture_output=True, text=True)
        return result.stdout.strip()
    else:
        root = tk.Tk()
        root.withdraw() 
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(title=title, filetypes=[("Medical Images", "*.nii *.nii.gz")])
        root.destroy()
        return path

class MultimodalTrajectoryPlanner:
    def __init__(self, ct_path):
        print(f"Loading CT Volume: {ct_path}")
        self.ct_image = sitk.ReadImage(ct_path)
        self.ct_array = sitk.GetArrayFromImage(self.ct_image) # Assumes standard [Z, Y, X] from Sitk
        
        self.num_slices = self.ct_array.shape[0] # Z-axis (Axial slices)
        self.height = self.ct_array.shape[1]      # Y-axis (Anterior-Posterior)
        self.width = self.ct_array.shape[2]       # X-axis (Left-Right)
        
        self.current_slice = self.num_slices // 2
        self.target_point = None # Stored as (X, Y, Z) voxel coords
        
        # Heuristic Weights (TFG defined parameters)
        self.W_DISTANCE = 1.3
        self.W_BONE = 1.2
        self.W_DENSITY = 1.1
        
        self.setup_ui()

    def setup_ui(self):
        self.fig = plt.figure(figsize=(18, 9))
        self.fig.canvas.manager.set_window_title("FUSpine Multimodal Treatment Planner (PoC)")
        
        # Define layout with GridSpec (3 columns)
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
        
        # 1. Axial Panel (Interactive Clicks)
        self.ax_axial = self.fig.add_subplot(gs[0])
        self.ax_axial.set_title("1. Axial Plane (Click to set Target)")
        self.ax_axial.axis('off')
        
        # 2. Sagital Panel (Visualization Only)
        self.ax_sagittal = self.fig.add_subplot(gs[1])
        self.ax_sagittal.set_title("2. Sagittal Plane (Transducer Placement)")
        self.ax_sagittal.axis('off')
        
        # 3. Profile Panel (Top-1 Validation)
        self.ax_profile = self.fig.add_subplot(gs[2])
        self.ax_profile.set_title("3. Top-1 Trajectory Density Profile")
        self.ax_profile.set_xlabel("Distance from Entry (pixels)")
        self.ax_profile.set_ylabel("Hounsfield Units (HU)")
        self.ax_profile.grid(True)
        self.profile_line, = self.ax_profile.plot([], [], 'g-', linewidth=2)
        self.ax_profile.set_ylim(-1000, 2000)

        plt.subplots_adjust(bottom=0.2)
        
        # Slice Slider
        self.ax_slider = plt.axes([0.1, 0.05, 0.3, 0.03])
        self.slider = Slider(self.ax_slider, 'Axial Slice', 0, self.num_slices - 1, valinit=self.current_slice, valstep=1)
        self.slider.on_changed(self.update_axial_slice)
        
        # Mouse interactions
        self.fig.canvas.mpl_connect('button_press_event', self.on_click_axial)
        self.fig.canvas.mpl_connect('scroll_event', self.mouse_scroll)
        
        self.redraw_axial()
        self.ax_sagittal.imshow(np.zeros((self.num_slices, self.height)), cmap='gray', origin='lower') # Initial black sagittal
        
        plt.show()

    def update_axial_slice(self, val):
        self.current_slice = int(self.slider.val)
        # Target reset logic optional. Here we keep it but don't re-optimize until new click.
        # self.target_point = None 
        self.redraw_axial()

    def mouse_scroll(self, event):
        if event.button == 'up':
            new_slice = min(self.slider.val + 1, self.num_slices - 1)
        elif event.button == 'down':
            new_slice = max(self.slider.val - 1, 0)
        else:
            return
        self.slider.set_val(new_slice)

    def redraw_axial(self):
        self.ax_axial.clear()
        self.ax_axial.imshow(self.ct_array[self.current_slice], cmap='gray', vmin=-1000, vmax=1500)
        self.ax_axial.set_title(f"Axial Slice {self.current_slice} - Click for Target")
        self.ax_axial.axis('off')
        
        if self.target_point and self.target_point[2] == self.current_slice:
            # Draw Target as red cross on AXIAL
            self.ax_axial.plot(self.target_point[0], self.target_point[1], 'r+', markersize=15, markeredgewidth=2)
            
        self.fig.canvas.draw_idle()

    def on_click_axial(self, event):
        if event.inaxes != self.ax_axial: return
        
        # Store Target as X, Y voxel coords on the current Z slice
        self.target_point = (int(event.xdata), int(event.ydata), self.current_slice)
        print(f"\nTarget acquired at X:{self.target_point[0]}, Y:{self.target_point[1]}, Z:{self.target_point[2]}")
        
        self.redraw_axial()
        self.run_multimodal_optimization()

    def run_multimodal_optimization(self):
        print("Running multimodal heuristic optimization...")
        x_target, y_target, z_target = self.target_point
        slice_data = self.ct_array[z_target] # Perform 2D search within this axial slice
        
        all_results = []
        
        # 1. Heuristic Scan (Iterate angles)
        angles = np.linspace(np.pi, 2 * np.pi, 60) # Arc sweep posterior spine
        
        for angle in angles:
            max_radius = max(self.width, self.height)
            x_ray = np.int_(x_target + np.cos(angle) * np.arange(max_radius))
            y_ray = np.int_(y_target + np.sin(angle) * np.arange(max_radius))
            
            valid_idx = (x_ray >= 0) & (x_ray < self.width) & (y_ray >= 0) & (y_ray < self.height)
            x_ray, y_ray = x_ray[valid_idx], y_ray[valid_idx]
            
            if len(x_ray) == 0: continue
            intensities = slice_data[y_ray, x_ray]
            
            air_indices = np.where(intensities < -300)[0]
            if len(air_indices) == 0: continue
            
            skin_idx = air_indices[0]
            if skin_idx < 10: continue 
            
            # Extract path from entry (skin) to target
            path_intensities = intensities[:skin_idx][::-1]
            entry_x = x_ray[skin_idx]
            entry_y = y_ray[skin_idx]
            
            # Evaluate Pillars
            distance = len(path_intensities)
            bone_pixels = np.sum(path_intensities > 300)
            mean_density = np.mean(path_intensities)
            
            # Apply weighted cost function (lower score is better)
            score = (self.W_DISTANCE * distance) + (self.W_BONE * bone_pixels) + (self.W_DENSITY * mean_density)
            
            all_results.append({
                'score': score,
                'entry_point': (entry_x, entry_y),
                'profile': path_intensities
            })

        # Sort results by score (ascending)
        all_results.sort(key=lambda x: x['score'])

        # --- FASE A: VISUALITZACIÓ AXIAL (Top-10 i Transductor) ---
        print(f"Optimization complete. Drawing Top-10 trajectories on Axial plane.")
        top_1 = all_results[0]
        
        # Draw Top 10 paths in faint red/orange, but use faint green for Top-1 background
        for i in range(min(10, len(all_results))):
            path = all_results[i]
            entry_x, entry_y = path['entry_point']
            alpha_val = 1.0 - (i * 0.08) # Fade faint paths
            
            color = 'g-' if i == 0 else 'r-' # Highlight best path line faintly
            self.ax_axial.plot([entry_x, x_target], [entry_y, y_target], color, alpha=alpha_val, linewidth=0.5)

        # Draw THE Transductor for Top-1 (Large green circle)
        top_1_entry = top_1['entry_point']
        self.ax_axial.plot(top_1_entry[0], top_1_entry[1], 'go', markersize=10, label="Optimal Transducer Placement")
        
        # --- FASE B: VISUALITZACIÓ SAGITAL (Projecció i Col·locació) ---
        print(f"Displaying Sagittal projection through X-coord {x_target}.")
        self.ax_sagittal.clear()
        
        # Sitk Array [Z, Y, X]. Sagittal cut fixes X. Need [Z, Y]. Origin lower so Z is Up.
        sagittal_slice = self.ct_array[:, :, x_target]
        
        self.ax_sagittal.imshow(sagittal_slice, cmap='gray', origin='lower', aspect=1.0) # aspect 1.0 ensures vertical scale is correct
        self.ax_sagittal.set_title(f"Sagittal Plane (Cut at X:{x_target})")
        self.ax_sagittal.set_xlabel("Anterior-Posterior (Y)")
        self.ax_sagittal.set_ylabel("Superior-Inferior (Z)")
        self.ax_sagittal.axis('off')
        
        # Project the target onto Sagittal (Y, Z) axes
        self.ax_sagittal.plot(y_target, z_target, 'r+', markersize=15, markeredgewidth=2)
        
        # Project the optimal Transducer (skin point) onto Sagittal (Y, Z) axes
        # Important clinical limitation: Sound is assumed to come from this vertical level (Z)
        top_1_entry_y = top_1_entry[1]
        self.ax_sagittal.plot(top_1_entry_y, z_target, 'go', markersize=10) # Draws transductor at skin

        # Draw clinical trajectory on sagittal
        self.ax_sagittal.plot([top_1_entry_y, y_target], [z_target, z_target], 'g-', linewidth=2)
        
        # --- FASE C: ACTUALITZACIÓ DEL PERFIL DE DENSITAT (Top-1) ---
        best_profile = top_1['profile']
        self.profile_line.set_data(range(len(best_profile)), best_profile)
        self.ax_profile.set_xlim(0, len(best_profile))
        self.ax_profile.set_title(f"Top-1 Density Profile (Score: {top_1['score']:.0f})")
        
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("FUSPINE CLINICAL TRAJECTORY PLANNER")
    print("="*60)

    # Replaced hardcoded path with dynamic selection
    ct_file_path = select_file("Select the CT or Pseudo-CT file (.nii or .nii.gz)")
    
    if not ct_file_path or ct_file_path == "":
        sys.exit("Process canceled. A CT file is required.")
        
    try:
        app = MultimodalTrajectoryPlanner(ct_file_path)
    except Exception as e:
        print(f"Failed to load image. Ensure Sitk NIfTI dimensions are [Z,Y,X]: {e}")