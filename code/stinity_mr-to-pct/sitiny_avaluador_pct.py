import os
import sys
import torch
import platform
import subprocess
import tkinter as tk
from tkinter import filedialog
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
from utils.infer_funcs import do_mr_to_pct

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

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("PSEUDO-CT GENERATOR AND EVALUATOR")
    print("="*60)

    # 1. MRI Selection
    input_mr_file = select_file("STEP 1: Select the original MRI (.nii or .nii.gz)")
    if not input_mr_file or input_mr_file == "": 
        sys.exit("Process terminated. MRI file is mandatory.")
    
    # 2. Real CT Selection
    real_ct_file = select_file("STEP 2: Select REAL CT (Cancel for inference only)")
    
    has_real_ct = False
    if real_ct_file and real_ct_file.strip() != "":
        has_real_ct = True

    # 3. Directory and Path Preparation
    # El resultado se guardará en la misma carpeta que la MRI de entrada
    output_directory = os.path.dirname(input_mr_file)
    file_name = os.path.basename(input_mr_file)
    output_pct_file = os.path.join(output_directory, f"{file_name.split('.')[0]}_sCT.nii.gz")

    # 4. Pseudo-CT Generation
    print("\n1. Generating Pseudo-CT from MRI...")
    model_route = "/Users/saramasdeusans/Desktop/TFG_FUSOFT/models/pretrained_net_final_20220825.pth" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_route):
        sys.exit(f"Error: Model weights not found at {model_route}")
        
    saved_model = torch.load(model_route, map_location=device)
    
    # Ejecución de la inferencia
    do_mr_to_pct(input_mr_file, output_pct_file, saved_model, device, prep_t1=True, plot_mrct=False)
    print(f"Inference completed. File saved at: {output_pct_file}")

    # Load images into memory
    pct_img = sitk.ReadImage(output_pct_file)
    pct_arr = sitk.GetArrayFromImage(pct_img)
    mri_arr = sitk.GetArrayFromImage(sitk.ReadImage(input_mr_file))
    
    num_slices = pct_arr.shape[0]
    initial_slice = num_slices // 2 

    # 5. Validation and Metrics
    mae = None
    if has_real_ct:
        print("\n2. Calculating comparison metrics (MAE)...")
        real_ct_img = sitk.ReadImage(real_ct_file)

        # Registro y remuestreo para asegurar coincidencia de dimensiones
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(pct_img)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1000) 
        real_ct_resampled = resampler.Execute(real_ct_img)

        ct_arr = sitk.GetArrayFromImage(real_ct_resampled)

        # Generación de máscara de tejido (excluyendo aire), dice coef os i teixit bla SSIM
        mask = ct_arr > -500 
        
        if np.any(mask):
            mae = np.mean(np.abs(pct_arr[mask] - ct_arr[mask]))
            print(f"RESULT: Mean Absolute Error (MAE) is {mae:.2f} HU.")
        else:
            print("WARNING: Tissue mask is empty. MAE cannot be calculated (spatial mismatch or threshold issue).")
            mae = 0.0

    # 6. Visualization
    if has_real_ct:
        print("\n3. Opening 3D interactive viewer (4 panels)...")
        error_map_3d = np.abs(pct_arr - ct_arr)

        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        plt.subplots_adjust(bottom=0.2)
        fig.suptitle(f"Validation - {file_name} - MAE: {mae:.2f} HU", fontsize=14)

        im1 = axes[0].imshow(mri_arr[initial_slice], cmap='gray')
        axes[0].set_title("1. Original MRI")
        axes[0].axis('off')

        im2 = axes[1].imshow(ct_arr[initial_slice], cmap='gray', vmin=-1000, vmax=1500)
        axes[1].set_title("2. Real CT (Resampled)")
        axes[1].axis('off')

        im3 = axes[2].imshow(pct_arr[initial_slice], cmap='gray', vmin=-1000, vmax=1500)
        axes[2].set_title("3. Generated Pseudo-CT")
        axes[2].axis('off')

        im4 = axes[3].imshow(error_map_3d[initial_slice], cmap='hot', vmin=0, vmax=500)
        axes[3].set_title("4. Error Map")
        axes[3].axis('off')
        fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    else:
        print("\n2. Opening 3D interactive viewer (2 panels)...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(bottom=0.2)
        fig.suptitle(f"Pseudo-CT Visualization: {file_name}", fontsize=14)

        im1 = axes[0].imshow(mri_arr[initial_slice], cmap='gray')
        axes[0].set_title("1. Original MRI")
        axes[0].axis('off')

        im3 = axes[1].imshow(pct_arr[initial_slice], cmap='gray', vmin=-1000, vmax=1500)
        axes[1].set_title("2. Generated Pseudo-CT")
        axes[1].axis('off')
        im2, im4, ct_arr, error_map_3d = None, None, None, None

    # --- INTERACTIVE CONTROL ---
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, num_slices - 1, valinit=initial_slice, valstep=1)

    def update_slice(val):
        idx = int(slider.val)
        im1.set_data(mri_arr[idx])
        im3.set_data(pct_arr[idx])
        if has_real_ct:
            im2.set_data(ct_arr[idx])
            im4.set_data(error_map_3d[idx])
        fig.canvas.draw_idle()

    slider.on_changed(update_slice)

    def mouse_scroll(event):
        if event.button == 'up':
            new_val = min(slider.val + 1, num_slices - 1)
        elif event.button == 'down':
            new_val = max(slider.val - 1, 0)
        else: return
        slider.set_val(new_val)

    fig.canvas.mpl_connect('scroll_event', mouse_scroll)
    plt.show()