import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import filedialog

# --- CONFIGURACIÓ ---
PATCH_DIR = "/Users/saramasdeusans/Desktop/TRAIN_DATASET_PATCHES"

# --- INTERFÍCIE GRÀFICA PER TRIAR PATCH ---
class PatchViewer:
    def __init__(self, patches_list):
        self.patches = patches_list
        self.current_idx = 0
        self.load_current_patch_data()

        # Creem la figura i els eixos
        self.fig, (self.ax_mri, self.ax_ct) = plt.subplots(1, 2, figsize=(12, 7))
        plt.subplots_adjust(bottom=0.25)

        # Configuració inicial dels gràfics
        self.fig.suptitle(f"Validant Data Augmentation / Patches\n{self.current_patch_base}", fontsize=14)
        
        self.im_mri = self.ax_mri.imshow(np.zeros((128, 128)), cmap='gray', vmin=-1.5, vmax=1.5)
        self.ax_mri.set_title("MRI Patch (Normalized)")
        self.ax_mri.axis('off')

        self.im_ct = self.ax_ct.imshow(np.zeros((128, 128)), cmap='gray', vmin=-1, vmax=1)
        self.ax_ct.set_title("CT Patch (Target)")
        self.ax_ct.axis('off')

        # Slider de navegació per les llesques (Axis Z)
        self.s_slice = Slider(plt.axes([0.25, 0.15, 0.5, 0.03]), 'Llesca Z', 0, self.mri_data.shape[0]-1, valinit=self.mri_data.shape[0]//2, valstep=1)
        self.s_slice.on_changed(self.update_slice)

        # Botons per canviar de patch
        self.btn_prev = Button(plt.axes([0.1, 0.05, 0.15, 0.075]), 'Anterior')
        self.btn_prev.on_clicked(self.prev_patch)

        self.btn_next = Button(plt.axes([0.75, 0.05, 0.15, 0.075]), 'Següent')
        self.btn_next.on_clicked(self.next_patch)

        # Dibuix inicial
        self.update_slice(self.mri_data.shape[0]//2)

    def load_current_patch_data(self):
        """Carrega el parell de patches (MRI i CT) actual"""
        self.current_patch_base = self.patches[self.current_idx]
        mri_path = os.path.join(PATCH_DIR, f"{self.current_patch_base}_mri.npy")
        ct_path = os.path.join(PATCH_DIR, f"{self.current_patch_base}_ct.npy")
        
        self.mri_data = np.load(mri_path)
        self.ct_data = np.load(ct_path)

    def update_slice(self, val):
        """Actualitza la visualització de la llesca"""
        idx = int(val)
        self.im_mri.set_data(np.flipud(self.mri_data[idx, :, :]))
        self.im_ct.set_data(np.flipud(self.ct_data[idx, :, :]))
        self.fig.canvas.draw_idle()

    def update_patch(self):
        """Actualitza tot el visor quan canviem de patch"""
        self.load_current_patch_data()
        self.fig.suptitle(f"Validant Data Augmentation / Patches\n{self.current_patch_base}", fontsize=14)
        
        # Ajustem els límits del slider per la nova mida de patch
        self.s_slice.valmax = self.mri_data.shape[0] - 1
        self.s_slice.set_val(self.mri_data.shape[0] // 2)
        self.fig.canvas.draw_idle()

    def next_patch(self, event):
        self.current_idx = (self.current_idx + 1) % len(self.patches)
        self.update_patch()

    def prev_patch(self, event):
        self.current_idx = (self.current_idx - 1) % len(self.patches)
        self.update_patch()

# --- BUCLE PRINCIPAL ---
if __name__ == "__main__":
    # 1. Comprovem que la carpeta existeix
    if not os.path.exists(PATCH_DIR):
        print(f"❌ Error: No s'ha trobat la carpeta {PATCH_DIR}")
        sys.exit()

    # 2. Busquem la llista de patches (només els de MRI, per no duplicar)
    mri_patches = sorted(glob.glob(os.path.join(PATCH_DIR, "*_mri.npy")))
    if len(mri_patches) == 0:
        print("❌ Error: No hi ha fitxers .npy a la carpeta.")
        sys.exit()

    # 3. Extraiem el nom base (subXXXX_patch_Y)
    patch_bases = [os.path.basename(f).replace("_mri.npy", "") for f in mri_patches]
    print(f"✅ S'han trobat {len(patch_bases)} parelles de patches per validar.")

    # 4. Iniciem el visor
    plt.close('all') 
    viewer = PatchViewer(patch_bases)
    plt.show()