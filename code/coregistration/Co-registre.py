import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
import numpy as np
import sys
import os
import subprocess
import platform
import tkinter as tk
from tkinter import filedialog

# --- FUNCIONS D'UTILITAT ---
def select_file(titol):
    sistem = platform.system() 
    if sistem == 'Darwin': 
        apple_script = f'try\n set theFile to choose file with prompt "{titol}"\n POSIX path of theFile\non error\n return ""\nend try'
        result = subprocess.run(['osascript', '-e', apple_script], capture_output=True, text=True)
        return result.stdout.strip()
    else:
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        path = filedialog.askopenfilename(title=titol, filetypes=[("Medical Images", "*.nii *.nii.gz *.dcm *.mha"), ("Tots els arxius", "*.*")])
        root.destroy()
        return path

def upload_image(ruta, tipus_pixel=None):
    if not os.path.exists(ruta): return None
    try:
        return sitk.ReadImage(ruta, tipus_pixel) if tipus_pixel else sitk.ReadImage(ruta)
    except: return None
    
def calcular_dice_global(mri_img, ct_img):
    # Llindars optimitzats per al càlcul matemàtic
    ct_mask = sitk.BinaryThreshold(ct_img, lowerThreshold=-150.0, upperThreshold=1500.0, insideValue=1, outsideValue=0)
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0); otsu_filter.SetOutsideValue(1)
    mri_mask = otsu_filter.Execute(mri_img)
    ct_mask = sitk.Cast(ct_mask, sitk.sitkUInt8); mri_mask = sitk.Cast(mri_mask, sitk.sitkUInt8)
    overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_filter.Execute(mri_mask, ct_mask)
    return overlap_filter.GetDiceCoefficient()

# --- CLASSES D'INTERFÍCIE ---

class VisorRaw:
    def __init__(self, mri_img, ct_img):
        self.mri = sitk.GetArrayFromImage(mri_img); self.ct = sitk.GetArrayFromImage(ct_img)
        self.eix = 0; self.pct = 0.5
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(bottom=0.25)
        self.slider = Slider(plt.axes([0.2, 0.1, 0.6, 0.03]), 'Position %', 0.0, 1.0, valinit=0.5)
        self.slider.on_changed(self.update)
        self.radio = RadioButtons(plt.axes([0.02, 0.4, 0.12, 0.15]), ('Axial (Z)', 'Coronal (Y)', 'Sagital (X)'))
        self.radio.on_clicked(self.change_view)
        self.draw()

    def get_cut(self, volum, pct):
        maxim = volum.shape[self.eix]; idx = int(pct * (maxim - 1))
        if self.eix == 0: return volum[idx, :, :], idx
        elif self.eix == 1: return volum[:, idx, :], idx
        else: return volum[:, :, idx], idx

    def update(self, val): self.pct = val; self.draw()
    def change_view(self, label):
        self.eix = 0 if 'Axial' in label else 1 if 'Coronal' in label else 2
        self.draw()

    def draw(self):
        mri_slide, idx_m = self.get_cut(self.mri, self.pct)
        ct_slide, idx_c = self.get_cut(self.ct, self.pct)
        self.ax1.clear(); self.ax1.imshow(np.flipud(mri_slide), cmap='gray'); self.ax1.axis('off')
        self.ax2.clear(); self.ax2.imshow(np.flipud(ct_slide), cmap='gray', vmin=-200, vmax=1000); self.ax2.axis('off')
        self.fig.canvas.draw_idle()

class CuttingTool:
    def __init__(self, ct_image):
        self.ct = ct_image; self.arr = sitk.GetArrayFromImage(ct_image)
        self.dim_z, self.dim_y, self.dim_x = self.arr.shape
        self.actual_cut = int(self.dim_z * 0.4); self.actual_slide = self.dim_y // 2
        self.fig, self.ax = plt.subplots(figsize=(8, 9))
        plt.subplots_adjust(bottom=0.3)
        self.s_nav = Slider(plt.axes([0.2, 0.15, 0.6, 0.03]), 'NAVIGATE (Y)', 0, self.dim_y-1, valinit=self.actual_slide, valstep=1)
        self.s_cut = Slider(plt.axes([0.2, 0.10, 0.6, 0.03]), 'CUT (Z)', 0, self.dim_z-1, valinit=self.actual_cut, valstep=1)
        self.s_nav.on_changed(self.update); self.s_cut.on_changed(self.update)
        self.btn = Button(plt.axes([0.4, 0.02, 0.2, 0.05]), 'CONFIRM CUT')
        self.btn.on_clicked(self.close)
        self.draw()

    def update(self, val):
        self.actual_slide = int(self.s_nav.val); self.actual_cut = int(self.s_cut.val)
        self.draw()

    def draw(self):
        vista = self.arr[:, self.actual_slide, :]; self.ax.clear()
        self.ax.imshow(np.flipud(vista), cmap='gray', vmin=-200, vmax=1000)
        self.ax.axhline(y=self.dim_z - self.actual_cut, color='red', linewidth=2, linestyle='--')
        self.ax.set_title(f"Slide Y: {self.actual_slide}"); self.ax.axis('off')
        self.fig.canvas.draw_idle()

    def close(self, event): plt.close(self.fig)

class CockpitMultiplanar:
    def __init__(self, mri, ct_crop, tx_geo):
        self.mri = mri; self.ct = ct_crop; self.tx_inicial = tx_geo
        self.m_arr = sitk.GetArrayFromImage(self.mri); self.dim_z, self.dim_y, self.dim_x = self.m_arr.shape
        self.idx_z = self.dim_z // 2; self.idx_y = self.dim_y // 2; self.idx_x = self.dim_x // 2
        self.params = {'rx':0, 'ry':0, 'rz':0, 'tx':0, 'ty':0, 'tz':0}
        self.final_manual_transformation = tx_geo
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 8))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.35)
        self.s_rx = Slider(plt.axes([0.05, 0.25, 0.2, 0.03]), 'Rot X', -45, 45, valinit=0)
        self.s_ry = Slider(plt.axes([0.05, 0.20, 0.2, 0.03]), 'Rot Y', -45, 45, valinit=0)
        self.s_rz = Slider(plt.axes([0.05, 0.15, 0.2, 0.03]), 'Rot Z', -45, 45, valinit=0)
        self.s_tx = Slider(plt.axes([0.35, 0.25, 0.2, 0.03]), 'Pos X', -100, 100, valinit=0)
        self.s_ty = Slider(plt.axes([0.35, 0.20, 0.2, 0.03]), 'Pos Y', -100, 100, valinit=0)
        self.s_tz = Slider(plt.axes([0.35, 0.15, 0.2, 0.03]), 'Pos Z', -150, 150, valinit=0)
        self.s_nz = Slider(plt.axes([0.65, 0.25, 0.25, 0.03]), 'Nav AXIAL', 0, self.dim_z-1, valinit=self.idx_z, valstep=1)
        self.s_ny = Slider(plt.axes([0.65, 0.20, 0.25, 0.03]), 'Nav CORONAL', 0, self.dim_y-1, valinit=self.idx_y, valstep=1)
        self.s_nx = Slider(plt.axes([0.65, 0.15, 0.25, 0.03]), 'Nav SAGITAL', 0, self.dim_x-1, valinit=self.idx_x, valstep=1)
        for s in [self.s_rx, self.s_ry, self.s_rz, self.s_tx, self.s_ty, self.s_tz]: s.on_changed(self.update_tx)
        for s in [self.s_nz, self.s_ny, self.s_nx]: s.on_changed(self.update_nav)
        self.btn = Button(plt.axes([0.4, 0.05, 0.2, 0.05]), 'CONFIRM ALIGNMENT')
        self.btn.on_clicked(self.close); self.c_mask = np.zeros_like(self.m_arr)
        self.draw_initial()

    def recalculate_transformation(self):
        rad_x, rad_y, rad_z = np.radians(self.params['rx']), np.radians(self.params['ry']), np.radians(self.params['rz'])
        center = self.tx_inicial.GetFixedParameters()[:3]
        user_tx = sitk.Euler3DTransform(center, rad_x, rad_y, rad_z, (self.params['tx'], self.params['ty'], self.params['tz']))
        self.final_manual_transformation = sitk.CompositeTransform([self.tx_inicial, user_tx])
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.mri); resampler.SetTransform(self.final_manual_transformation)
        resampler.SetDefaultPixelValue(-1000); resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        c_arr = sitk.GetArrayFromImage(resampler.Execute(self.ct))
        self.c_mask = np.where(c_arr > 200, 1.0, 0.0)

    def draw_initial(self):
        self.recalculate_transformation(); self.draw_views()

    def update_tx(self, val):
        for k, s in zip(['rx','ry','rz','tx','ty','tz'], [self.s_rx, self.s_ry, self.s_rz, self.s_tx, self.s_ty, self.s_tz]): self.params[k] = s.val
        self.recalculate_transformation(); self.draw_views()
    def update_nav(self, val):
        self.idx_z, self.idx_y, self.idx_x = int(self.s_nz.val), int(self.s_ny.val), int(self.s_nx.val); self.draw_views()
    def draw_views(self):
        self.ax1.clear(); self.ax1.imshow(np.flipud(self.m_arr[self.idx_z, :, :]), cmap='gray')
        self.ax1.imshow(np.flipud(self.c_mask[self.idx_z, :, :]), cmap='Greens', alpha=0.5, vmin=0, vmax=1)
        self.ax1.set_title(f"AXIAL (Z={self.idx_z})"); self.ax1.axis('off')

        self.ax2.clear(); self.ax2.imshow(np.flipud(self.m_arr[:, self.idx_y, :]), cmap='gray')
        self.ax2.imshow(np.flipud(self.c_mask[:, self.idx_y, :]), cmap='Greens', alpha=0.5, vmin=0, vmax=1)
        self.ax2.set_title(f"CORONAL (Y={self.idx_y})"); self.ax2.axis('off')

        self.ax3.clear(); self.ax3.imshow(np.flipud(self.m_arr[:, :, self.idx_x]), cmap='gray')
        self.ax3.imshow(np.flipud(self.c_mask[:, :, self.idx_x]), cmap='Greens', alpha=0.5, vmin=0, vmax=1)
        self.ax3.set_title(f"SAGITAL (X={self.idx_x})"); self.ax3.axis('off')
        self.fig.canvas.draw_idle()

    def close(self, event): plt.close(self.fig)

class FinalVisorInteractiu:
    def __init__(self, mri, ct_reg, dice_val):
        self.mri_arr = sitk.GetArrayFromImage(mri)
        self.ct_arr = sitk.GetArrayFromImage(ct_reg)
        
        # Generem màscares per a visualització (mateixes que el Dice)
        self.mri_mask = sitk.GetArrayFromImage(sitk.OtsuThreshold(mri, 0, 1))
        self.ct_mask = np.where(self.ct_arr > 200, 1, 0)
        
        self.z, self.y, self.x = self.mri_arr.shape
        self.idx = self.z // 2
        
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 8))
        self.fig.canvas.manager.set_window_title(f'Validació Co-registre (DICE: {dice_val:.4f})')
        plt.subplots_adjust(bottom=0.15)

        # Mantenim el slider per visualització, però connectem l'scroll
        self.slider = Slider(plt.axes([0.2, 0.05, 0.6, 0.03]), 'Tall Z', 0, self.z-1, valinit=self.idx, valstep=1)
        self.slider.on_changed(self.update_from_slider)
        
        # Connectem l'esdeveniment de la roda del ratolí
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Inicialitzem els gràfics (pre-allocation per velocitat)
        self.im1 = self.ax1.imshow(np.flipud(self.mri_arr[self.idx]), cmap='gray')
        self.ov1 = self.ax1.imshow(np.ma.masked_where(np.flipud(self.mri_mask[self.idx])==0, np.ones_like(self.mri_arr[0])), cmap='winter', alpha=0.3)
        
        self.im2 = self.ax2.imshow(np.flipud(self.ct_arr[self.idx]), cmap='gray', vmin=-200, vmax=1000)
        self.ov2 = self.ax2.imshow(np.ma.masked_where(np.flipud(self.ct_mask[self.idx])==0, np.ones_like(self.ct_arr[0])), cmap='autumn', alpha=0.3)
        
        self.im3_m = self.ax3.imshow(np.flipud(self.mri_mask[self.idx]), cmap='Blues', alpha=0.7)
        self.im3_c = self.ax3.imshow(np.flipud(self.ct_mask[self.idx]), cmap='Oranges', alpha=0.5)

        for ax in [self.ax1, self.ax2, self.ax3]: ax.axis('off')
        self.update_display()

    def on_scroll(self, event):
        if event.button == 'up':
            self.idx = min(self.idx + 1, self.z - 1)
        elif event.button == 'down':
            self.idx = max(self.idx - 1, 0)
        
        # Sincronitzem el slider visualment i actualitzem
        self.slider.set_val(self.idx)

    def update_from_slider(self, val):
        self.idx = int(val)
        self.update_display()

    def update_display(self):
        # Fem servir set_data per evitar el "lag"
        self.im1.set_data(np.flipud(self.mri_arr[self.idx]))
        self.ov1.set_data(np.ma.masked_where(np.flipud(self.mri_mask[self.idx])==0, np.ones_like(self.mri_arr[0])))
        
        self.im2.set_data(np.flipud(self.ct_arr[self.idx]))
        self.ov2.set_data(np.ma.masked_where(np.flipud(self.ct_mask[self.idx])==0, np.ones_like(self.ct_arr[0])))
        
        self.im3_m.set_data(np.flipud(self.mri_mask[self.idx]))
        self.im3_c.set_data(np.flipud(self.ct_mask[self.idx]))
        
        self.ax1.set_title(f"MRI + Màscara (Tall {self.idx})")
        self.fig.canvas.draw_idle()

# --- EXECUCIÓ PRINCIPAL ---
if __name__ == "__main__":
    plt.close('all')
    mri_path = select_file("Selecciona MRI"); ct_path = select_file("Selecciona CT")
    if not mri_path or not ct_path: sys.exit("Fitxers no seleccionats.")

    mri_obj = upload_image(mri_path, sitk.sitkFloat32); ct_obj = upload_image(ct_path, sitk.sitkFloat32)
    
    VisorRaw(mri_obj, ct_obj); plt.show()
        
    crop_tool = CuttingTool(ct_obj); plt.show()
    z_cut = crop_tool.actual_cut
    ct_crop_obj = sitk.RegionOfInterest(ct_obj, [ct_obj.GetSize()[0], ct_obj.GetSize()[1], ct_obj.GetSize()[2]-z_cut], [0,0,z_cut])

    print("\n--- PAS 3: AJUST GEOMÈTRIC ---")
    opcio = input("Vols fer un ajust MANUAL abans de l'automàtic? (s/n): ").lower()
    tx_geometry = sitk.CenteredTransformInitializer(mri_obj, ct_crop_obj, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    if opcio == 's':
        cockpit = CockpitMultiplanar(mri_obj, ct_crop_obj, tx_geometry); plt.show()
        tx_inicial = cockpit.final_manual_transformation
    else:
        tx_inicial = tx_geometry

    print("\n--- PAS 4: REGISTRE AUTOMÀTIC ---")
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(50)
    R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.1)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1, minStep=0.001, numberOfIterations=50)
    R.SetInitialTransform(tx_inicial, inPlace=False); R.SetInterpolator(sitk.sitkLinear)
    
    tx_final = R.Execute(mri_obj, ct_crop_obj)
    ct_final = sitk.Resample(ct_crop_obj, mri_obj, tx_final, sitk.sitkLinear, -1000.0)
    
    val_dice = calcular_dice_global(mri_obj, ct_final)
    print(f"DICE FINAL: {val_dice:.4f}")

    # Visor Final amb Scroll Optimitzat
    FinalVisorInteractiu(mri_obj, ct_final, val_dice); plt.show()

    # Guardar Resultat
    output_dir = os.path.dirname(ct_path)
    output_full_path = os.path.join(output_dir, "Registered_CT_Result.nii.gz")
    try:
        sitk.WriteImage(ct_final, output_full_path)
        print(f"Fitxer desat correctament a: {output_full_path}")
    except Exception as e: print(f"Error en desar: {e}")