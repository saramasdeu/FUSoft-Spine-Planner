import SimpleITK as sitk
import matplotlib.pyplot as plt
<<<<<<< HEAD
from matplotlib.widgets import Slider, RadioButtons, Button
=======
from matplotlib.widgets import Slider, RadioButtons, Button, RectangleSelector
>>>>>>> fe36d020 (Clean repo without large files)
import numpy as np
import sys
import os
import subprocess
import platform
import tkinter as tk
from tkinter import filedialog

def select_file(titol):
    sistem = platform.system() 
    if sistem == 'Darwin': 
        # MAC: AppleScript per evitar el tancament inesperat
        apple_script = f'''
        try
            set theFile to choose file with prompt "{titol}"
            POSIX path of theFile
        on error
            return ""
        end try
        '''
        result = subprocess.run(['osascript', '-e', apple_script], capture_output=True, text=True)
        return result.stdout.strip()
        
    else:
        # WINDOWS / LINUX: Tkinter
        root = tk.Tk()
        root.withdraw() 
        root.attributes('-topmost', True)
        
        path = filedialog.askopenfilename(
            title=titol,
            filetypes=[("Medical Images", "*.nii *.nii.gz *.dcm *.mha"), ("Tots els arxius", "*.*")]
        )
        
        root.destroy()
        return path

# UTILITY FUNCTIONS
def upload_image(ruta, tipus_pixel=None):
    if not os.path.exists(ruta):
        print(f"ERROR: The file was not found at the path.: {ruta}")
        return None
    
    try:
        if tipus_pixel:
            return sitk.ReadImage(ruta, tipus_pixel)
        else:
            return sitk.ReadImage(ruta)
    except Exception as e:
        print(f"ERROR reading image '{os.path.basename(ruta)}'.")
        print(f"Technical details: {e}")
        return None
    
def calcular_dice_global(mri_img, ct_img):
    """
    Calcula el Coeficient Dice binaritzant la MRI i el CT per comprovar 
    la superposició global de l'anatomia (ignorant l'aire).
    """
    # 1. Binaritzar el CT (L'aire sol ser -1000, agafem tot el que sigui > -500)
    ct_mask = sitk.BinaryThreshold(ct_img, lowerThreshold=-500.0, upperThreshold=3000.0, insideValue=1, outsideValue=0)
    
    # 2. Binaritzar la MRI (Fem servir el mètode automàtic d'Otsu per separar fons d'anatomia)
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    mri_mask = otsu_filter.Execute(mri_img)
    
    # Igualem els tipus de dades perquè SimpleITK no es queixi
    ct_mask = sitk.Cast(ct_mask, sitk.sitkUInt8)
    mri_mask = sitk.Cast(mri_mask, sitk.sitkUInt8)
    
    # 3. Calculem la superposició (Dice)
    overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_filter.Execute(mri_mask, ct_mask)
    
    return overlap_filter.GetDiceCoefficient()

# GRAPHICAL INTERFACE TOOLS
class VisorRaw:
    def __init__(self, mri_img, ct_img):
        self.mri = sitk.GetArrayFromImage(mri_img)
        self.ct = sitk.GetArrayFromImage(ct_img)
        self.eix = 0
        self.pct = 0.5
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(bottom=0.25)
        
        self.slider = Slider(plt.axes([0.2, 0.1, 0.6, 0.03]), 'Position %', 0.0, 1.0, valinit=0.5)
        self.slider.on_changed(self.update)
        
        self.radio = RadioButtons(plt.axes([0.02, 0.4, 0.12, 0.15]), ('Axial (Z)', 'Coronal (Y)', 'Sagital (X)'))
        self.radio.on_clicked(self.change_view)
        
        self.fig.suptitle("INITIAL EXPLORATION\nCheck that the images are correct.")
        self.draw()

    def get_cut(self, volum, pct):
        maxim = volum.shape[self.eix]
        idx = int(pct * (maxim - 1))
        if self.eix == 0:   return volum[idx, :, :], idx
        elif self.eix == 1: return volum[:, idx, :], idx
        else:               return volum[:, :, idx], idx

    def update(self, val):
        self.pct = val
        self.draw()

    def change_view(self, label):
        if 'Axial' in label: self.eix = 0
        elif 'Coronal' in label: self.eix = 1
        else: self.eix = 2
        self.draw()

    def draw(self):
        mri_slide, idx_m = self.get_cut(self.mri, self.pct)
        ct_slide, idx_c  = self.get_cut(self.ct, self.pct)
        
        self.ax1.clear()
        self.ax1.imshow(np.flipud(mri_slide), cmap='gray')
        self.ax1.set_title(f"Original MRI (Slide: {idx_m})")
        self.ax1.axis('off')
        
        self.ax2.clear()
        self.ax2.imshow(np.flipud(ct_slide), cmap='gray', vmin=-200, vmax=1000)
        self.ax2.set_title(f"Original CT (Slide: {idx_c})")
        self.ax2.axis('off')
        
        self.fig.canvas.draw_idle()

# 1. CUTTING TOOL
class CuttingTool:
    def __init__(self, ct_image):
        self.ct = ct_image
        self.arr = sitk.GetArrayFromImage(ct_image)
        self.dim_z, self.dim_y, self.dim_x = self.arr.shape
<<<<<<< HEAD
        self.actual_cut = int(self.dim_z * 0.4) 
        self.actual_slide = self.dim_y // 2

        self.fig, self.ax = plt.subplots(figsize=(8, 9))
        plt.subplots_adjust(bottom=0.3)
        
        self.s_nav = Slider(plt.axes([0.2, 0.15, 0.6, 0.03]), 'NAVIGATE (Y)', 0, self.dim_y-1, valinit=self.actual_slide, valstep=1)
        self.s_cut = Slider(plt.axes([0.2, 0.10, 0.6, 0.03]), 'CUT (Z)', 0, self.dim_z-1, valinit=self.actual_cut, valstep=1)
        
        self.s_nav.on_changed(self.update)
        self.s_cut.on_changed(self.update)
        
        btn_ax = plt.axes([0.4, 0.02, 0.2, 0.05])
        self.btn = Button(btn_ax, 'CONFIRM CUT')
        self.btn.on_clicked(self.close)

        self.draw()

    def update(self, val):
        self.actual_slide = int(self.s_nav.val)
        self.actual_cut = int(self.s_cut.val)
        self.draw()

    def draw(self):
        vista = self.arr[:, self.actual_slide, :]
        self.ax.clear()
        self.ax.imshow(np.flipud(vista), cmap='gray', vmin=-200, vmax=1000)
        
        y_line = self.dim_z - self.actual_cut
        self.ax.axhline(y=y_line, color='red', linewidth=2, linestyle='--')
        self.ax.set_title(f"Slide Y: {self.actual_slide}")
=======
        
        self.slice_idx = self.dim_z // 2
        self.roi = None
        
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.2)
        
        # Slider per navegar en Z
        self.slider = Slider(
            plt.axes([0.2, 0.1, 0.6, 0.03]),
            'Slice Z', 0, self.dim_z-1,
            valinit=self.slice_idx, valstep=1
        )
        self.slider.on_changed(self.update)
        
        # Botó confirmar
        btn_ax = plt.axes([0.4, 0.02, 0.2, 0.05])
        self.btn = Button(btn_ax, 'CONFIRM ROI')
        self.btn.on_clicked(self.close)
        
        # Selector de rectangle
        self.selector = RectangleSelector(
            self.ax,
            self.on_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            interactive=True
        )
        
        self.draw()

    def on_select(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        self.roi = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
        print(f"ROI seleccionat: {self.roi}")

    def update(self, val):
        self.slice_idx = int(self.slider.val)
        self.draw()

    def draw(self):
        self.ax.clear()
        slice_img = self.arr[self.slice_idx, :, :]
        self.ax.imshow(np.flipud(slice_img), cmap='gray', vmin=-200, vmax=1000)
        self.ax.set_title(f"Slice Z: {self.slice_idx}")
>>>>>>> fe36d020 (Clean repo without large files)
        self.ax.axis('off')
        self.fig.canvas.draw_idle()

    def close(self, event):
<<<<<<< HEAD
        self.fig.clf()
=======
        if self.roi is None:
            print("ERROR: No has seleccionat cap ROI!")
>>>>>>> fe36d020 (Clean repo without large files)
        plt.close(self.fig)

# 2. MULTIPLANAR TOOL
class CockpitMultiplanar:
    def __init__(self, mri, ct_crop, tx_geo):
        self.mri = mri
        self.ct = ct_crop
        self.tx_inicial = tx_geo
        
        self.m_arr = sitk.GetArrayFromImage(self.mri)
        self.dim_z, self.dim_y, self.dim_x = self.m_arr.shape
        
        self.idx_z = self.dim_z // 2
        self.idx_y = self.dim_y // 2
        self.idx_x = self.dim_x // 2
        
        self.params = {'rx':0, 'ry':0, 'rz':0, 'tx':0, 'ty':0, 'tz':0}
        
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

        for s in [self.s_rx, self.s_ry, self.s_rz, self.s_tx, self.s_ty, self.s_tz]:
            s.on_changed(self.update_tx)
            
        for s in [self.s_nz, self.s_ny, self.s_nx]:
            s.on_changed(self.update_nav)

        self.btn = Button(plt.axes([0.4, 0.05, 0.2, 0.05]), 'REGISTER NOW')
        self.btn.on_clicked(self.close)
        self.fig.suptitle("NAVIGABLE COCKPIT\nAlign images and move browsers to check")
        
        self.c_mask = np.zeros_like(self.m_arr)
        self.draw_initial()

    def update_tx(self, val):
        self.params['rx'] = self.s_rx.val; self.params['ry'] = self.s_ry.val; self.params['rz'] = self.s_rz.val
        self.params['tx'] = self.s_tx.val; self.params['ty'] = self.s_ty.val; self.params['tz'] = self.s_tz.val
        self.recalculate_transformation()
        self.update_views()

    def update_nav(self, val):
        self.idx_z = int(self.s_nz.val)
        self.idx_y = int(self.s_ny.val)
        self.idx_x = int(self.s_nx.val)
        self.update_views()

    def recalculate_transformation(self):
        rad_x, rad_y, rad_z = np.radians(self.params['rx']), np.radians(self.params['ry']), np.radians(self.params['rz'])
        center = self.tx_inicial.GetFixedParameters()[:3]
        user_tx = sitk.Euler3DTransform(center, rad_x, rad_y, rad_z, (self.params['tx'], self.params['ty'], self.params['tz']))
        composite_tx = sitk.CompositeTransform([self.tx_inicial, user_tx])
        self.final_manual_transformation = composite_tx

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.mri)
        resampler.SetTransform(composite_tx)
        resampler.SetDefaultPixelValue(-1000)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        ct_mogut = resampler.Execute(self.ct)
        c_arr = sitk.GetArrayFromImage(ct_mogut)
        
        self.c_mask = np.where(c_arr > 200, 1.0, 0.0)

    def draw_initial(self):
        self.recalculate_transformation()

        self.im1_mri = self.ax1.imshow(np.flipud(self.m_arr[self.idx_z, :, :]), cmap='gray')
        self.im1_ct = self.ax1.imshow(np.flipud(self.c_mask[self.idx_z, :, :]), cmap='Greens', alpha=0.5, vmin=0, vmax=1)
        self.ax1.axis('off'); self.ax1.set_title("AXIAL")

        self.im2_mri = self.ax2.imshow(np.flipud(self.m_arr[:, self.idx_y, :]), cmap='gray')
        self.im2_ct = self.ax2.imshow(np.flipud(self.c_mask[:, self.idx_y, :]), cmap='Greens', alpha=0.5, vmin=0, vmax=1)
        self.ax2.axis('off'); self.ax2.set_title("CORONAL")

        self.im3_mri = self.ax3.imshow(np.flipud(self.m_arr[:, :, self.idx_x]), cmap='gray')
        self.im3_ct = self.ax3.imshow(np.flipud(self.c_mask[:, :, self.idx_x]), cmap='Greens', alpha=0.5, vmin=0, vmax=1)
        self.ax3.axis('off'); self.ax3.set_title("SAGITAL")

    def update_views(self):
        self.im1_mri.set_data(np.flipud(self.m_arr[self.idx_z, :, :]))
        self.im1_ct.set_data(np.flipud(self.c_mask[self.idx_z, :, :]))

        self.im2_mri.set_data(np.flipud(self.m_arr[:, self.idx_y, :]))
        self.im2_ct.set_data(np.flipud(self.c_mask[:, self.idx_y, :]))

        self.im3_mri.set_data(np.flipud(self.m_arr[:, :, self.idx_x]))
        self.im3_ct.set_data(np.flipud(self.c_mask[:, :, self.idx_x]))

        self.fig.canvas.draw_idle()

    def close(self, event):
        plt.close(self.fig)

# 3. FINAL VIEWER
class FinalVisor:
    def __init__(self, mri, ct_reg):
        self.mri = sitk.GetArrayFromImage(mri)
        self.ct = sitk.GetArrayFromImage(ct_reg)
        self.z, self.y, self.x = self.mri.shape
        self.pct = 0.5
        
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(bottom=0.2)
        
        self.s = Slider(plt.axes([0.2, 0.05, 0.6, 0.03]), 'Browse Synchronized', 0, 1, valinit=0.5, valstep=0.01)
        self.s.on_changed(self.upd)
        
        self.fig.suptitle("FINAL RESULT")
        
        iz, iy, ix = int(self.pct*(self.z-1)), int(self.pct*(self.y-1)), int(self.pct*(self.x-1))
        
        self.im1_base = self.ax1.imshow(np.flipud(self.mri[iz,:,:]), cmap='gray')
        self.im1_over = self.ax1.imshow(self.get_mask(self.ct[iz,:,:]), cmap='spring', alpha=0.6)
        self.ax1.axis('off'); self.ax1.set_title("Axial")

        self.im2_base = self.ax2.imshow(np.flipud(self.mri[:,iy,:]), cmap='gray')
        self.im2_over = self.ax2.imshow(self.get_mask(self.ct[:,iy,:]), cmap='spring', alpha=0.6)
        self.ax2.axis('off'); self.ax2.set_title("Coronal")

        self.im3_base = self.ax3.imshow(np.flipud(self.mri[:,:,ix]), cmap='gray')
        self.im3_over = self.ax3.imshow(self.get_mask(self.ct[:,:,ix]), cmap='spring', alpha=0.6)
        self.ax3.axis('off'); self.ax3.set_title("Sagital")

    def get_mask(self, slice_data):
        data = np.flipud(slice_data)
        return np.ma.masked_where(data < 200, data)

    def upd(self, val):
        self.pct = val
        iz, iy, ix = int(self.pct*(self.z-1)), int(self.pct*(self.y-1)), int(self.pct*(self.x-1))
        
        self.im1_base.set_data(np.flipud(self.mri[iz,:,:]))
        self.im2_base.set_data(np.flipud(self.mri[:,iy,:]))
        self.im3_base.set_data(np.flipud(self.mri[:,:,ix]))
        
        self.im1_over.set_data(self.get_mask(self.ct[iz,:,:]))
        self.im2_over.set_data(self.get_mask(self.ct[:,iy,:]))
        self.im3_over.set_data(self.get_mask(self.ct[:,:,ix]))
        
        self.fig.canvas.draw_idle()


# MAIN PROGRAM EXECUTION
if __name__ == "__main__":
    plt.close('all') 
    
    # FILE SELECTION
    print("\n--- 0. START OF CO-REGISTRATION ---")
    print("Opening the search engine for MRI...")
    mri_path = select_file("Select the MRI File (.nii / .dcm)")
    
    if not mri_path:
        print("Error: You have not selected any MRI. Exiting the program.")
        sys.exit()

    print("Opening the search engine for the CT...")
    ct_path = select_file("Select the CT File (.nii / .dcm)")
    
    if not ct_path:
        print("Error: You have not selected any CT. Exiting the program.")
        sys.exit()

    print(f" MRI selected: {mri_path}")
    print(f" CT selected:   {ct_path}")
    
    BASE_DIR = os.path.dirname(ct_path)

    #  STEP 1: LOAD AND INITIAL EXPLORATION
    print("\n--- 1. UPLOAD IMAGES AND EXPLORE ---")
    mri_obj = upload_image(mri_path, sitk.sitkFloat32)
    ct_obj = upload_image(ct_path, sitk.sitkFloat32)
    
    if mri_obj is None or ct_obj is None:
        sys.exit("Error loading images. Please check that the files are not corrupt..")

    print("Opening the initial image explorer...")
    initial_visor = VisorRaw(mri_obj, ct_obj)
    plt.show()
        
    # STEP 2: CUT
    print("\n--- 2. CT CUT ---")
    print("Opening snipping tool...")
    crop_tool = CuttingTool(ct_obj)
    plt.show()
<<<<<<< HEAD
    
    z_final_cut = crop_tool.actual_cut
    plt.close(crop_tool.fig)
    plt.close('all')
    plt.pause(0.1)
    
    roi_filter = sitk.RegionOfInterestImageFilter()
    size_ct = ct_obj.GetSize()
    z_final_cut = max(0, min(z_final_cut, size_ct[2]-1))
    roi_filter.SetIndex([0,0,z_final_cut])
    roi_filter.SetSize([size_ct[0], size_ct[1], size_ct[2]-z_final_cut])
    ct_crop_obj = roi_filter.Execute(ct_obj)

=======

    roi = crop_tool.roi
    if roi is None:
        sys.exit("Error: No ROI selected.")

    x1, y1, x2, y2 = roi

    size_ct = ct_obj.GetSize()

    # Conversió coordenades (compte amb flip vertical)
    start_x = x1
    start_y = size_ct[1] - y2
    start_z = 0

    size_x = x2 - x1
    size_y = y2 - y1
    size_z = size_ct[2]

    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetIndex([start_x, start_y, start_z])
    roi_filter.SetSize([size_x, size_y, size_z])

    ct_crop_obj = roi_filter.Execute(ct_obj)
    
>>>>>>> fe36d020 (Clean repo without large files)
    # STEP 3: MANUAL COCKPIT 
    print("\n--- 3. 3D MANUAL ALIGNMENT (BROWSABLE) ---")
    tx_geometry = sitk.CenteredTransformInitializer(mri_obj, ct_crop_obj, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    cockpit_tool = CockpitMultiplanar(mri_obj, ct_crop_obj, tx_geometry)
    plt.show() 

    #tx_manual_final = tx_geometry #Per fer la prova automatic vs manual

    tx_manual_final = cockpit_tool.final_manual_transformation
    plt.close('all')
<<<<<<< HEAD
    plt.pause(0.1)
=======
    #plt.pause(0.1)
>>>>>>> fe36d020 (Clean repo without large files)


    # STEP 4: AUTOMATIC REGISTRATION
    print("\n--- 4. AUTOMATIC REGISTRATION (MICRO-ADJUSTMENT) ---")
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.1)
    
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.1,      
        minStep=0.001,         
        numberOfIterations=30  
    )
    
    R.SetInitialTransform(tx_manual_final, inPlace=False)
    R.SetInterpolator(sitk.sitkLinear)
    
    tx_final_obj = R.Execute(mri_obj, ct_crop_obj)
    print(f"   -> Final Error: {R.GetMetricValue():.4f}")
    
    print("   Generating final image...")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(mri_obj)
    resampler.SetTransform(tx_final_obj)
    resampler.SetDefaultPixelValue(-1000)
    ct_final_obj = resampler.Execute(ct_crop_obj)
    
    print("\n   Calculant Coeficient de Similitud de Dice (DSC)...")
    valor_dice = calcular_dice_global(mri_obj, ct_final_obj)
    print(f"   -> DICE GLOBAL: {valor_dice:.4f} (Màxim ideal: 1.0)")

    # STEP 5: FINAL VIEWER
    print("\n--- 5. OPENING FLUID VISOR ---")
    final_visor = FinalVisor(mri_obj, ct_final_obj)
    plt.show() 

    os._exit(0)