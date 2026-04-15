import os
import glob
import numpy as np
import SimpleITK as sitk

# --- CONFIGURACIÓ DE RUTES ---
BASE_PATH = "/Users/saramasdeusans/Desktop/coregistration_and_segmentation"
OUTPUT_PATH = "/Users/saramasdeusans/Desktop/TRAIN_DATASET_PATCHES"
PATCH_SIZE = (128, 128, 64)  # Mida del cub (X, Y, Z) - Ajustable segons GPU
PATCHES_PER_PATIENT = 5      # Quants retalls fem per pacient

os.makedirs(OUTPUT_PATH, exist_ok=True)

def normalize_mri(image):
    """Normalització Z-score per a la MRI"""
    array = sitk.GetArrayFromImage(image)
    # Ignorem el fons (zero) per calcular la mitjana
    mask = array > 0
    mean = array[mask].mean()
    std = array[mask].std()
    array = (array - mean) / (std + 1e-5)
    return array

def normalize_ct(image):
    """Clipping i normalització lineal [-1, 1] per al CT"""
    array = sitk.GetArrayFromImage(image)
    # Fem clipping: focus en os (mínim aire -1000, màxim os dens 2000)
    array = np.clip(array, -1000, 2000)
    # Escalem a [-1, 1]
    array = (array - (-1000)) / (2000 - (-1000)) * 2 - 1
    return array

def extract_patches(mri_arr, ct_arr, mask_arr, patient_id):
    """Extrau patches centrats on hi ha segmentació"""
    # Busquem les coordenades on hi ha vèrtebra (mask > 0)
    coords = np.argwhere(mask_arr > 0)
    
    if len(coords) == 0:
        print(f"⚠️ Alerta: No s'ha trobat segmentació per a {patient_id}")
        return

    for i in range(PATCHES_PER_PATIENT):
        # Triem un punt aleatori de la segmentació per centrar el patch
        center = coords[np.random.choice(len(coords))]
        
        # Calculem els límits del cub
        z_start = max(0, center[0] - PATCH_SIZE[2]//2)
        y_start = max(0, center[1] - PATCH_SIZE[1]//2)
        x_start = max(0, center[2] - PATCH_SIZE[0]//2)
        
        # Ens assegurem de no sortir de la imatge
        z_end = min(mri_arr.shape[0], z_start + PATCH_SIZE[2])
        y_end = min(mri_arr.shape[1], y_start + PATCH_SIZE[1])
        x_end = min(mri_arr.shape[2], x_start + PATCH_SIZE[0])

        # Extraiem els talls
        mri_patch = mri_arr[z_start:z_end, y_start:y_end, x_start:x_end]
        ct_patch = ct_arr[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Padding si el patch ha quedat més petit per estar a la vora
        # (Això és necessari perquè la IA vol mides fixes)
        # ... (codi de padding simplificat per ara)

        # Guardem com a fitxer .npy (molt ràpid de llegir durant l'entrenament)
        np.save(os.path.join(OUTPUT_PATH, f"{patient_id}_patch_{i}_mri.npy"), mri_patch)
        np.save(os.path.join(OUTPUT_PATH, f"{patient_id}_patch_{i}_ct.npy"), ct_patch)

# --- BUCLE PRINCIPAL ---
patient_folders = sorted([f for f in os.listdir(BASE_PATH) if f.startswith('sub')])

for sub in patient_folders:
    print(f"Processing {sub}...")
    sub_path = os.path.join(BASE_PATH, sub)
    
    # Busquem els fitxers (usant wildcards per si varien una mica els noms)
    try:
        mri_file = glob.glob(os.path.join(sub_path, "*mri_reference.nii.gz"))[0]
        ct_file = glob.glob(os.path.join(sub_path, "*ct_registered.nii.gz"))[0]
        mask_file = glob.glob(os.path.join(sub_path, "*segmentation.seg.nrrd"))[0]
        
        # Carreguem
        mri_img = sitk.ReadImage(mri_file)
        ct_img = sitk.ReadImage(ct_file)
        mask_img = sitk.ReadImage(mask_file)
        
        # Normalitzem
        mri_norm = normalize_mri(mri_img)
        ct_norm = normalize_ct(ct_img)
        mask_arr = sitk.GetArrayFromImage(mask_img)
        
        # Extreiem patches
        extract_patches(mri_norm, ct_norm, mask_arr, sub)
        
    except IndexError:
        print(f"❌ Error: Falten fitxers a la carpeta {sub}. Saltant...")

print("\n✅ Dataset de patches generat correctament!")