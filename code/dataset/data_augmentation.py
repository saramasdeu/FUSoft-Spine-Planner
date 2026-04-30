import os
import glob
import torch
import numpy as np
import torchio as tio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ==========================================================
# CONFIGURACIÓ DE RUTES
# ==========================================================
PATCH_DIR = "/Users/saramasdeusans/Desktop/TRAIN_DATASET_PATCHES"

# ==========================================================
# 1. DEFINICIÓ DEL DATASET (Amb correcció de mides)
# ==========================================================
class SpineDataset(Dataset):
    def __init__(self, patch_dir, transform=None):
        self.patch_dir = patch_dir
        self.transform = transform
        # Busquem totes les bases de patches (usant els de MRI com a guia)
        mri_files = sorted(glob.glob(os.path.join(patch_dir, "*_mri.npy")))
        self.patch_names = [os.path.basename(f).replace("_mri.npy", "") for f in mri_files]
        
        if len(self.patch_names) == 0:
            raise RuntimeError(f"No s'han trobat fitxers .npy a {patch_dir}")

    def __len__(self):
        return len(self.patch_names)

    def __getitem__(self, idx):
        name = self.patch_names[idx]
        
        # Carregar els fitxers de NumPy
        mri_arr = np.load(os.path.join(self.patch_dir, f"{name}_mri.npy"))
        ct_arr = np.load(os.path.join(self.patch_dir, f"{name}_ct.npy"))

        # Convertir a tensors de Torch [1, Z, Y, X]
        mri_tensor = torch.from_numpy(mri_arr).float().unsqueeze(0)
        ct_tensor = torch.from_numpy(ct_arr).float().unsqueeze(0)

        # Crear objecte Subject de TorchIO
        subject = tio.Subject(
            mri=tio.ScalarImage(tensor=mri_tensor),
            ct=tio.ScalarImage(tensor=ct_tensor)
        )

        # ARREGLAT: Forçar que tots els patches mesurin exactament 128x128x64
        # Això evita l'error de 'stack expects each tensor to be equal size'
        # Nota: L'ordre a TorchIO és (Anterior-Posterior, Right-Left, Superior-Inferior)
        fix_size = tio.CropOrPad((128, 128, 64), padding_mode='minimum')
        subject = fix_size(subject)

        # Aplicar Data Augmentation (Soroll, rotació, etc.)
        if self.transform:
            subject = self.transform(subject)

        return subject.mri.data, subject.ct.data

# ==========================================================
# 2. CONFIGURACIÓ DEL DATA AUGMENTATION (On-the-fly)
# ==========================================================
augment_pipeline = tio.Compose([
    tio.RandomFlip(axes=(0,)),                       # Inversió mirall esquerra-dreta
    tio.RandomNoise(std=(0, 0.05)),                  # Soroll Gaussià aleatori a la MRI
    tio.RandomBiasField(coefficients=(0, 0.1)),      # Simula variacions d'intensitat MRI (pujar)
    tio.RandomAffine(scales=(0.95, 1.05), degrees=5),# Rotacions de +-5 graus
])

# ==========================================================
# 3. CREACIÓ DEL DATALOADER I VISUALITZACIÓ
# ==========================================================
if __name__ == "__main__":
    # Creem el dataset i el loader
    print(f"🚀 Carregant dataset des de: {PATCH_DIR}")
    dataset = SpineDataset(PATCH_DIR, transform=augment_pipeline)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    try:
        # Intentem treure un "batch" per validar que no hi ha errors de mida
        print("🔄 Validant batching i transformacions...")
        mri_batch, ct_batch = next(iter(train_loader))
        print(f"✅ Batch generat correctament!")
        print(f"   Mida MRI Batch: {mri_batch.shape} (Batch, Canal, Z, Y, X)")
        print(f"   Mida CT Batch:  {ct_batch.shape}")

        # VISUALITZACIÓ DELS RESULTATS
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i in range(min(4, mri_batch.shape[0])):
            # Llesca central del cub (Z)
            slice_idx = mri_batch.shape[2] // 2
            
            # Mostrem MRI (Augmented)
            axes[0, i].imshow(mri_batch[i, 0, slice_idx, :, :], cmap='gray')
            axes[0, i].set_title(f"Augmented MRI {i}")
            axes[0, i].axis('off')
            
            # Mostrem CT (Target corresponent)
            axes[1, i].imshow(ct_batch[i, 0, slice_idx, :, :], cmap='gray', vmin=-1, vmax=1)
            axes[1, i].set_title(f"Target CT {i}")
            axes[1, i].axis('off')

        plt.suptitle("VALIDACIÓ: Patches amb Augmentation i Mida Corregida", fontsize=16)
        plt.show()

    except Exception as e:
        print(f"❌ Ha fallat la validació: {e}")