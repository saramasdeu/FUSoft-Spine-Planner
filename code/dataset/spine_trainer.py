import os
import torch
import torch.nn as nn
import torchio as tio
from torch.utils.data import DataLoader
from monai.networks.nets import UNet
from monai.networks.blocks import SubpixelUpsample, Convolution
from monai.networks.layers import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.losses import SSIMLoss
import numpy as np
import glob

# ==========================================================
# 1. MODEL
#    Usem la nostra definició de ShuffleUNet amb Tanh a la sortida.
#    Tanh no té pesos propis, per tant load_state_dict funciona
#    perfectament amb els pesos pre-entrenats de sitiny.
#    Garanteix que la sortida estigui acotada a [-1, 1],
#    consistent amb els CT targets normalitzats.
# ==========================================================
class ShuffleUNet(UNet):
    def __init__(self, dimensions, in_channels, out_channels, channels, strides,
                 kernel_size=3, up_kernel_size=3, num_res_units=2,
                 act=Act.PRELU, norm=Norm.INSTANCE):
        super().__init__(
            spatial_dims=dimensions, in_channels=in_channels, out_channels=out_channels,
            channels=channels, strides=strides, kernel_size=kernel_size,
            num_res_units=num_res_units, act=act, norm=norm
        )
        self.up_kernel_size = up_kernel_size
        self.model = self._create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _create_block(self, inc, outc, channels, strides, is_top):
        c = channels[0]; s = strides[0]
        if len(channels) > 2:
            subblock = self._create_block(c, c, channels[1:], strides[1:], False)
            upc = c * 2
        else:
            subblock = self._get_bottom_layer(c, channels[1])
            upc = c + channels[1]
        down = self._get_down_layer(inc, c, s, is_top)
        up   = self._get_up_layer(upc, outc, s, is_top)
        return nn.Sequential(down, SkipConnection(subblock), up)

    def _get_up_layer(self, in_channels, out_channels, strides, is_top):
        decode = nn.Sequential()
        decode.add_module('shuffle', SubpixelUpsample(
            spatial_dims=self.dimensions, in_channels=in_channels,
            out_channels=out_channels, scale_factor=strides
        ))
        if is_top:
            decode.add_module('conv', Convolution(
                spatial_dims=self.dimensions, in_channels=out_channels,
                out_channels=out_channels, strides=1, kernel_size=self.up_kernel_size,
                act=None, norm=None, conv_only=True
            ))
            decode.add_module('tanh', nn.Tanh())   # acota sortida a [-1, 1]
        else:
            decode.add_module('conv', Convolution(
                spatial_dims=self.dimensions, in_channels=out_channels,
                out_channels=out_channels, strides=1, kernel_size=self.up_kernel_size,
                act=Act.PRELU, norm=Norm.INSTANCE
            ))
        return decode


# ==========================================================
# 2. CONFIG
# ==========================================================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"💻 Dispositiu: {DEVICE}")

PATCH_DIR   = "/Users/saramasdeusans/Desktop/TRAIN_DATASET_PATCHES"

# ✅ CORREGIT: fitxer correcte del model pre-entrenat de sitiny
PESOS_PRETRAINED = "/Users/saramasdeusans/Desktop/TFG_FUSOFT/models/pretrained_net_final_20220825.pth"

# Per continuar un fine-tuning anterior, posa la ruta aquí.
# Si vols començar des del model pre-entrenat, deixa-ho a None.
CHECKPOINT_RESUME = None   # tornem a partir del model pre-entrenat

START_EPOCH = 0
NUM_EPOCHS  = 50
SAVE_EVERY  = 10
BATCH_SIZE  = 2
LR          = 1e-4

augment = tio.Compose([
    # Transformacions geomètriques: s'apliquen igual a MRI i CT (correcte)
    tio.RandomFlip(axes=(0,)),
    tio.RandomAffine(scales=(0.95, 1.05), degrees=5),
    tio.CropOrPad((128, 128, 64)),
    # Transformacions d'intensitat: NOMÉS a la MRI, mai al CT target
    tio.RandomNoise(std=(0, 0.05), include=['mri']),
    tio.RandomBiasField(coefficients=(0, 0.1), include=['mri']),
])


# ==========================================================
# 3. DATASET
#    Els patches MRI estan guardats amb Z-score (creation_patches.py).
#    Els re-normalitzem a [-1, 1] per ser consistents amb com
#    el model pre-entrenat va ser entrenat (ScaleIntensity [-1,1]).
# ==========================================================
class SpineDataset(torch.utils.data.Dataset):
    def __init__(self, patch_dir, transform=None):
        self.patch_dir = patch_dir
        self.transform = transform
        self.files = sorted(glob.glob(os.path.join(patch_dir, "*_mri.npy")))
        if len(self.files) == 0:
            raise RuntimeError(f"Cap fitxer _mri.npy trobat a: {patch_dir}")
        print(f"✅ Dataset carregat: {len(self.files)} patches")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = os.path.basename(self.files[idx]).replace("_mri.npy", "")

        mri = np.load(self.files[idx]).astype(np.float32)
        ct  = np.load(os.path.join(self.patch_dir, f"{name}_ct.npy")).astype(np.float32)

        # Re-normalitzem MRI de Z-score a [-1, 1]
        # (el CT ja està guardat en [-1, 1] per creation_patches.py)
        mri_min, mri_max = mri.min(), mri.max()
        if mri_max > mri_min:
            mri = (mri - mri_min) / (mri_max - mri_min) * 2.0 - 1.0
        else:
            mri = np.zeros_like(mri)

        mri_t = torch.from_numpy(mri).float().unsqueeze(0)
        ct_t  = torch.from_numpy(ct).float().unsqueeze(0)

        subject = tio.Subject(
            mri=tio.ScalarImage(tensor=mri_t),
            ct=tio.ScalarImage(tensor=ct_t)
        )
        if self.transform:
            subject = self.transform(subject)

        return subject.mri.data, subject.ct.data


# ==========================================================
# 4. MODEL
# ==========================================================
model = ShuffleUNet(
    dimensions=3,
    in_channels=1,
    out_channels=1,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    kernel_size=3,
    up_kernel_size=3,
    num_res_units=2
).to(DEVICE)

# Comprovació de forma
with torch.no_grad():
    _x = torch.randn(1, 1, 128, 128, 64).to(DEVICE)
    _y = model(_x)
print(f"🔍 Shape check — Input: {_x.shape}  Output: {_y.shape}")


# ==========================================================
# 5. CÀRREGA DE PESOS
#    Ordre de prioritat:
#      1. CHECKPOINT_RESUME  (continuar fine-tuning anterior)
#      2. PESOS_PRETRAINED   (começar des del model de crani)
# ==========================================================
if CHECKPOINT_RESUME and os.path.exists(CHECKPOINT_RESUME):
    print(f"🔄 Reprenent fine-tuning des de: {CHECKPOINT_RESUME}")
    state = torch.load(CHECKPOINT_RESUME, map_location=DEVICE)
    model.load_state_dict(state)
    # Intentem extreure l'epoch de continuació del nom del fitxer
    try:
        START_EPOCH = int(CHECKPOINT_RESUME.split("epoch_")[-1].replace(".pth", ""))
        print(f"   Continuant des de l'epoch {START_EPOCH}")
    except Exception:
        pass

elif os.path.exists(PESOS_PRETRAINED):
    print(f"🧠 Carregant pesos pre-entrenats (model de crani): {PESOS_PRETRAINED}")
    # ✅ CORREGIT: ara sí que apliquem els pesos al model
    state = torch.load(PESOS_PRETRAINED, map_location=DEVICE)
    result = model.load_state_dict(state, strict=False)
    n_loaded  = len(state) - len(result.missing_keys)
    n_missing = len(result.missing_keys)
    n_unexpected = len(result.unexpected_keys)
    print(f"   Pesos carregats (strict=False):")
    print(f"     ✅ Compatibles:    {n_loaded}")
    print(f"     ⚠️  Falten (nous): {n_missing}  → s'inicialitzen aleatòriament")
    print(f"     ℹ️  Sobren (vells): {n_unexpected} → ignorats")
    print(f"   Fine-tuning sobre vèrtebres...")
    START_EPOCH = 0

else:
    print(f"⚠️  No s'ha trobat cap fitxer de pesos. Entrenant des de zero.")
    print(f"   Buscat: {PESOS_PRETRAINED}")


# ==========================================================
# 6. TRAINING SETUP
# ==========================================================
train_dataset = SpineDataset(PATCH_DIR, transform=augment)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
criterion  = nn.L1Loss()
ssim_loss  = SSIMLoss(spatial_dims=3, data_range=1.0)


# ==========================================================
# 7. TRAINING LOOP
# ==========================================================
# ==========================================================
# 8. VERIFICACIÓ DE RANGS DEL DATASET
# ==========================================================
print("\n🔎 Verificant rangs del dataset...")
_mri_sample, _ct_sample = train_dataset[0]
print(f"   MRI — min: {_mri_sample.min():.3f}  max: {_mri_sample.max():.3f}  (esperat: -1 a 1)")
print(f"   CT  — min: {_ct_sample.min():.3f}  max: {_ct_sample.max():.3f}  (esperat: -1 a 1)")
if _ct_sample.max() > 10 or _ct_sample.min() < -10:
    print("   ⚠️  CT FORA DE RANG! Els patches semblen estar en HU bruts, no normalitzats.")
    print("      Solució: regenerar patches amb creation_patches.py o normalitzar al dataset.")
else:
    print("   ✅ Rangs correctes.")

print(f"\n🚀 Iniciant entrenament des de l'epoch {START_EPOCH + 1} fins {NUM_EPOCHS}...\n")

for epoch in range(START_EPOCH, NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    last_l1, last_ssim = 0.0, 0.0

    for mri, ct in train_loader:
        mri, ct = mri.to(DEVICE), ct.to(DEVICE)

        optimizer.zero_grad()
        output = model(mri)

        l1 = criterion(output, ct)

        # SSIM necessita valors a [0, 1]
        out_ssim = (output + 1.0) / 2.0
        ct_ssim  = (ct     + 1.0) / 2.0
        ssim = ssim_loss(out_ssim, ct_ssim)

        loss = 0.8 * l1 + 0.2 * ssim

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        last_l1, last_ssim = l1.item(), ssim.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch + 1:3d}/{NUM_EPOCHS}]  Loss: {avg_loss:.4f}  "
          f"L1: {last_l1:.4f}  SSIM: {last_ssim:.4f}")

    if (epoch + 1) % SAVE_EVERY == 0:
        save_path = f"fine_tuned_spine_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"💾 Guardat: {save_path}\n")

print("\n✅ Entrenament finalitzat!")
