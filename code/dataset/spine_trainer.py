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
# ==========================================================
class ShuffleUNet(UNet):
    def __init__(self, dimensions, in_channels, out_channels, channels, strides,
                 kernel_size=3, up_kernel_size=3, num_res_units=2,
                 act=Act.PRELU, norm=Norm.INSTANCE):

        super().__init__(
            spatial_dims=dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            num_res_units=num_res_units,
            act=act,
            norm=norm
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
        up = self._get_up_layer(upc, outc, s, is_top)

        return nn.Sequential(down, SkipConnection(subblock), up)

    def _get_up_layer(self, in_channels, out_channels, strides, is_top):
        decode = nn.Sequential()

        decode.add_module(
            'shuffle',
            SubpixelUpsample(
                dimensions=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                scale_factor=strides
            )
        )

        if is_top:
            decode.add_module(
                'conv',
                Convolution(
                    dimensions=self.dimensions,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=self.up_kernel_size,
                    act=None,
                    norm=None,
                    conv_only=True
                )
            )
            decode.add_module('tanh', nn.Tanh())
        else:
            decode.add_module(
                'conv',
                Convolution(
                    dimensions=self.dimensions,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=self.up_kernel_size,
                    act=Act.PRELU,
                    norm=Norm.INSTANCE
                )
            )

        return decode


# ==========================================================
# 2. CONFIG
# ==========================================================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

PATCH_DIR = "/Users/saramasdeusans/Desktop/TRAIN_DATASET_PATCHES"
PESOS_PATH = "/Users/saramasdeusans/Desktop/TFG_FUSOFT/code/stinity_mr-to-pct/best_net_G.pth"

augment = tio.Compose([
    tio.RandomFlip(axes=(0,)),
    tio.RandomNoise(std=(0, 0.05)),
    tio.RandomAffine(scales=(0.95, 1.05), degrees=5),
    tio.CropOrPad((128, 128, 64))
])


# ==========================================================
# 3. DATASET
# ==========================================================
class SpineDataset(torch.utils.data.Dataset):
    def __init__(self, patch_dir, transform=None):
        self.files = sorted(glob.glob(os.path.join(patch_dir, "*_mri.npy")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = os.path.basename(self.files[idx]).replace("_mri.npy", "")

        mri = torch.from_numpy(np.load(self.files[idx])).float().unsqueeze(0)
        ct = torch.from_numpy(np.load(os.path.join(PATCH_DIR, f"{name}_ct.npy"))).float().unsqueeze(0)

        subject = tio.Subject(
            mri=tio.ScalarImage(tensor=mri),
            ct=tio.ScalarImage(tensor=ct)
        )

        if self.transform:
            subject = self.transform(subject)

        return subject.mri.data, subject.ct.data


# ==========================================================
# 4. MODEL INIT
# ==========================================================
model = ShuffleUNet(
    dimensions=3,
    in_channels=1,
    out_channels=1,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2)
).to(DEVICE)

# 🔍 SHAPE CHECK
x = torch.randn(1, 1, 128, 128, 64).to(DEVICE)
with torch.no_grad():
    y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)


# ==========================================================
# 5. LOAD WEIGHTS
# ==========================================================
print(f"🔄 Carregant pesos des de {PESOS_PATH}...")
try:
    checkpoint = torch.load("fine_tuned_spine_epoch_20.pth",map_location=DEVICE)
except Exception as e:
    print(f"⚠️ Error carregant pesos: {e}")


# ==========================================================
# 6. TRAIN SETUP
# ==========================================================
train_loader = DataLoader(
    SpineDataset(PATCH_DIR, transform=augment),
    batch_size=2,
    shuffle=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)

criterion = nn.L1Loss()
ssim_loss = SSIMLoss(spatial_dims=3)


# ==========================================================
# 7. TRAIN LOOP
# ==========================================================
print("\n🚀 Entrenant...")

start_epoch = 20
num_epochs = 50
for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss = 0

    for mri, ct in train_loader:
        mri, ct = mri.to(DEVICE), ct.to(DEVICE)

        optimizer.zero_grad()

        output = model(mri)

        l1 = criterion(output, ct)

        output_ssim = (output + 1) / 2
        ct_ssim = (ct + 1) / 2

        data_range = torch.tensor(1.0, device=output.device)
        ssim = ssim_loss(output_ssim, ct_ssim, data_range=data_range)

        loss = 0.8 * l1 + 0.2 * (1 - ssim) # x 10 ambos

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/50] - Loss: {epoch_loss/len(train_loader):.4f}")
    print(f"L1: {l1.item():.4f} | SSIM: {ssim.item():.4f}")

    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), f"fine_tuned_spine_epoch_{epoch+1}.pth")
        print(f"💾 Guardat epoch {epoch+1}")