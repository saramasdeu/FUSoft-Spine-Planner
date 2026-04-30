import os
import torch
import torchio as tio
import numpy as np
import SimpleITK as sitk
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.networks.blocks import SubpixelUpsample, Convolution
from monai.networks.layers import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
import torch.nn as nn 

# ==========================================================
# 1. RUTES I CONFIGURACIÓ
# ==========================================================
MRI_PATH = "/Users/saramasdeusans/Desktop/sub-0003_T1w cropped.nii.gz"
MODEL_WEIGHTS = "/Users/saramasdeusans/Desktop/TFG_FUSOFT/fine_tuned_spine_epoch_50.pth" 
OUTPUT_NAME = "Pseudo_CT_sub-0001_Resultat.nii.gz"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ==========================================================
# 2. ARQUITECTURA
# ==========================================================
class ShuffleUNet(UNet):
    def __init__(self, dimensions, in_channels, out_channels, channels, strides, 
                 kernel_size=3, up_kernel_size=3, num_res_units=2, act=Act.PRELU, 
                 norm=Norm.INSTANCE, dropout=0.0, bias=True):
        super().__init__(spatial_dims=dimensions, in_channels=in_channels, out_channels=out_channels,
                         channels=channels, strides=strides, kernel_size=kernel_size,
                         num_res_units=num_res_units, act=act, norm=norm, dropout=dropout, bias=bias)
        self.up_kernel_size = up_kernel_size
        self.model = self._create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _create_block(self, inc, outc, channels, strides, is_top):
        c = channels[0]; s = strides[0]
        if len(channels) > 2:
            subblock = self._create_block(c, c, channels[1:], strides[1:], False)
            upc = c * 2
        else:
            subblock = self._get_bottom_layer(c, channels[1]); upc = c + channels[1]
        down = self._get_down_layer(inc, c, s, is_top)
        up = self._get_up_layer(upc, outc, s, is_top)
        return nn.Sequential(down, SkipConnection(subblock), up)

    def _get_up_layer(self, in_channels, out_channels, strides, is_top):
        decode = nn.Sequential()
        shuffle = SubpixelUpsample(dimensions=self.dimensions, in_channels=in_channels, 
                                   out_channels=out_channels, scale_factor=strides)
        conv = Convolution(dimensions=self.dimensions, in_channels=out_channels, out_channels=out_channels,
                           strides=1, kernel_size=self.up_kernel_size, act=self.act, norm=self.norm,
                           dropout=self.dropout, conv_only=is_top and self.num_res_units == 0)
        decode.add_module('shuffle', shuffle); decode.add_module('conv', conv)
        return decode

# ==========================================================
# 3. FUNCIÓ DE NORMALITZACIÓ
# ==========================================================
def preprocess_mri(path):
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    
    mask = array > 0
    if np.any(mask):
        mean = array[mask].mean()
        std = array[mask].std()
        array = (array - mean) / (std + 1e-5)
    
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0) 
    return tensor, image

# ==========================================================
# 4. EXECUCIÓ DE LA INFERÈNCIA
# ==========================================================
print(f"🚀 Carregant model i pesos...")
model = ShuffleUNet(
    dimensions=3, in_channels=1, out_channels=1,
    channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=2
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
model.eval()

print(f"📦 Preprocessant MRI: {os.path.basename(MRI_PATH)}")
input_tensor, original_itk = preprocess_mri(MRI_PATH)

# --- NOU: Print stats entrada ---
print(f"📊 MRI Normalitzada: Min={input_tensor.min().item():.4f}, Max={input_tensor.max().item():.4f}")

input_tensor = input_tensor.to(DEVICE)

print("🧠 Generant Pseudo-CT (Sliding Window)...")
with torch.no_grad():
    output = sliding_window_inference(
        inputs=input_tensor, 
        roi_size=(64, 128, 128), 
        sw_batch_size=2, 
        predictor=model,
        overlap=0.25
    )
# --- POST-PROCESSAMENT ULTRA-ROBUST ---

# 1. Agafem el resultat (que ara amb la Tanh estarà entre -1 i 1)
output_array = output.cpu().numpy()[0, 0, :, :, :]

# 2. DESNORMALITZACIÓ EXACTA (Lògica inversa de creation_patches.py)
# Formula: HU = ((val + 1) / 2) * (max_hu - min_hu) + min_hu
pseudo_ct_hu = ((output_array + 1) / 2) * (2000 - (-1000)) + (-1000)

# 3. FILTRE DE SEGURETAT
pseudo_ct_hu = np.clip(pseudo_ct_hu, -1000, 2000)

print(f"📊 NOU RANG VISIBLE: Min={pseudo_ct_hu.min():.1f}, Max={pseudo_ct_hu.max():.1f}")

# 5. GUARDAR
result_itk = sitk.GetImageFromArray(pseudo_ct_hu.astype(np.float32))
result_itk.CopyInformation(original_itk)
sitk.WriteImage(result_itk, OUTPUT_NAME)

print("\n✅ Procés finalitzat!")