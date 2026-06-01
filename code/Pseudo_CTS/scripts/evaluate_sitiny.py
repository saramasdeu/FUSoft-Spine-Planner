import os
import sys
import torch
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import structural_similarity as ssim_fn

# --- PATHS (server) ---
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
sitiny_repo = str(REPO_ROOT / "stinity_mr-to-pct")
DATA_DIR    = REPO_ROOT / "MRXFDG-PET-CT-MRI"
model_route = str(REPO_ROOT / "models" / "pretrained_net_final_20220825.pth")
OUT_DIR     = REPO_ROOT / "code" / "Pseudo_CTS" / "results" / "sitiny_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, sitiny_repo)
import monai.transforms as _mt
if not hasattr(_mt, "AddChannel"):
    class _AddChannel:
        def __call__(self, x):
            if hasattr(x, "unsqueeze"): return x.unsqueeze(0)
            return np.expand_dims(x, 0)
    _mt.AddChannel = _AddChannel

try:
    from utils.infer_funcs import do_mr_to_pct
    PREP_T1 = True
except Exception:
    from utils.infer_funcs_noants import do_mr_to_pct
    PREP_T1 = False

# --- FIND ALL PAIRS ---
pairs = []
for mri_path in sorted(DATA_DIR.glob("sub-*_T1w.nii.gz")):
    sid = mri_path.name.replace("_T1w.nii.gz", "")
    sid_nodash = sid.replace("sub-", "sub")
    ct_path = DATA_DIR / f"{sid_nodash}-ct_registered.nii.gz"
    if ct_path.exists():
        pairs.append((sid, str(mri_path), str(ct_path)))
    else:
        print(f"[SKIP] CT not found for {sid}")

if not pairs:
    sys.exit("No paired MRI/CT found in " + str(DATA_DIR))

print(f"\nFound {len(pairs)} pairs. Processing...\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if not os.path.exists(model_route):
    sys.exit(f"Error: Model weights not found at {model_route}")
saved_model = torch.load(model_route, map_location=device)

# --- PROCESS EACH PAIR (identical logic to stinity_avaluator.py) ---
for sid, input_mr_file, real_ct_file in pairs:
    print("\n" + "="*60)
    print(f"Processing: {sid}")
    print("="*60)

    file_name = os.path.basename(input_mr_file)
    output_pct_file = str(OUT_DIR / f"{file_name.split('.')[0]}_sCT.nii.gz")

    # 1. Inference
    print("\n1. Generating Pseudo-CT from MRI...")
    do_mr_to_pct(input_mr_file, output_pct_file, saved_model, device, prep_t1=PREP_T1, plot_mrct=False)
    print(f"   Inference completed. File saved at: {output_pct_file}")

    # Load images
    pct_img = sitk.ReadImage(output_pct_file)
    pct_arr = sitk.GetArrayFromImage(pct_img)
    mri_arr = sitk.GetArrayFromImage(sitk.ReadImage(input_mr_file))
    num_slices    = pct_arr.shape[0]
    initial_slice = num_slices // 2

    # 2. Metrics
    print("\n2. Calculating comparison metrics (MAE, SSIM, Bones, Soft Tissue)...")
    real_ct_img = sitk.ReadImage(real_ct_file)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(pct_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)
    real_ct_resampled = resampler.Execute(real_ct_img)
    ct_arr = sitk.GetArrayFromImage(real_ct_resampled)

    mask = ct_arr > -500
    mae = 0.0
    value_ssim = 0.0
    if np.any(mask):
        mae = np.mean(np.abs(pct_arr[mask] - ct_arr[mask]))

        mask_soft_tissue = (ct_arr >= -500) & (ct_arr <= 300)
        mask_bone        = ct_arr > 300
        mae_soft = np.mean(np.abs(pct_arr[mask_soft_tissue] - ct_arr[mask_soft_tissue])) if np.any(mask_soft_tissue) else 0.0
        mae_bone = np.mean(np.abs(pct_arr[mask_bone]        - ct_arr[mask_bone]))        if np.any(mask_bone)        else 0.0

        data_range = float(np.max(ct_arr[mask]) - np.min(ct_arr[mask]))
        value_ssim = ssim_fn(ct_arr, pct_arr, data_range=data_range)

        print("\n" + "-"*45)
        print(f"{'MÈTRICA':<25} | {'VALOR':<15}")
        print("-"*45)
        print(f"{'MAE Global':<25} | {mae:.2f} HU")
        print(f"{'MAE Teixit Tou (Soft)':<25} | {mae_soft:.2f} HU")
        print(f"{'MAE Os (Bone)':<25} | {mae_bone:.2f} HU")
        print(f"{'SSIM Global':<25} | {value_ssim:.4f}")
        print("-"*45 + "\n")
    else:
        print("WARNING: Tissue mask is empty.")

    # 3. Save figure (identical layout, savefig instead of plt.show)
    print("3. Saving comparison figure...")
    error_map_3d = np.abs(pct_arr - ct_arr)
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    plt.subplots_adjust(bottom=0.2)
    fig.suptitle(f"Validation - {file_name} - MAE Global: {mae:.2f} HU | SSIM: {value_ssim:.3f}", fontsize=14)

    im1 = axes[0].imshow(mri_arr[initial_slice], cmap='gray');                      axes[0].set_title("1. Original MRI");          axes[0].axis('off')
    im2 = axes[1].imshow(ct_arr[initial_slice],  cmap='gray', vmin=-1000, vmax=1500); axes[1].set_title("2. Real CT (Resampled)");   axes[1].axis('off')
    im3 = axes[2].imshow(pct_arr[initial_slice], cmap='gray', vmin=-1000, vmax=1500); axes[2].set_title("3. Generated Pseudo-CT");   axes[2].axis('off')
    im4 = axes[3].imshow(error_map_3d[initial_slice], cmap='hot', vmin=0, vmax=500);  axes[3].set_title("4. Error Map");             axes[3].axis('off')
    fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    fig_path = OUT_DIR / f"{sid}_comparison.png"
    plt.tight_layout()
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Figure saved: {fig_path}")
