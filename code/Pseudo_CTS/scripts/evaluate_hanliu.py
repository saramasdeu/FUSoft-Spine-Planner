import os
import sys
import subprocess
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import structural_similarity as ssim_fn

# --- PATHS ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[2]
HANLIU_SRC = REPO_ROOT / "SynCT_TcMRgFUS" / "src"
DATA_DIR   = REPO_ROOT / "MRXFDG-PET-CT-MRI"
OUT_DIR    = REPO_ROOT / "code" / "Pseudo_CTS" / "results" / "hanliu_eval"
PCT_DIR    = OUT_DIR / "pseudo_cts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PCT_DIR.mkdir(exist_ok=True)

# --- FIND ALL PAIRS ---
pairs = []
for mri_path in sorted(DATA_DIR.glob("sub-*_T1w.nii.gz")):
    sid = mri_path.name.replace("_T1w.nii.gz", "")
    sid_nodash = sid.replace("sub-", "sub")
    ct_path = DATA_DIR / f"{sid_nodash}-ct_registered.nii.gz"
    if ct_path.exists():
        pairs.append((sid, mri_path, ct_path))
    else:
        print(f"[SKIP] CT not found for {sid}")

if not pairs:
    sys.exit("No paired MRI/CT found in " + str(DATA_DIR))

print(f"\nFound {len(pairs)} pairs.\n")

python_bin = sys.executable

for sid, mri_path, ct_path in pairs:
    print("\n" + "="*60)
    print(f"Processing: {sid}")
    print("="*60)

    output_pct_file = PCT_DIR / f"{sid}_hanliu_sCT.nii.gz"

    # 1. Inference via subprocess (avoids import conflicts)
    print("\n1. Generating Pseudo-CT from MRI...")

    # Create a temp input dir with just this subject's MRI
    tmp_input = OUT_DIR / "tmp_input"
    tmp_input.mkdir(exist_ok=True)
    link = tmp_input / mri_path.name
    if link.exists(): link.unlink()
    link.symlink_to(mri_path)

    result = subprocess.run(
        [python_bin, "run_inference.py",
         "--input_dir",     str(tmp_input),
         "--output_dir",    str(PCT_DIR),
         "--eval",
         "--overlap_ratio", "0.6"],
        cwd=str(HANLIU_SRC),
        capture_output=False,
    )
    link.unlink()

    # run_inference.py saves as {pid}_sCT.nii.gz where pid = filename without extension
    pid = mri_path.name.split('.')[0]
    generated = PCT_DIR / f"{pid}_sCT.nii.gz"
    if generated.exists() and generated != output_pct_file:
        generated.rename(output_pct_file)

    if not output_pct_file.exists():
        print(f"  [ERROR] pCT not generated for {sid}"); continue
    print(f"  pCT saved: {output_pct_file}")

    # Load arrays
    pct_img = sitk.ReadImage(str(output_pct_file))
    pct_arr = sitk.GetArrayFromImage(pct_img)
    mri_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mri_path)))
    num_slices    = pct_arr.shape[0]
    initial_slice = num_slices // 2

    # 2. Metrics (identical to stinity_avaluator.py)
    print("\n2. Calculating comparison metrics...")
    real_ct_img = sitk.ReadImage(str(ct_path))
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(pct_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)
    ct_arr = sitk.GetArrayFromImage(resampler.Execute(real_ct_img))

    mask = ct_arr > -500
    mae = 0.0
    value_ssim = 0.0
    if np.any(mask):
        mae      = np.mean(np.abs(pct_arr[mask] - ct_arr[mask]))
        mask_soft = (ct_arr >= -500) & (ct_arr <= 300)
        mask_bone = ct_arr > 300
        mae_soft = np.mean(np.abs(pct_arr[mask_soft] - ct_arr[mask_soft])) if np.any(mask_soft) else 0.0
        mae_bone = np.mean(np.abs(pct_arr[mask_bone] - ct_arr[mask_bone])) if np.any(mask_bone) else 0.0
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

    # 3. Save figure
    print("3. Saving comparison figure...")
    error_map = np.abs(pct_arr - ct_arr)
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    plt.subplots_adjust(bottom=0.2)
    fig.suptitle(f"Han-Liu - {sid} - MAE Global: {mae:.2f} HU | SSIM: {value_ssim:.3f}", fontsize=14)

    axes[0].imshow(mri_arr[initial_slice],  cmap='gray');                        axes[0].set_title("1. Original MRI");        axes[0].axis('off')
    axes[1].imshow(ct_arr[initial_slice],   cmap='gray', vmin=-1000, vmax=1500); axes[1].set_title("2. Real CT");              axes[1].axis('off')
    axes[2].imshow(pct_arr[initial_slice],  cmap='gray', vmin=-1000, vmax=1500); axes[2].set_title("3. Pseudo-CT Han-Liu");    axes[2].axis('off')
    im4 = axes[3].imshow(error_map[initial_slice], cmap='hot', vmin=0, vmax=500); axes[3].set_title("4. Error Map");           axes[3].axis('off')
    fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    fig_path = OUT_DIR / f"{sid}_comparison.png"
    plt.tight_layout()
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Figure saved: {fig_path}")

# Cleanup
tmp_input = OUT_DIR / "tmp_input"
if tmp_input.exists():
    tmp_input.rmdir()