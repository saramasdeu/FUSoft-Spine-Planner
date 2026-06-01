"""
evaluate_vertebrae.py
---------------------
Evaluates pretrained vs fine-tuned model on a single vertebrae subject.
Uses CT_1.nii (uint8) and MRI_1.nii from Dataset vertebres format ok.

Usage:
    cd ~/FUSoft-Spine-Planner
    python code/dataset/evaluate_vertebrae.py
"""

import os, sys
import numpy as np
import torch
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import structural_similarity as ssim_fn

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SITINY_REPO = "/home/sara/FUSoft-Spine-Planner/stinity_mr-to-pct"
MRI_PATH    = "/home/sara/FUSoft-Spine-Planner/Dataset vertebres format ok/MRI_1.nii"
CT_PATH     = "/home/sara/FUSoft-Spine-Planner/Dataset vertebres format ok/CT_1.nii"
MODEL_PRE   = "/home/sara/FUSoft-Spine-Planner/models/pretrained_net_final_20220825.pth"
MODEL_FT    = "/home/sara/FUSoft-Spine-Planner/DATASET_NET/finetuned_models/finetuned_best.pth"
OUT_DIR     = Path("/home/sara/FUSoft-Spine-Planner/code/Pseudo_CTS/results/vertebrae_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
sys.path.insert(0, SITINY_REPO)

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

print(f"PREP_T1={PREP_T1}")

# ---------------------------------------------------------------------------
# METRICS
# The CT here is uint8 (0-255), not HU. We normalise both arrays to [0,1]
# for a fair comparison, and also report MAE in uint8 units.
# ---------------------------------------------------------------------------
def compute_metrics_uint8(pct_arr, ct_arr_u8):
    """Compare pCT output against uint8 CT after normalising both to [0,1]."""
    # Normalise pCT to [0,1] using its own range
    pct_n = (pct_arr - pct_arr.min()) / (pct_arr.max() - pct_arr.min() + 1e-8)
    ct_n  = ct_arr_u8.astype(np.float32) / 255.0

    mae   = np.mean(np.abs(pct_n - ct_n))
    dr    = float(ct_n.max() - ct_n.min())
    ssim  = ssim_fn(ct_n, pct_n, data_range=dr) if dr > 0 else 0.0
    return {"mae_norm": mae, "ssim": ssim}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    saved_pre = torch.load(MODEL_PRE, map_location=device)
    saved_ft  = torch.load(MODEL_FT,  map_location=device)

    # Load real CT (uint8)
    ct_img  = sitk.ReadImage(CT_PATH)
    ct_arr  = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    mri_arr = sitk.GetArrayFromImage(sitk.ReadImage(MRI_PATH))
    print(f"CT  shape: {ct_arr.shape}  range: [{ct_arr.min():.0f}, {ct_arr.max():.0f}]")
    print(f"MRI shape: {mri_arr.shape}")

    results = {}

    for label, saved_model in [("pretrained", saved_pre), ("finetuned", saved_ft)]:
        print(f"\n--- {label} ---")
        out_path = str(OUT_DIR / f"vertebrae_{label}_sCT.nii.gz")
        do_mr_to_pct(MRI_PATH, out_path, saved_model, device, prep_t1=PREP_T1, plot_mrct=False)

        pct_img = sitk.ReadImage(out_path)
        pct_arr = sitk.GetArrayFromImage(pct_img).astype(np.float32)
        print(f"  pCT range: [{pct_arr.min():.2f}, {pct_arr.max():.2f}]")

        # Resample CT to pCT space for fair comparison
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(pct_img)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        ct_resampled = sitk.GetArrayFromImage(resampler.Execute(ct_img)).astype(np.float32)

        m = compute_metrics_uint8(pct_arr, ct_resampled)
        results[label] = {"pct": pct_arr, "m": m}
        print(f"  MAE (normalised): {m['mae_norm']:.4f}")
        print(f"  SSIM            : {m['ssim']:.4f}")

    # Figure
    sl_mri = mri_arr.shape[0] // 2
    sl_pct = results["pretrained"]["pct"].shape[0] // 2

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    pre = results["pretrained"]
    ft  = results["finetuned"]
    fig.suptitle(
        f"Vertebrae subject 1 — "
        f"SSIM pre={pre['m']['ssim']:.3f}  ft={ft['m']['ssim']:.3f} | "
        f"MAE(norm) pre={pre['m']['mae_norm']:.4f}  ft={ft['m']['mae_norm']:.4f}",
        fontsize=11)

    axes[0].imshow(mri_arr[sl_mri],              cmap='gray');                axes[0].set_title("MRI");            axes[0].axis('off')
    axes[1].imshow(ct_arr[sl_mri],               cmap='gray');                axes[1].set_title("Real CT (uint8)"); axes[1].axis('off')
    axes[2].imshow(pre["pct"][sl_pct],            cmap='gray');                axes[2].set_title("pCT Pretrained"); axes[2].axis('off')
    axes[3].imshow(ft["pct"][sl_pct],             cmap='gray');                axes[3].set_title("pCT Fine-tuned"); axes[3].axis('off')

    fig_path = OUT_DIR / "vertebrae_comparison.png"
    plt.tight_layout()
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    print("\n--- SUMMARY ---")
    print(f"  {'Model':<15} {'MAE (norm)':>12} {'SSIM':>10}")
    print(f"  {'-'*38}")
    for k in ["pretrained", "finetuned"]:
        m = results[k]["m"]
        print(f"  {k:<15} {m['mae_norm']:>12.4f} {m['ssim']:>10.4f}")