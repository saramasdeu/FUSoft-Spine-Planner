"""
evaluate_finetuned.py
---------------------
Compares pretrained vs fine-tuned ShuffleUNet on all DB1 pairs.
Inference pipeline identical to stinity_avaluator.py (the working one).

Computes both global metrics AND vertebrae-region metrics (using either
segmentation files from SEGMENTATION_DIR, or a fallback spine-column mask
derived from the CT itself).

Usage:
    cd ~/FUSoft-Spine-Planner
    python code/dataset/evaluate_finetuned.py
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
SITINY_REPO     = "/home/sara/FUSoft-Spine-Planner/stinity_mr-to-pct"
DATA_DIR        = Path("/home/sara/FUSoft-Spine-Planner/MRXFDG-PET-CT-MRI")
SEGMENTATION_DIR = Path("/home/sara/FUSoft-Spine-Planner/coregistration_and_segmentation")
MODEL_PRE       = "/home/sara/FUSoft-Spine-Planner/models/pretrained_net_final_20220825.pth"
MODEL_FT        = "/home/sara/FUSoft-Spine-Planner/DATASET_NET/finetuned_models/finetuned_best.pth"
OUT_DIR         = Path("/home/sara/FUSoft-Spine-Planner/code/Pseudo_CTS/results/finetuned_eval")
TMP_DIR         = OUT_DIR / "tmp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# IMPORTS — identical to stinity_avaluator.py
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
# VERTEBRAE MASK HELPERS
# ---------------------------------------------------------------------------
def get_vertebrae_mask_from_ct(ct_arr):
    """Fallback: approximate spine column using central region + bone HU.
    Takes central 50% of axial FOV and returns bone voxels (HU > 200) there."""
    z, y, x  = ct_arr.shape
    x0, x1   = x // 4, 3 * x // 4   # central 50% laterally
    y0, y1   = y // 4, 3 * y // 4   # central 50% anterior-posterior
    mask      = np.zeros_like(ct_arr, dtype=bool)
    mask[:, y0:y1, x0:x1] = ct_arr[:, y0:y1, x0:x1] > 200
    return mask


def try_load_segmentation(sid, ct_img, ct_arr):
    """Try to load a vertebrae segmentation .nrrd for this subject.
    Returns binary mask resampled to ct_img space, or None if not found."""
    # Try to match sub-XXXX or subXXXX naming
    num   = "".join(filter(str.isdigit, sid))
    candidates = [
        SEGMENTATION_DIR / f"sub{num.zfill(4)}" / f"sub{num.zfill(4)}_segmentation.seg.nrrd",
        SEGMENTATION_DIR / f"sub-{num.zfill(4)}" / f"sub-{num.zfill(4)}_segmentation.seg.nrrd",
    ]
    for seg_path in candidates:
        if seg_path.exists():
            try:
                seg_img = sitk.ReadImage(str(seg_path))
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(ct_img)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetDefaultPixelValue(0)
                seg_resampled = sitk.GetArrayFromImage(resampler.Execute(seg_img))
                mask = seg_resampled > 0
                if np.any(mask):
                    print(f"    Segmentation mask loaded: {seg_path.name} ({mask.sum()} voxels)")
                    return mask
            except Exception as e:
                print(f"    [WARN] Could not load seg {seg_path}: {e}")
    return None


# ---------------------------------------------------------------------------
# METRICS — identical to stinity_avaluator.py
# ---------------------------------------------------------------------------
def _metrics_in_mask(pct_arr, ct_arr, mask):
    if not np.any(mask):
        return None
    mae       = float(np.mean(np.abs(pct_arr[mask] - ct_arr[mask])))
    mask_soft = mask & (ct_arr >= -500) & (ct_arr <= 300)
    mask_bone = mask & (ct_arr > 300)
    mae_soft  = float(np.mean(np.abs(pct_arr[mask_soft] - ct_arr[mask_soft]))) if np.any(mask_soft) else 0.0
    mae_bone  = float(np.mean(np.abs(pct_arr[mask_bone] - ct_arr[mask_bone]))) if np.any(mask_bone) else 0.0
    dr        = float(np.max(ct_arr[mask]) - np.min(ct_arr[mask]))
    value_ssim = ssim_fn(ct_arr, pct_arr, data_range=dr)
    return {"mae": mae, "mae_soft": mae_soft, "mae_bone": mae_bone, "ssim": value_ssim}


def compute_metrics(pct_file, real_ct_file, sid=""):
    pct_img  = sitk.ReadImage(pct_file)
    pct_arr  = sitk.GetArrayFromImage(pct_img)

    real_ct_img = sitk.ReadImage(real_ct_file)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(pct_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)
    ct_arr = sitk.GetArrayFromImage(resampler.Execute(real_ct_img))
    ct_img_rs = resampler.Execute(real_ct_img)  # resampled CT SimpleITK image

    mask_global = ct_arr > -500
    if not np.any(mask_global):
        return None, None, ct_arr, pct_arr

    metrics_global = _metrics_in_mask(pct_arr, ct_arr, mask_global)

    # Vertebrae mask: try segmentation file first, then fallback
    vert_mask = try_load_segmentation(sid, ct_img_rs, ct_arr)
    if vert_mask is None:
        vert_mask = get_vertebrae_mask_from_ct(ct_arr)
        print(f"    Using fallback spine-column mask ({vert_mask.sum()} voxels)")
    metrics_vert = _metrics_in_mask(pct_arr, ct_arr, vert_mask)

    return metrics_global, metrics_vert, ct_arr, pct_arr


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    saved_pre = torch.load(MODEL_PRE, map_location=device)
    saved_ft  = torch.load(MODEL_FT,  map_location=device)

    # Find pairs
    pairs = []
    for mri_path in sorted(DATA_DIR.glob("sub-*_T1w.nii.gz")):
        sid        = mri_path.name.replace("_T1w.nii.gz", "")
        sid_nodash = sid.replace("sub-", "sub")
        ct_path    = DATA_DIR / f"{sid_nodash}-ct_registered.nii.gz"
        if ct_path.exists():
            pairs.append((sid, str(mri_path), str(ct_path)))

    print(f"Found {len(pairs)} pairs.\n")
    results = []

    for sid, mri_path, ct_path in pairs:
        print("=" * 60)
        print(f"Processing: {sid}")
        print("=" * 60)

        # --- Pretrained ---
        pct_pre_path = str(TMP_DIR / f"{sid}_pre_sCT.nii.gz")
        print("  [1/2] Pretrained inference...")
        do_mr_to_pct(mri_path, pct_pre_path, saved_pre, device, prep_t1=PREP_T1, plot_mrct=False)
        m_pre, mv_pre, ct_arr, pct_pre_arr = compute_metrics(pct_pre_path, ct_path, sid)

        # --- Fine-tuned ---
        pct_ft_path = str(TMP_DIR / f"{sid}_ft_sCT.nii.gz")
        print("  [2/2] Fine-tuned inference...")
        do_mr_to_pct(mri_path, pct_ft_path, saved_ft, device, prep_t1=PREP_T1, plot_mrct=False)
        m_ft, mv_ft, _, pct_ft_arr = compute_metrics(pct_ft_path, ct_path, sid)

        if m_pre and m_ft:
            print(f"\n  {'Metric':<26} {'Pretrained':>12} {'Fine-tuned':>12} {'Δ':>10}")
            print(f"  {'-'*62}")
            for k, label in [("mae","MAE Global (HU)"),("mae_soft","MAE Soft (HU)"),
                              ("mae_bone","MAE Bone (HU)"),("ssim","SSIM")]:
                delta = m_ft[k] - m_pre[k]
                sign  = "+" if delta > 0 else ""
                print(f"  {label:<26} {m_pre[k]:>12.3f} {m_ft[k]:>12.3f} {sign}{delta:>9.3f}")

            if mv_pre and mv_ft:
                print(f"  {'--- Vertebrae region ---':<62}")
                for k, label in [("mae","MAE Vertebrae (HU)"),("mae_bone","MAE Bone Vert (HU)"),("ssim","SSIM Vertebrae")]:
                    delta = mv_ft[k] - mv_pre[k]
                    sign  = "+" if delta > 0 else ""
                    print(f"  {label:<26} {mv_pre[k]:>12.3f} {mv_ft[k]:>12.3f} {sign}{delta:>9.3f}")

            results.append({"sid": sid, "pre": m_pre, "ft": m_ft,
                             "vpre": mv_pre, "vft": mv_ft})

        # Save figure (5 panels)
        mri_arr     = sitk.GetArrayFromImage(sitk.ReadImage(mri_path))
        sl          = mri_arr.shape[0] // 2
        error_map   = np.abs(pct_ft_arr - ct_arr)

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        m_pre_s = m_pre or {}
        m_ft_s  = m_ft  or {}
        fig.suptitle(
            f"{sid} — MAE pre={m_pre_s.get('mae',0):.1f} ft={m_ft_s.get('mae',0):.1f} HU | "
            f"SSIM pre={m_pre_s.get('ssim',0):.3f} ft={m_ft_s.get('ssim',0):.3f}", fontsize=11)

        axes[0].imshow(mri_arr[sl],     cmap='gray');                        axes[0].set_title("MRI");            axes[0].axis('off')
        axes[1].imshow(ct_arr[sl],      cmap='gray', vmin=-1000, vmax=1500); axes[1].set_title("Real CT");        axes[1].axis('off')
        axes[2].imshow(pct_pre_arr[sl], cmap='gray', vmin=-1000, vmax=1500); axes[2].set_title("pCT Pretrained"); axes[2].axis('off')
        axes[3].imshow(pct_ft_arr[sl],  cmap='gray', vmin=-1000, vmax=1500); axes[3].set_title("pCT Fine-tuned"); axes[3].axis('off')
        im5 = axes[4].imshow(error_map[sl], cmap='hot', vmin=0, vmax=500);  axes[4].set_title("Error (ft)");     axes[4].axis('off')
        fig.colorbar(im5, ax=axes[4], fraction=0.046, pad=0.04)

        fig_path = OUT_DIR / f"{sid}_comparison.png"
        plt.tight_layout()
        plt.savefig(str(fig_path), dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Figure: {fig_path}")

    # Summary
    if results:
        print("\n" + "=" * 68)
        print("SUMMARY — mean across all subjects")
        print("=" * 68)

        print(f"\n  {'Metric':<28} {'Pretrained':>12} {'Fine-tuned':>12} {'Δ':>10}")
        print(f"  {'-'*62}")
        for k, label in [("mae","MAE Global (HU)"),("mae_soft","MAE Soft (HU)"),
                          ("mae_bone","MAE Bone (HU)"),("ssim","SSIM")]:
            mean_pre = np.mean([r["pre"][k] for r in results])
            mean_ft  = np.mean([r["ft"][k]  for r in results])
            delta    = mean_ft - mean_pre
            sign     = "+" if delta > 0 else ""
            print(f"  {label:<28} {mean_pre:>12.3f} {mean_ft:>12.3f} {sign}{delta:>9.3f}")

        vert_results = [r for r in results if r["vpre"] and r["vft"]]
        if vert_results:
            print(f"\n  --- Vertebrae region ({len(vert_results)} subjects) ---")
            for k, label in [("mae","MAE Vertebrae (HU)"),("mae_bone","MAE Bone Vert (HU)"),("ssim","SSIM Vertebrae")]:
                mean_pre = np.mean([r["vpre"][k] for r in vert_results])
                mean_ft  = np.mean([r["vft"][k]  for r in vert_results])
                delta    = mean_ft - mean_pre
                sign     = "+" if delta > 0 else ""
                better   = "✓ BETTER" if (k == "ssim" and delta > 0) or (k != "ssim" and delta < 0) else ""
                print(f"  {label:<28} {mean_pre:>12.3f} {mean_ft:>12.3f} {sign}{delta:>9.3f}  {better}")

        csv_path = OUT_DIR / "results_comparison.csv"
        with open(csv_path, "w") as f:
            f.write("subject,"
                    "mae_pre,mae_ft,mae_soft_pre,mae_soft_ft,mae_bone_pre,mae_bone_ft,ssim_pre,ssim_ft,"
                    "mae_vert_pre,mae_vert_ft,mae_bone_vert_pre,mae_bone_vert_ft,ssim_vert_pre,ssim_vert_ft\n")
            for r in results:
                vp = r["vpre"] or {}
                vf = r["vft"]  or {}
                f.write(f"{r['sid']},"
                        f"{r['pre']['mae']:.4f},{r['ft']['mae']:.4f},"
                        f"{r['pre']['mae_soft']:.4f},{r['ft']['mae_soft']:.4f},"
                        f"{r['pre']['mae_bone']:.4f},{r['ft']['mae_bone']:.4f},"
                        f"{r['pre']['ssim']:.4f},{r['ft']['ssim']:.4f},"
                        f"{vp.get('mae',0):.4f},{vf.get('mae',0):.4f},"
                        f"{vp.get('mae_bone',0):.4f},{vf.get('mae_bone',0):.4f},"
                        f"{vp.get('ssim',0):.4f},{vf.get('ssim',0):.4f}\n")
        print(f"\n  CSV: {csv_path}")

    # Cleanup
    import shutil
    shutil.rmtree(str(TMP_DIR), ignore_errors=True)
    print("\nDone.")