"""
scripts/run_evaluation.py
--------------------------
Evaluate pseudo-CT generation on all paired MRI/CT subjects.

Data layout expected:
    <data_dir>/
        sub-0020/
            anat/sub-0020_T1w.nii.gz
            ct/sub-0020_ct.nii.gz
        sub-0021/
            ...

The script:
  1. Discovers all sub-XXXX folders that have both MRI and CT.
  2. Runs inference with the selected model (skips if output already exists).
  3. Resamples the real CT into pseudo-CT space and computes metrics.
  4. Saves a CSV with per-subject results + mean row.

Examples
--------
# Sitiny model:
python scripts/run_evaluation.py \
    --model   sitiny \
    --data_dir /Users/saramasdeusans/Desktop/MRXFDG-PET-CT-MRI/ALL \
    --weights  /path/to/pretrained_net_final_20220825.pth \
    --out_dir  results/sitiny

# Han-liu model:
python scripts/run_evaluation.py \
    --model    hanliu \
    --data_dir /Users/saramasdeusans/Desktop/MRXFDG-PET-CT-MRI/ALL \
    --repo     /path/to/SynCT_TcMRgFUS \
    --out_dir  results/hanliu

# Only specific subjects:
python scripts/run_evaluation.py --model sitiny ... --subjects sub-0020 sub-0025
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.preprocessing import resample_to_reference
from evaluation.metrics import compute_all


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

def discover_subjects(data_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return (subject_id, mri_path, ct_path) for every valid subject folder."""
    subjects = []
    for subj_dir in sorted(data_dir.iterdir()):
        if not subj_dir.is_dir() or not subj_dir.name.startswith("sub-"):
            continue
        sid      = subj_dir.name
        mri_path = subj_dir / "anat" / f"{sid}_T1w.nii.gz"
        ct_path  = subj_dir / "ct"   / f"{sid}_ct.nii.gz"
        if mri_path.exists() and ct_path.exists():
            subjects.append((sid, mri_path, ct_path))
        else:
            missing = []
            if not mri_path.exists(): missing.append("MRI")
            if not ct_path.exists():  missing.append("CT")
            print(f"  [WARN] {sid}: missing {', '.join(missing)} — skipping.")
    return subjects


# ---------------------------------------------------------------------------
# Per-subject evaluation
# ---------------------------------------------------------------------------

def evaluate_subject(
    pct_path: Path,
    ct_path: Path,
    bone_threshold: float = 400.0,
) -> dict:
    """Load pseudo-CT and real CT, resample CT to pCT space, compute metrics."""
    pct_itk = sitk.ReadImage(str(pct_path))
    ct_itk  = sitk.ReadImage(str(ct_path))

    # Resample real CT into pseudo-CT voxel grid
    ct_resampled = resample_to_reference(
        moving=ct_itk,
        reference=pct_itk,
        default_value=-1000.0,   # air HU outside FOV
    )

    pct_arr = sitk.GetArrayFromImage(pct_itk).astype(np.float32)
    ct_arr  = sitk.GetArrayFromImage(ct_resampled).astype(np.float32)

    return compute_all(pct_arr, ct_arr, bone_threshold=bone_threshold)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pseudo-CT models on paired MRI/CT data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",    required=True, choices=["sitiny", "hanliu"])
    parser.add_argument("--data_dir", required=True, type=Path,
                        help="Root folder containing sub-XXXX subdirectories")
    parser.add_argument("--out_dir",  required=True, type=Path,
                        help="Output folder for pseudo-CTs and metrics CSV")

    # sitiny
    parser.add_argument("--weights",  type=Path, help="[sitiny] Path to .pth weights file")
    parser.add_argument("--device",   default="auto",
                        help="[sitiny] Device: cuda | mps | cpu | auto")
    parser.add_argument("--overlap",  default=0.25, type=float,
                        help="[sitiny] Sliding-window overlap (0–1)")

    # hanliu
    parser.add_argument("--repo",          type=Path,
                        help="[hanliu] Path to cloned SynCT_TcMRgFUS repo")
    parser.add_argument("--hanliu_overlap", default=0.6, type=float,
                        help="[hanliu] Sliding-window overlap (0–1)")
    parser.add_argument("--python",        default="python",
                        help="[hanliu] Python executable (e.g. venv/bin/python)")

    # general
    parser.add_argument("--bone_threshold", default=400.0, type=float,
                        help="HU threshold for bone mask in DSC computation")
    parser.add_argument("--subjects", nargs="*", metavar="sub-XXXX",
                        help="Restrict to these subject IDs (default: all)")

    args = parser.parse_args()

    # ----------------------------------------------------------------- device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # ----------------------------------------------------------- validation
    if args.model == "sitiny" and not args.weights:
        sys.exit("[ERROR] --weights is required for the sitiny model.")
    if args.model == "hanliu" and not args.repo:
        sys.exit("[ERROR] --repo is required for the hanliu model.")

    # --------------------------------------------------------- discover
    subjects = discover_subjects(args.data_dir)
    if args.subjects:
        subjects = [(s, m, c) for s, m, c in subjects if s in args.subjects]
    if not subjects:
        sys.exit("[ERROR] No valid subjects found.")

    print(f"\n{'='*60}")
    print(f"Model   : {args.model.upper()}")
    print(f"Device  : {device}")
    print(f"Subjects: {len(subjects)}")
    print(f"Output  : {args.out_dir}")
    print(f"{'='*60}")

    # -------------------------------------------------------- output dirs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pct_dir = args.out_dir / "pseudo_cts"
    pct_dir.mkdir(exist_ok=True)

    # -------------------------------------------------- load infer fn
    if args.model == "sitiny":
        from inference.predict_sitiny import run_sitiny_inference
        def infer(mri_path: Path, out_path: Path) -> None:
            run_sitiny_inference(
                mri_path=mri_path,
                output_path=out_path,
                weights_path=args.weights,
                device=device,
                overlap=args.overlap,
            )
    else:
        from inference.predict_hanliu import run_hanliu_inference
        def infer(mri_path: Path, out_path: Path) -> None:
            run_hanliu_inference(
                mri_path=mri_path,
                output_path=out_path,
                hanliu_repo=args.repo,
                overlap_ratio=args.hanliu_overlap,
                python_bin=args.python,
            )

    # ------------------------------------------------- process subjects
    all_metrics: list[dict] = []
    metric_keys = ["MAE_global", "MAE_bone", "MAE_soft_tissue", "SSIM", "PSNR", "DSC_bone"]

    for sid, mri_path, ct_path in subjects:
        print(f"\n→ {sid}")

        pct_path = pct_dir / f"{sid}_{args.model}_pCT.nii.gz"

        # Inference
        if pct_path.exists():
            print(f"  [SKIP] {pct_path.name} already exists.")
        else:
            print(f"  Inference ... ", end="", flush=True)
            try:
                infer(mri_path, pct_path)
                print("done.")
            except Exception as exc:
                print(f"FAILED: {exc}")
                continue

        # Evaluation
        print(f"  Evaluation ... ", end="", flush=True)
        try:
            m = evaluate_subject(pct_path, ct_path, args.bone_threshold)
            m["subject"] = sid
            all_metrics.append(m)
            print(
                f"MAE={m['MAE_global']:.1f} HU  "
                f"MAE_bone={m['MAE_bone']:.1f} HU  "
                f"DSC_bone={m['DSC_bone']:.3f}  "
                f"SSIM={m['SSIM']:.4f}  "
                f"PSNR={m['PSNR']:.2f} dB"
            )
        except Exception as exc:
            print(f"FAILED: {exc}")

    if not all_metrics:
        print("\n[WARN] No metrics computed — check errors above.")
        return

    # ------------------------------------------------- summary + CSV
    mean_row: dict = {"subject": "MEAN"}
    for k in metric_keys:
        vals = [m[k] for m in all_metrics if not np.isnan(float(m[k]))]
        mean_row[k] = round(float(np.mean(vals)), 4) if vals else float("nan")

    csv_path = args.out_dir / f"metrics_{args.model}.csv"
    fieldnames = ["subject"] + metric_keys
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)
        writer.writerow(mean_row)

    print(f"\n{'='*60}")
    print(f"SUMMARY — {args.model.upper()} ({len(all_metrics)} subjects)")
    for k in metric_keys:
        print(f"  {k:<20}: {mean_row[k]}")
    print(f"\nCSV saved: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()