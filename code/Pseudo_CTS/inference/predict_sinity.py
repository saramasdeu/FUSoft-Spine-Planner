"""
inference/predict_sitiny.py
---------------------------
Sliding-window inference with the sitiny ShuffleUNet model.

Accepts a single MRI NIfTI file, returns a pseudo-CT NIfTI file
in the same physical space (same origin / spacing / direction).

Usage (standalone):
    python inference/predict_sitiny.py \
        --mri   /path/to/sub-0020_T1w.nii.gz \
        --out   /path/to/sub-0020_sitiny_pCT.nii.gz \
        --weights /path/to/pretrained_net_final_20220825.pth

Programmatic:
    from inference.predict_sitiny import run_sitiny_inference
    run_sitiny_inference(mri_path, output_path, weights_path, device)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference

# Allow imports from project root when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.shuffle_unet import build_model
from data.preprocessing import normalise_mri, denormalise_ct


def run_sitiny_inference(
    mri_path: str | Path,
    output_path: str | Path,
    weights_path: str | Path,
    device: torch.device,
    roi_size: tuple[int, int, int] = (128, 128, 64),
    sw_batch_size: int = 2,
    overlap: float = 0.25,
) -> Path:
    """Run ShuffleUNet inference on one MRI and save the pseudo-CT.

    Parameters
    ----------
    mri_path      : input T1w MRI (.nii / .nii.gz)
    output_path   : where to save the pseudo-CT (.nii.gz)
    weights_path  : model checkpoint (.pth)
    device        : torch device (cuda / mps / cpu)
    roi_size      : sliding-window patch size (X, Y, Z)
    sw_batch_size : number of patches processed simultaneously
    overlap       : sliding-window overlap fraction (0–1)

    Returns
    -------
    Path to the saved pseudo-CT file.
    """
    mri_path     = Path(mri_path)
    output_path  = Path(output_path)
    weights_path = Path(weights_path)

    # ------------------------------------------------------------------ load
    mri_itk = sitk.ReadImage(str(mri_path))
    mri_arr = normalise_mri(mri_itk)          # float32, [-1, 1]
    input_tensor = (
        torch.from_numpy(mri_arr)
        .float()
        .unsqueeze(0)   # channel
        .unsqueeze(0)   # batch
        .to(device)
    )

    # --------------------------------------------------------------- model
    model = build_model(device)
    state = torch.load(str(weights_path), map_location=device)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"  [INFO] Missing keys (init randomly): {len(result.missing_keys)}")
    if result.unexpected_keys:
        print(f"  [INFO] Unexpected keys (ignored): {len(result.unexpected_keys)}")
    model.eval()

    # ------------------------------------------------------------- inference
    with torch.no_grad():
        output = sliding_window_inference(
            inputs=input_tensor,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )

    output_arr = output.cpu().numpy()[0, 0]   # (Z, Y, X)
    pseudo_ct  = denormalise_ct(output_arr)   # HU, float32

    # ----------------------------------------------------------------- save
    result_itk = sitk.GetImageFromArray(pseudo_ct)
    result_itk.CopyInformation(mri_itk)       # preserve geometry
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(result_itk, str(output_path))

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sitiny ShuffleUNet inference")
    parser.add_argument("--mri",     required=True, type=Path, help="Input MRI (.nii/.nii.gz)")
    parser.add_argument("--out",     required=True, type=Path, help="Output pseudo-CT path")
    parser.add_argument("--weights", required=True, type=Path, help="Model weights (.pth)")
    parser.add_argument("--device",  default="auto",            help="cuda | mps | cpu | auto")
    parser.add_argument("--overlap", default=0.25, type=float,  help="Sliding-window overlap")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(args.device)

    print(f"Device : {dev}")
    print(f"MRI    : {args.mri}")
    print(f"Weights: {args.weights}")

    out = run_sitiny_inference(
        mri_path=args.mri,
        output_path=args.out,
        weights_path=args.weights,
        device=dev,
        overlap=args.overlap,
    )
    print(f"Saved  : {out}")