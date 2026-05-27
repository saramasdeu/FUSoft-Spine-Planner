#!/usr/bin/env python3
"""
segmenta_lamina.py — Pipeline complet de segmentació de la làmina vertebral
=============================================================================
Donada una imatge CT o MRI, executa:
  1. Detecció automàtica de modalitat (CT / MRI)
  2. Segmentació de vèrtebres amb TotalSegmentator
  3. Crop per vèrtebra (preprocessing nnUNet)
  4. Inferència nnUNet amb el model corresponent
  5. Reconstrucció de la màscara completa en espai original

Ús:
    python segmenta_lamina.py /ruta/imatge.nii.gz
    python segmenta_lamina.py /ruta/imatge.nii.gz -o /ruta/sortida.nii.gz
    python segmenta_lamina.py   ← demana la ruta interactivament
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# ─── CONFIGURACIÓ ─────────────────────────────────────────────────────────────

VENV_BIN = Path("/home/sara/FUSoft-Spine-Planner/.venv/bin")

# Entorn nnUNet
NNUNET_RAW_CT  = Path("/home/sara")   # parent de Dataset001_Lamina
NNUNET_RAW_MRI = Path("/home/sara")   # parent de Dataset002_Lamina

NNUNET_PREPROC_CT  = Path("/home/sara/nnunet_work/nnunet_preprocessed")
NNUNET_PREPROC_MRI = Path("/home/sara/FUSoft-Spine-Planner/MRI_segmentation/NUNNET_WORK/MRI/nnunet_preprocessed")

NNUNET_RESULTS_CT  = Path("/home/sara/nnunet_work/nnunet_results")
NNUNET_RESULTS_MRI = Path("/home/sara/FUSoft-Spine-Planner/MRI_segmentation/NUNNET_WORK/MRI/nnunet_results")

DATASET_ID_CT  = "001"
DATASET_ID_MRI = "002"
CONFIG = "3d_fullres"
FOLD   = "0"

# Preprocessing
VERTEBRA_LEVELS = [
    "L5", "L4", "L3", "L2", "L1",
    "T12", "T11", "T10", "T9", "T8", "T7", "T6", "T5", "T4", "T3", "T2", "T1",
    "C7", "C6", "C5", "C4", "C3", "C2", "C1",
]
MARGIN_MM      = 10
TARGET_SPACING = (1.0, 1.0, 1.0)

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def run(cmd: list, env=None, check=True):
    print(f"\n▶ {' '.join(str(c) for c in cmd)}\n{'─'*60}")
    result = subprocess.run([str(c) for c in cmd], env=env or os.environ.copy())
    if check and result.returncode != 0:
        print(f"✗ Error (codi {result.returncode})")
        sys.exit(result.returncode)
    return result.returncode


def detect_modality(img: sitk.Image) -> str:
    """
    Detecta si la imatge és CT o MRI.
    CT: té valors negatius (unitats Hounsfield, min < -200)
    MRI: tots els valors >= 0
    """
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    min_val = float(arr.min())
    print(f"  Valor mínim de la imatge: {min_val:.1f}")
    if min_val < -200:
        return "CT"
    else:
        return "MRI"


def resample_isotropic(img: sitk.Image, spacing: tuple,
                       is_label: bool = False) -> sitk.Image:
    orig_spacing = img.GetSpacing()
    orig_size    = img.GetSize()
    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / spacing[i]))
        for i in range(3)
    ]
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(img)


def crop_with_margin(img: sitk.Image, mask: sitk.Image,
                     margin_mm: float):
    arr  = sitk.GetArrayFromImage(mask)
    idxs = np.argwhere(arr > 0)
    if idxs.size == 0:
        return None, None

    mn = idxs.min(axis=0)
    mx = idxs.max(axis=0)

    spacing    = np.array(img.GetSpacing())
    margin_vox = np.ceil(margin_mm / spacing[::-1]).astype(int)
    size       = np.array(img.GetSize())

    mn_pad = np.maximum(mn - margin_vox, 0)
    mx_pad = np.minimum(mx + margin_vox, size[::-1] - 1)

    lower = [int(mn_pad[2]), int(mn_pad[1]), int(mn_pad[0])]
    upper = [int(size[0] - 1 - mx_pad[2]),
             int(size[1] - 1 - mx_pad[1]),
             int(size[2] - 1 - mx_pad[0])]
    upper = [max(u, 0) for u in upper]

    crop_filter = sitk.CropImageFilter()
    crop_filter.SetLowerBoundaryCropSize(lower)
    crop_filter.SetUpperBoundaryCropSize(upper)

    return crop_filter.Execute(img), crop_filter.Execute(mask)


# ─── PAS 1: TOTALSEGMENTATOR ──────────────────────────────────────────────────

def run_totalsegmentator(image_path: Path, output_dir: Path,
                          modality: str) -> Path:
    """
    Executa TotalSegmentator per segmentar les vèrtebres.
    Retorna la ruta del fitxer de segmentació generat.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

    print(f"\n[1/5] TotalSegmentator → vèrtebres (task vertebrae_mr)...")
    ret = run([
        VENV_BIN / "TotalSegmentator",
        "-i", image_path,
        "-o", output_dir,
        "--task", "vertebrae_mr",
    ], env=env, check=False)
    if ret != 0:
        print("  ⚠ TotalSegmentator ha fallat.")
        sys.exit(1)
    vertebra_mask, label_map = combine_totalseg_vertebrae_ct(output_dir, image_path)
    return vertebra_mask, label_map


def combine_totalseg_vertebrae_ct(seg_dir: Path, ref_path: Path):
    """
    Combina els fitxers individuals de TotalSegmentator en una màscara multi-label.
    Retorna (path_màscara, label_map) on label_map = {label_val: nom_vertebra}.
    """
    ref = sitk.ReadImage(str(ref_path), sitk.sitkFloat32)
    combined = sitk.Image(ref.GetSize(), sitk.sitkUInt8)
    combined.CopyInformation(ref)

    label_map = {}  # {label_val: nom_vertebra}
    for i, level in enumerate(reversed(VERTEBRA_LEVELS), start=1):
        candidates = list(seg_dir.glob(f"vertebrae_{level}.nii.gz"))
        if not candidates:
            candidates = list(seg_dir.glob(f"*{level}*.nii.gz"))
        if not candidates:
            continue

        seg = sitk.ReadImage(str(candidates[0]), sitk.sitkUInt8)
        seg = sitk.Resample(seg, combined, sitk.Transform(),
                            sitk.sitkNearestNeighbor, 0)
        arr_combined = sitk.GetArrayFromImage(combined)
        arr_seg      = sitk.GetArrayFromImage(seg)
        arr_combined[arr_seg > 0] = i
        combined = sitk.GetImageFromArray(arr_combined)
        combined.CopyInformation(ref)
        label_map[i] = level  # {1: "C1", 2: "C2", ...}

    out_path = seg_dir / "vertebrae_combined.nii.gz"
    sitk.WriteImage(combined, str(out_path))
    print(f"  ✓ Màscara vertebral combinada: {len(label_map)} vèrtebres trobades")
    return out_path, label_map


def save_as_seg_nrrd(img: sitk.Image, label_map: dict, output_path: Path):
    """
    Desa la segmentació com a .seg.nrrd de 3D Slicer amb noms de segments.
    label_map: {label_val: nom_vertebra} p.ex. {1: "C1", 2: "C2"}
    """
    import nrrd

    arr = sitk.GetArrayFromImage(img)          # [z, y, x]
    arr_nrrd = arr.transpose(2, 1, 0)          # nrrd: [x, y, z]

    spacing    = img.GetSpacing()              # (sx, sy, sz)
    origin     = img.GetOrigin()
    direction  = np.array(img.GetDirection()).reshape(3, 3)

    space_dirs = [
        (direction[0] * spacing[0]).tolist(),
        (direction[1] * spacing[1]).tolist(),
        (direction[2] * spacing[2]).tolist(),
    ]

    # Colors per regió: cervical=blau, toràcic=verd, lumbar=vermell
    def color_for(name):
        if name.startswith("C"):   return "0.2 0.4 0.9"
        if name.startswith("T"):   return "0.2 0.8 0.3"
        return "0.9 0.2 0.2"

    header = {
        "type": "unsigned char",
        "dimension": 3,
        "space": "left-posterior-superior",
        "sizes": list(arr_nrrd.shape),
        "space directions": space_dirs,
        "space origin": list(origin),
        "kinds": ["domain", "domain", "domain"],
        "encoding": "gzip",
        "segmentation_masterRepresentation": "Binary labelmap",
        "segmentation_containedRepresentationNames": "Binary labelmap",
    }

    for idx, (lval, vname) in enumerate(sorted(label_map.items())):
        seg_name = f"{vname}_Lamina"
        header[f"Segment{idx}_ID"]                    = seg_name
        header[f"Segment{idx}_Name"]                  = seg_name
        header[f"Segment{idx}_Color"]                 = color_for(vname)
        header[f"Segment{idx}_ColorAutoGenerated"]    = "0"
        header[f"Segment{idx}_Layer"]                 = "0"
        header[f"Segment{idx}_LabelValue"]            = str(lval)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nrrd.write(str(output_path), arr_nrrd.astype(np.uint8), header)
    print(f"  ✓ Segmentació guardada: {output_path}")


# ─── PAS 2: PREPROCESSING (CROPS) ────────────────────────────────────────────

def generate_crops(image_path: Path, vertebra_mask_path: Path,
                   crops_dir: Path, modality: str) -> list[Path]:
    """
    Genera els crops per vèrtebra en format nnUNet (2 canals).
    Retorna la llista de fitxers _0000.nii.gz creats.
    """
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/5] Generant crops per vèrtebra ({MARGIN_MM}mm marge)...")

    img = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    seg = sitk.ReadImage(str(vertebra_mask_path), sitk.sitkUInt8)

    # Assegura que la màscara té la mateixa geometria que la imatge
    seg = sitk.Resample(seg, img, sitk.Transform(),
                        sitk.sitkNearestNeighbor, 0)

    # Resampla a resolució isòtropa
    img_iso = resample_isotropic(img, TARGET_SPACING, is_label=False)
    seg_iso = resample_isotropic(seg, TARGET_SPACING, is_label=True)
    seg_arr = sitk.GetArrayFromImage(seg_iso)

    # Obtenim els valors únics de la màscara (cada vèrtebra = un valor)
    unique_labels = [int(v) for v in np.unique(seg_arr) if v > 0]

    crop_files = []
    n = 0
    for label_val in unique_labels:
        vert_arr = (seg_arr == label_val).astype(np.uint8)
        if vert_arr.sum() == 0:
            continue

        vert_mask = sitk.GetImageFromArray(vert_arr)
        vert_mask.CopyInformation(seg_iso)

        img_crop, mask_crop = crop_with_margin(img_iso, vert_mask, MARGIN_MM)
        if img_crop is None:
            continue

        case_id = f"input_{label_val:03d}"
        p0 = crops_dir / f"{case_id}_0000.nii.gz"
        p1 = crops_dir / f"{case_id}_0001.nii.gz"
        sitk.WriteImage(img_crop,  str(p0))
        sitk.WriteImage(mask_crop, str(p1))
        crop_files.append(p0)
        n += 1

    print(f"  ✓ {n} crops generats")
    return crop_files


# ─── PAS 3: INFERÈNCIA nnUNet ─────────────────────────────────────────────────

def run_nnunet_inference(crops_dir: Path, preds_dir: Path,
                          modality: str) -> Path:
    """Executa nnUNetv2_predict sobre els crops."""
    preds_dir.mkdir(parents=True, exist_ok=True)

    if modality == "CT":
        dataset_id   = DATASET_ID_CT
        nnunet_raw   = NNUNET_RAW_CT
        nnunet_pre   = NNUNET_PREPROC_CT
        nnunet_res   = NNUNET_RESULTS_CT
    else:
        dataset_id   = DATASET_ID_MRI
        nnunet_raw   = NNUNET_RAW_MRI
        nnunet_pre   = NNUNET_PREPROC_MRI
        nnunet_res   = NNUNET_RESULTS_MRI

    env = os.environ.copy()
    env["nnUNet_raw"]          = str(nnunet_raw)
    env["nnUNet_preprocessed"] = str(nnunet_pre)
    env["nnUNet_results"]      = str(nnunet_res)

    print(f"\n[3/5] Inferència nnUNet (modalitat: {modality}, dataset: {dataset_id})...")
    run([
        VENV_BIN / "nnUNetv2_predict",
        "-i", crops_dir,
        "-o", preds_dir,
        "-d", dataset_id,
        "-c", CONFIG,
        "-f", FOLD,
        "-chk", "checkpoint_best.pth",
    ], env=env)

    return preds_dir


# ─── PAS 4: RECONSTRUCCIÓ ────────────────────────────────────────────────────

def reconstruct_full_mask(preds_dir: Path, reference_path: Path) -> sitk.Image:
    """
    Reconstrueix la màscara completa combinant tots els crops predits
    en l'espai de la imatge original.
    """
    print("\n[4/5] Reconstruint màscara completa...")

    reference = sitk.ReadImage(str(reference_path), sitk.sitkFloat32)
    ref_iso   = resample_isotropic(reference, TARGET_SPACING, is_label=False)

    # Imatge de sortida (mateixa geometria isòtropa)
    out_arr = np.zeros(sitk.GetArrayFromImage(ref_iso).shape, dtype=np.uint8)

    pred_files = sorted(preds_dir.glob("*.nii.gz"))
    print(f"  Combinant {len(pred_files)} prediccions...")

    for pred_path in pred_files:
        # Extreu el label de la vèrtebra del nom del fitxer (input_001.nii.gz → 1)
        try:
            stem = pred_path.name.replace(".nii.gz", "").replace(".nii", "")
            label_val = int(stem.split("_")[1])
        except (IndexError, ValueError):
            label_val = 1

        pred = sitk.ReadImage(str(pred_path), sitk.sitkUInt8)
        pred_arr_raw = sitk.GetArrayFromImage(pred)
        pred_resampled = sitk.Resample(
            pred, ref_iso, sitk.Transform(),
            sitk.sitkNearestNeighbor, 0
        )
        pred_arr = sitk.GetArrayFromImage(pred_resampled)
        n_raw = int((pred_arr_raw > 0).sum())
        n_res = int((pred_arr > 0).sum())
        print(f"    {pred_path.name}: label={label_val}, vòxels_pred={n_raw}, vòxels_resampled={n_res}")
        # Assigna el label de la vèrtebra a cada predicció
        out_arr[pred_arr > 0] = label_val

    out_img = sitk.GetImageFromArray(out_arr.astype(np.uint8))
    out_img.CopyInformation(ref_iso)

    # Torna a l'espai original (spacing original)
    out_original = sitk.Resample(
        out_img, reference, sitk.Transform(),
        sitk.sitkNearestNeighbor, 0
    )

    n_voxels = int((out_arr > 0).sum())
    print(f"  ✓ Màscara reconstruïda ({n_voxels} vòxels de làmina)")
    return out_original


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Segmentació automàtica de la làmina vertebral (CT/MRI)"
    )
    parser.add_argument("input", nargs="?", help="Ruta a la imatge .nii.gz (CT o FLAIR)")
    parser.add_argument("-o", "--output", help="Ruta de sortida (opcional)")
    parser.add_argument("--t1", help="Imatge T1 per detecció vertebral (necessari si entrada és FLAIR)")
    args = parser.parse_args()

    # ── Obté ruta d'entrada ──────────────────────────────────────────
    if args.input:
        image_path = Path(args.input)
    else:
        print("=" * 60)
        print("SEGMENTACIÓ AUTOMÀTICA DE LA LÀMINA VERTEBRAL")
        print("=" * 60)
        ruta = input("\nIntrodueix la ruta de la imatge (.nii.gz): ").strip()
        image_path = Path(ruta)

    if not image_path.exists():
        print(f"✗ Fitxer no trobat: {image_path}")
        sys.exit(1)

    # ── Ruta de sortida ──────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
        # Elimina extensions .nii.gz o .nii i afegeix _lamina.nii.gz
        name = image_path.name
        if name.endswith(".nii.gz"):
            stem = name[:-7]
        elif name.endswith(".nii"):
            stem = name[:-4]
        else:
            stem = name
        output_path = image_path.parent / f"{stem}_lamina.nii.gz"

    print("=" * 60)
    print("SEGMENTACIÓ AUTOMÀTICA DE LA LÀMINA VERTEBRAL")
    print("=" * 60)
    print(f"\n  Entrada : {image_path}")
    print(f"  Sortida : {output_path}")

    # ── Detecta modalitat ────────────────────────────────────────────
    print("\n[0/5] Detectant modalitat...")
    img = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    modality = detect_modality(img)
    print(f"  ✓ Modalitat detectada: {modality}")

    # ── Carpeta temporal ─────────────────────────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="lamina_"))
    print(f"\n  Carpeta temporal: {tmp_dir}")

    try:
        seg_dir   = tmp_dir / "totalseg"
        crops_dir = tmp_dir / "crops"
        preds_dir = tmp_dir / "predictions"

        # Pas 1: TotalSegmentator
        totalseg_input = image_path
        if modality == "MRI" and args.t1:
            totalseg_input = Path(args.t1)
            print(f"\n  Usant T1 per detecció vertebral: {totalseg_input}")
        vertebra_mask, label_map = run_totalsegmentator(totalseg_input, seg_dir, modality)

        # Pas 2: Crops
        generate_crops(image_path, vertebra_mask, crops_dir, modality)

        # Pas 3: Inferència
        run_nnunet_inference(crops_dir, preds_dir, modality)

        # Pas 4: Reconstrucció
        full_mask = reconstruct_full_mask(preds_dir, image_path)

        # Pas 5: Desa resultat
        print(f"\n[5/5] Desant resultat...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(full_mask, str(output_path))
        print(f"  ✓ Segmentació guardada: {output_path}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("✓ PIPELINE COMPLETAT!")
    print(f"  Modalitat   : {modality}")
    print(f"  Resultat    : {output_path}")
    print(f"\n  Pots carregar el resultat a 3D Slicer per visualitzar-lo.")
    print("=" * 60)


if __name__ == "__main__":
    main()