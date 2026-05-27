#!/usr/bin/env python3
"""
Preprocessat làmina vertebral — MRI (FLAIR)
=============================================
Llegeix imatges FLAIR + segmentacions de vèrtebres i làmina (3D Slicer .seg.nrrd)
i genera el dataset en format nnUNet (Dataset002_LaminaMRI).

Estructura d'entrada esperada
------------------------------
MRI:
  /MRXFDG-PET-CT-MRI/ALL/sub-XXXX/anat/sub-XXXX_FLAIR.nii.gz

Anotacions (3D Slicer):
  /LAMINA_ANNOTATIONS/Mri/SUB-XXXX/sub-XXXX-mri-segmentation.seg.nrrd  ← vèrtebres (tots)
  /LAMINA_ANNOTATIONS/Mri/SUB-XXXX/sub-XXXX-mri-lamines.seg.nrrd        ← làmina (10 primers)

Estructura de sortida (nnUNet)
-------------------------------
Dataset002_LaminaMRI/
  imagesTr/   sub0001_L1_0000.nii.gz  (canal 0: FLAIR)
              sub0001_L1_0001.nii.gz  (canal 1: màscara vèrtebra)
  labelsTr/   sub0001_L1.nii.gz       (ground truth làmina)
  imagesTs/   sub0011_L1_0000.nii.gz  (sense làmina → test)
  dataset.json
"""

import json
import re
import sys
from pathlib import Path

import nrrd
import numpy as np
import SimpleITK as sitk

# ─── CONFIGURACIÓ ─────────────────────────────────────────────────────────────

MRI_DIR        = Path("/Users/saramasdeusans/Desktop/MRXFDG-PET-CT-MRI/ALL")
ANNOT_DIR      = Path("/Users/saramasdeusans/Desktop/LAMINA_ANNOTATIONS/Mri")
OUTPUT_DIR     = Path("/Users/saramasdeusans/Desktop/NUNNET_WORK/MRI/Dataset002_Lamina")

# Subjectes: 20 totals, 10 primers amb làmina
ALL_SUBJECTS    = [f"sub-{i:04d}" for i in range(1, 21)]   # sub-0001 … sub-0020
LAMINA_SUBJECTS = [f"sub-{i:04d}" for i in range(1, 11)]   # sub-0001 … sub-0010

# Nivells vertebrals que busquem (nom curt = com apareix al .seg.nrrd)
# Format al fitxer de Sara: "L1_Lamina", "T12_Lamina", "C1_Lamina"...
# El script busca el nom curt dins del nom del segment (flexible)
VERTEBRA_LEVELS = [
    "L5", "L4", "L3", "L2", "L1",
    "T12", "T11", "T10", "T9", "T8", "T7", "T6", "T5", "T4", "T3", "T2", "T1",
    "C7", "C6", "C5", "C4", "C3", "C2", "C1",
]

MARGIN_MM  = 10    # marge al voltant de la vèrtebra (mm)
TARGET_SPACING = (1.0, 1.0, 1.0)  # resolució estàndard de sortida

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_seg_nrrd(path: Path) -> tuple[np.ndarray, dict]:
    """Carrega un .seg.nrrd de 3D Slicer."""
    data, header = nrrd.read(str(path))
    return data, header


def get_segment_label_map(header: dict) -> dict[str, int]:
    """
    Retorna {nom_segment: valor_enter} llegint els metadades del .seg.nrrd.
    """
    label_map = {}
    i = 0
    while f"Segment{i}_Name" in header:
        name  = header[f"Segment{i}_Name"]
        # LabelValue pot ser string o int
        value = int(header.get(f"Segment{i}_LabelValue", i + 1))
        label_map[name] = value
        i += 1
    return label_map


def nrrd_to_sitk(data: np.ndarray, header: dict,
                  reference: sitk.Image = None) -> sitk.Image:
    """
    Converteix array numpy + capçalera nrrd a SimpleITK Image.

    Si es passa 'reference' i la mida coincideix, copia directament la
    geometria de la referència (evita errors de conversió RAS/LPS entre
    versions de 3D Slicer).
    """
    # nrrd de Slicer: ordre [k,j,i] → transposem a [i,j,k] = [x,y,z]
    if data.ndim == 3:
        arr = data.transpose(2, 1, 0).astype(np.int16)
    else:
        arr = data.astype(np.int16)

    img = sitk.GetImageFromArray(arr)

    # ── Estratègia preferida: copiar geometria de la MRI de referència ──
    if reference is not None and tuple(arr.shape[::-1]) == reference.GetSize():
        img.CopyInformation(reference)
        return img

    # ── Fallback: extreure geometria del header nrrd ──────────────────
    # Detecta si l'espai és RAS (cal negar x i y per passar a LPS)
    space = header.get("space", "").lower().replace("-", "").replace(" ", "")
    is_ras = space in ("rightanteriorsuperior", "ras")

    # Spacing
    if "space directions" in header:
        sd = np.array(header["space directions"])
        spacing = tuple(float(np.linalg.norm(sd[i])) for i in range(3))
        img.SetSpacing(spacing)
    elif "spacings" in header:
        img.SetSpacing(tuple(float(s) for s in header["spacings"]))

    # Origin (converteix RAS → LPS si cal)
    if "space origin" in header:
        origin = [float(v) for v in header["space origin"]]
        if is_ras:
            origin[0] = -origin[0]
            origin[1] = -origin[1]
        img.SetOrigin(tuple(origin))

    # Direction (converteix RAS → LPS si cal)
    if "space directions" in header:
        sd = np.array(header["space directions"])
        spacing_arr = np.array([np.linalg.norm(sd[i]) for i in range(3)])
        direction = sd / spacing_arr[:, None]
        if is_ras:
            direction[0] = -direction[0]
            direction[1] = -direction[1]
        img.SetDirection(direction.flatten().tolist())

    return img


def resample_to_reference(moving: sitk.Image, reference: sitk.Image,
                           is_label: bool = False) -> sitk.Image:
    """Resampla 'moving' a l'espai de 'reference'."""
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(moving)


def resample_isotropic(img: sitk.Image, spacing: tuple,
                       is_label: bool = False) -> sitk.Image:
    """Resampla a una resolució isòtropa."""
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
                     margin_mm: float) -> tuple[sitk.Image, sitk.Image, list]:
    """
    Retorna la regió de img i mask al voltant de la bounding box de mask,
    amb un marge de margin_mm per cada costat.
    """
    arr  = sitk.GetArrayFromImage(mask)  # [z, y, x]
    idxs = np.argwhere(arr > 0)
    if idxs.size == 0:
        return None, None, None

    mn = idxs.min(axis=0)  # [z, y, x]
    mx = idxs.max(axis=0)

    spacing = np.array(img.GetSpacing())  # [x, y, z]
    margin_vox = np.ceil(margin_mm / spacing[::-1]).astype(int)  # [z, y, x]

    size   = np.array(img.GetSize())   # [x, y, z]
    mn_pad = np.maximum(mn - margin_vox, 0)
    mx_pad = np.minimum(mx + margin_vox, size[::-1] - 1)

    # SimpleITK crop: índexs en ordre [x, y, z]
    lower = [int(mn_pad[2]), int(mn_pad[1]), int(mn_pad[0])]
    upper = [int(size[0] - 1 - mx_pad[2]),
             int(size[1] - 1 - mx_pad[1]),
             int(size[2] - 1 - mx_pad[0])]
    upper = [max(u, 0) for u in upper]

    crop = sitk.CropImageFilter()
    crop.SetLowerBoundaryCropSize(lower)
    crop.SetUpperBoundaryCropSize(upper)

    return crop.Execute(img), crop.Execute(mask), lower


def save_nifti(img: sitk.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))


# ─── PROCESSAMENT D'UN SUBJECTE ───────────────────────────────────────────────

def process_subject(subj: str, has_lamina: bool,
                    images_dir: Path, labels_dir: Path):
    """
    Processa un subjecte MRI i genera els crops per vèrtebra.
    """
    subj_upper = subj.upper()          # sub-0001 → SUB-0001
    subj_nn    = subj.replace("-", "") # sub-0001 → sub0001

    # ── Carrega FLAIR ──────────────────────────────────────────────
    flair_path = MRI_DIR / subj / "anat" / f"{subj}_FLAIR.nii.gz"
    if not flair_path.exists():
        print(f"  ✗ FLAIR no trobat: {flair_path}")
        return 0

    mri = sitk.ReadImage(str(flair_path), sitk.sitkFloat32)
    print(f"  ✓ FLAIR carregat: {flair_path.name}")

    # ── Carrega segmentació de vèrtebres ───────────────────────────
    seg_path = ANNOT_DIR / subj_upper / f"{subj}-mri-segmentation.seg.nrrd"
    if not seg_path.exists():
        print(f"  ✗ Segmentació vèrtebres no trobada: {seg_path}")
        return 0

    seg_data, seg_header = load_seg_nrrd(seg_path)
    seg_sitk = nrrd_to_sitk(seg_data, seg_header, reference=mri)
    seg_sitk = resample_to_reference(seg_sitk, mri, is_label=True)
    seg_label_map = get_segment_label_map(seg_header)
    print(f"  ✓ Segmentació vèrtebres: {len(seg_label_map)} segments")
    print(f"     Noms trobats: {list(seg_label_map.keys())}")

    # ── Carrega anotació de làmina (si existeix) ───────────────────
    lamina_sitk = None
    lam_header  = {}
    if has_lamina:
        lam_path = ANNOT_DIR / subj_upper / f"{subj}-mri-lamines.seg.nrrd"
        if not lam_path.exists():
            print(f"  ✗ Làmina no trobada: {lam_path}")
            has_lamina = False
        else:
            lam_data, lam_header = load_seg_nrrd(lam_path)
            lamina_sitk = nrrd_to_sitk(lam_data, lam_header, reference=mri)
            lamina_sitk = resample_to_reference(lamina_sitk, mri, is_label=True)
            lam_lbl = get_segment_label_map(lam_header)
            print(f"  ✓ Làmina carregada: {list(lam_lbl.keys())}")

    # ── Resampla MRI a resolució isòtropa ──────────────────────────
    mri_iso = resample_isotropic(mri, TARGET_SPACING, is_label=False)
    seg_iso = resample_isotropic(seg_sitk, TARGET_SPACING, is_label=True)
    if lamina_sitk is not None:
        lam_iso = resample_isotropic(lamina_sitk, TARGET_SPACING, is_label=True)
    else:
        lam_iso = None

    # ── Genera crops per vèrtebra ──────────────────────────────────
    seg_arr     = sitk.GetArrayFromImage(seg_iso)
    seg_lbl_map = get_segment_label_map(seg_header)

    # Mapa de làmina per nivell: {"L1": valor_enter, "T12": valor_enter, ...}
    lam_lbl_map = {}
    if lam_iso is not None:
        lam_arr_full = sitk.GetArrayFromImage(lam_iso)
        lam_lbl_map_raw = get_segment_label_map(lam_header if has_lamina else {})

    n_crops = 0

    for level in VERTEBRA_LEVELS:
        # ── Cerca vèrtebra pel nivell (flexible: "L1", "vertebrae_L1", etc.) ──
        label_val = None
        for seg_name, val in seg_lbl_map.items():
            # Coincideix si el nivell és una paraula dins del nom del segment
            # ex: "L1" encaixa amb "L1", "vertebrae_L1", "L1_body", etc.
            parts = re.split(r'[_\-\s]', seg_name.upper())
            if level.upper() in parts or seg_name.upper() == level.upper():
                label_val = val
                break
        if label_val is None:
            continue

        # Màscara binària d'aquesta vèrtebra
        vert_arr = (seg_arr == label_val).astype(np.uint8)
        if vert_arr.sum() == 0:
            continue

        vert_mask = sitk.GetImageFromArray(vert_arr)
        vert_mask.CopyInformation(seg_iso)

        # Crop MRI i màscara vèrtebra
        mri_crop, mask_crop, bounds = crop_with_margin(
            mri_iso, vert_mask, MARGIN_MM)
        if mri_crop is None:
            continue

        # Nom del cas nnUNet: sub0001_L1
        case_id = f"{subj_nn}_{level}"

        # Desa imatge (canal 0) i màscara vèrtebra (canal 1)
        save_nifti(mri_crop,  images_dir / f"{case_id}_0000.nii.gz")
        save_nifti(mask_crop, images_dir / f"{case_id}_0001.nii.gz")

        # Desa làmina (label) si existeix
        # Busca el segment "{level}_Lamina" al fitxer de làmina
        if lam_iso is not None and labels_dir is not None:
            lam_arr_full = sitk.GetArrayFromImage(lam_iso)
            lam_level_arr = np.zeros_like(lam_arr_full, dtype=np.uint8)

            # Cerca segment que contingui el nivell + "lamina"
            lam_lbl_map_raw = get_segment_label_map(lam_header)
            for lam_name, lam_val in lam_lbl_map_raw.items():
                name_up = lam_name.upper().replace(" ", "_")
                if level.upper() in name_up and "LAMINA" in name_up:
                    lam_level_arr = (lam_arr_full == lam_val).astype(np.uint8)
                    break

            lam_level_img = sitk.GetImageFromArray(lam_level_arr)
            lam_level_img.CopyInformation(lam_iso)

            lam_crop, _, _ = crop_with_margin(lam_level_img, vert_mask, MARGIN_MM)
            if lam_crop is not None:
                save_nifti(lam_crop, labels_dir / f"{case_id}.nii.gz")

        n_crops += 1
        print(f"    ✓ {case_id}")

    return n_crops


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PREPROCESSAT MRI — Dataset002_LaminaMRI")
    print("=" * 60)

    images_tr = OUTPUT_DIR / "imagesTr"
    labels_tr = OUTPUT_DIR / "labelsTr"
    images_ts = OUTPUT_DIR / "imagesTs"
    for d in [images_tr, labels_tr, images_ts]:
        d.mkdir(parents=True, exist_ok=True)

    total_tr = 0
    total_ts = 0

    # Subjectes amb làmina → imagesTr + labelsTr
    print(f"\n── Subjectes entrenament ({len(LAMINA_SUBJECTS)}) ──")
    for subj in LAMINA_SUBJECTS:
        print(f"\n{subj}")
        n = process_subject(subj, has_lamina=True,
                            images_dir=images_tr, labels_dir=labels_tr)
        total_tr += n

    # Subjectes sense làmina → imagesTs
    test_subjects = [s for s in ALL_SUBJECTS if s not in LAMINA_SUBJECTS]
    print(f"\n── Subjectes test ({len(test_subjects)}) ──")
    for subj in test_subjects:
        print(f"\n{subj}")
        n = process_subject(subj, has_lamina=False,
                            images_dir=images_ts, labels_dir=None)
        total_ts += n

    # dataset.json
    # Recull tots els casos de train per al JSON
    tr_cases = sorted(images_tr.glob("*_0000.nii.gz"))
    training_list = [
        {"image": f"./imagesTr/{f.name.replace('_0000.nii.gz','')}",
         "label": f"./labelsTr/{f.name.replace('_0000.nii.gz','')}.nii.gz"}
        for f in tr_cases
    ]

    dataset = {
        "channel_names": {"0": "MRI", "1": "MRI"},
        "labels": {"background": 0, "lamina": 1},
        "numTraining": len(training_list),
        "file_ending": ".nii.gz"
    }

    with open(OUTPUT_DIR / "dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print("\n" + "=" * 60)
    print("✓ PREPROCESSAT COMPLETAT")
    print(f"  Crops entrenament : {total_tr}")
    print(f"  Crops test        : {total_ts}")
    print(f"  Sortida           : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()