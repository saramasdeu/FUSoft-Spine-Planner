#!/usr/bin/env python3
"""
Pipeline de preprocessat per segmentació de làmina vertebral
=============================================================
Entrada:
  - CTs:       --ct_dir / sub-XXXX/ct/sub-XXXX_ct.nii.gz
  - Vèrtebres: --seg_dir / SUB-XXXX/sub-XXXX_ct segmentation.seg.nrrd
  - Làmines:   --seg_dir / SUB-XXXX/sub-XXXX_ct_LAMINES.seg.nrrd

Sortida (format nnUNet):
  --out_dir / Dataset001_Lamina/
  ├── imagesTr/
  ├── labelsTr/
  ├── imagesTs/
  └── dataset.json

Ús (rutes per defecte per a la Sara):
    python preprocess_lamina.py

Ús amb rutes personalitzades (per al tutor o altre ordinador):
    python preprocess_lamina.py \\
        --ct_dir  /ruta/als/CTs \\
        --seg_dir /ruta/a/segmentacio_totalseg \\
        --out_dir /ruta/sortida

    python preprocess_lamina.py --inspect
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Afegeix el directori pare al path per importar seg_nrrd_utils
sys.path.append(str(Path(__file__).parent))
from seg_nrrd_utils import (
    read_seg_nrrd,
    find_matching_lamina_segments,
    print_segments_info,
)

# ─── PATHS PER DEFECTE (màquina de la Sara) ───────────────────────────────────
# Canvia aquestes rutes si executes en un altre ordinador,
# o usa els arguments --ct_dir / --seg_dir / --out_dir per línia de comandes.
CT_BASE  = Path("/Users/saramasdeusans/Desktop/MRXFDG-PET-CT-MRI/ALL")
SEG_BASE = Path("/Users/saramasdeusans/Desktop/nnunet_lamina/segmentacio_totalseg")
OUT_BASE = Path("/Users/saramasdeusans/Desktop/nnunet_lamina/Dataset001_Lamina")

# Subjectes amb vèrtebra segmentada (SUB-0001 a SUB-0020)
SUBJECTS_WITH_VERTEBRA = [f"sub-{i:04d}" for i in range(1, 21)]

# Subjectes amb làmina anotada (SUB-0001 a SUB-0005)
SUBJECTS_WITH_LAMINA = [f"sub-{i:04d}" for i in range(1, 6)]

# ─── PARÀMETRES ───────────────────────────────────────────────────────────────
PADDING_MM    = 15     # Marge al voltant de cada vèrtebra (mm)
TARGET_SPACING = (1.0, 1.0, 1.0)  # Resolució isòtropa final (mm)
HU_MIN        = -200   # Finestra òssia mínima (Hounsfield Units)
HU_MAX        = 1800   # Finestra òssia màxima (Hounsfield Units)
MIN_VOXELS    = 100    # Mínim de vòxels per considerar un segment vàlid


# ─── FUNCIONS DE PREPROCESSAT ─────────────────────────────────────────────────

def mask_from_seg_nrrd_sitk(
    seg_path: str,
    segment: dict,
    data_ndim: int,
    ct_sitk: sitk.Image,
) -> np.ndarray:
    """
    Llegeix una màscara de segment usant SimpleITK (eixos sempre en Z,Y,X com el CT).

    pynrrd retorna l'array en ordre natiu del fitxer (X,Y,Z), mentre que SimpleITK
    sempre retorna (Z,Y,X) independentment del format. Per tant, usar SimpleITK
    per llegir la màscara garanteix que els eixos coincideixin amb el CT.

    Fa sitk.Resample al grid del CT per gestionar diferències de resolució o mida.
    """
    seg_img = sitk.ReadImage(str(seg_path))
    label_val = int(segment["label_value"])

    if data_ndim == 4:
        # Fitxer 4D: SimpleITK el llegeix com a VectorImage
        # (un component = un layer de segments)
        n_comp = seg_img.GetNumberOfComponentsPerPixel()
        if n_comp > 1:
            layer = int(segment["layer"])
            layer_img = sitk.VectorIndexSelectionCast(seg_img, layer, sitk.sitkUInt8)
        else:
            # Fallback: intenta llegir com a 4D escalar i extreu el layer
            arr4d = sitk.GetArrayFromImage(seg_img)
            if arr4d.ndim == 4:
                layer_arr = arr4d[int(segment["layer"])].astype(np.float32)
            else:
                layer_arr = arr4d.astype(np.float32)
            layer_img = sitk.GetImageFromArray(layer_arr)
            layer_img.SetSpacing(seg_img.GetSpacing())
            layer_img.SetOrigin(seg_img.GetOrigin())
            layer_img.SetDirection(seg_img.GetDirection())
    else:
        # Fitxer 3D labelmap
        layer_img = sitk.Cast(seg_img, sitk.sitkUInt8)

    # Binaritza pel label_value del segment
    mask_img = sitk.BinaryThreshold(layer_img, label_val, label_val, 1, 0)

    # Resampleja al grid exacte del CT (nearest-neighbor per màscares binàries)
    resampled = sitk.Resample(
        sitk.Cast(mask_img, sitk.sitkFloat32),
        ct_sitk,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0.0,
        sitk.sitkFloat32,
    )
    arr = sitk.GetArrayFromImage(resampled)
    return (arr > 0.5).astype(np.uint8)


def normalize_ct(array: np.ndarray) -> np.ndarray:
    """Clipeja a finestra òssia i normalitza a [0, 1]."""
    clipped = np.clip(array, HU_MIN, HU_MAX)
    return ((clipped - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def get_bounding_box_mm(mask_sitk: sitk.Image, padding_mm: float = 15.0):
    """
    Calcula el bounding box d'una màscara en coordenades de vòxel,
    afegint un marge en mm convertit a vòxels.
    Retorna (idx_min, idx_max) en format (Z, Y, X) per slicing numpy.
    """
    spacing = mask_sitk.GetSpacing()  # (X, Y, Z)
    arr = sitk.GetArrayFromImage(mask_sitk)  # (Z, Y, X)

    coords = np.argwhere(arr > 0)
    if len(coords) == 0:
        return None

    # Padding en vòxels per cada eix (convertim mm a vòxels)
    pad_z = int(np.ceil(padding_mm / spacing[2]))
    pad_y = int(np.ceil(padding_mm / spacing[1]))
    pad_x = int(np.ceil(padding_mm / spacing[0]))

    z_min = max(0, coords[:, 0].min() - pad_z)
    z_max = min(arr.shape[0], coords[:, 0].max() + pad_z + 1)
    y_min = max(0, coords[:, 1].min() - pad_y)
    y_max = min(arr.shape[1], coords[:, 1].max() + pad_y + 1)
    x_min = max(0, coords[:, 2].min() - pad_x)
    x_max = min(arr.shape[2], coords[:, 2].max() + pad_x + 1)

    return (z_min, y_min, x_min), (z_max, y_max, x_max)


def crop_and_resample(
    ct_arr: np.ndarray,
    mask_arr: np.ndarray,
    bbox,
    ct_sitk_original: sitk.Image,
    interpolator_mask=sitk.sitkNearestNeighbor,
) -> tuple[sitk.Image, sitk.Image]:
    """
    Retalla CT i màscara al bounding box i resampleja a TARGET_SPACING.
    Preserva l'origen i la matriu de direcció del CT original perquè
    les prediccions quedin en el sistema de coordenades anatòmic correcte.
    Retorna (ct_resampled_sitk, mask_resampled_sitk).
    """
    (z0, y0, x0), (z1, y1, x1) = bbox

    ct_crop   = ct_arr[z0:z1, y0:y1, x0:x1]
    mask_crop = mask_arr[z0:z1, y0:y1, x0:x1]

    # Origen físic del vòxel (x0, y0, z0) en l'espai del CT original
    # TransformIndexToPhysicalPoint espera ordre (X, Y, Z)
    crop_origin    = ct_sitk_original.TransformIndexToPhysicalPoint(
        (int(x0), int(y0), int(z0))
    )
    crop_direction = ct_sitk_original.GetDirection()
    orig_spacing   = ct_sitk_original.GetSpacing()   # (X, Y, Z)

    def to_sitk(arr, interp):
        img = sitk.GetImageFromArray(arr.astype(np.float32))
        img.SetSpacing(orig_spacing)
        img.SetOrigin(crop_origin)
        img.SetDirection(crop_direction)

        orig_size = img.GetSize()          # (X, Y, Z)
        new_size = [
            int(round(orig_size[i] * orig_spacing[i] / TARGET_SPACING[i]))
            for i in range(3)
        ]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(TARGET_SPACING)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(crop_direction)
        resampler.SetOutputOrigin(crop_origin)
        resampler.SetInterpolator(interp)
        return resampler.Execute(img)

    ct_resampled   = to_sitk(ct_crop,   sitk.sitkLinear)
    mask_resampled = to_sitk(mask_crop, interpolator_mask)

    return ct_resampled, mask_resampled


# ─── PROCESSAMENT D'UN SUBJECTE ───────────────────────────────────────────────

def process_subject(
    subject_id: str,
    has_lamina: bool,
    case_counter: list,   # llista d'un element per poder modificar-la per referència
    dirs: dict,
    stats: dict,
) -> None:
    """
    Processa un subjecte complet: llegeix CT + segmentacions,
    genera un crop per vèrtebra i el guarda en format nnUNet.
    """
    print(f"\n{'─'*60}")
    print(f"Processant: {subject_id}  (làmina: {'Sí' if has_lamina else 'No'})")

    # ── Carrega CT ──────────────────────────────────────────────
    ct_path = CT_BASE / subject_id / "ct" / f"{subject_id}_ct.nii.gz"
    if not ct_path.exists():
        print(f"  [SKIP] CT no trobat: {ct_path}")
        stats["skipped"] += 1
        return

    ct_sitk   = sitk.ReadImage(str(ct_path))
    ct_arr    = sitk.GetArrayFromImage(ct_sitk)          # (Z, Y, X)
    ct_norm   = normalize_ct(ct_arr)
    spacing   = ct_sitk.GetSpacing()                     # (X, Y, Z) — per a bounding box en mm

    # ── Carrega segmentació de vèrtebres ────────────────────────
    vert_path = (SEG_BASE / subject_id.upper() /
                 f"{subject_id}_ct segmentation.seg.nrrd")
    if not vert_path.exists():
        print(f"  [SKIP] Segmentació de vèrtebres no trobada: {vert_path}")
        stats["skipped"] += 1
        return

    vert_data, vert_segments, _ = read_seg_nrrd(str(vert_path))
    print(f"  Vèrtebres trobades: {[s['name'] for s in vert_segments]}")

    # ── Carrega segmentació de làmines (si existeix) ─────────────
    lamina_data, lamina_segments = None, []
    if has_lamina:
        lam_path = (SEG_BASE / subject_id.upper() /
                    f"{subject_id}_ct_LAMINES.seg.nrrd")
        if lam_path.exists():
            lamina_data, lamina_segments, _ = read_seg_nrrd(str(lam_path))
            print(f"  Làmines trobades:  {[s['name'] for s in lamina_segments]}")
        else:
            print(f"  [WARN] Fitxer de làmines no trobat: {lam_path}")
            has_lamina = False

    # ── Processa cada vèrtebra ───────────────────────────────────
    for vert_seg in vert_segments:
        # Llegeix la màscara amb SimpleITK (eixos Z,Y,X correctes, com el CT)
        vert_mask_arr = mask_from_seg_nrrd_sitk(
            str(vert_path), vert_seg, vert_data.ndim, ct_sitk
        )

        if vert_mask_arr.sum() < MIN_VOXELS:
            print(f"    [SKIP] {vert_seg['name']}: massa pocs vòxels")
            continue

        # Bounding box de la vèrtebra
        vert_mask_sitk = sitk.GetImageFromArray(vert_mask_arr.astype(np.float32))
        vert_mask_sitk.SetSpacing(spacing)
        bbox = get_bounding_box_mm(vert_mask_sitk, padding_mm=PADDING_MM)
        if bbox is None:
            continue

        # Crop + resample CT i màscara de vèrtebra (passem ct_sitk per preservar origen/direcció)
        ct_crop_sitk, vert_crop_sitk = crop_and_resample(
            ct_norm, vert_mask_arr, bbox, ct_sitk
        )

        # Nom del cas: subject_vertebra (ex: "sub0001_L1")
        vert_name_clean = vert_seg["name"].replace(" ", "_")
        case_id = f"{subject_id.replace('-', '')}_{vert_name_clean}"

        if has_lamina and lamina_data is not None:
            # ── Obté màscara de làmina per aquesta vèrtebra ─────
            matching = find_matching_lamina_segments(vert_seg["name"], lamina_segments)

            if not matching:
                # Si no hi ha coincidència per nom, usa tota la làmina del subjecte
                # (menys precís però evita perdre dades)
                # Sense coincidència: usa tota la làmina del subjecte restringida a la vèrtebra
                lamina_mask_arr = np.zeros_like(ct_arr, dtype=np.uint8)
                for lseg in lamina_segments:
                    lm = mask_from_seg_nrrd_sitk(
                        str(lam_path), lseg, lamina_data.ndim, ct_sitk
                    )
                    lamina_mask_arr = np.maximum(lamina_mask_arr, lm)
                lamina_mask_arr = lamina_mask_arr * vert_mask_arr
            else:
                lamina_mask_arr = np.zeros_like(ct_arr, dtype=np.uint8)
                for lseg in matching:
                    lm = mask_from_seg_nrrd_sitk(
                        str(lam_path), lseg, lamina_data.ndim, ct_sitk
                    )
                    lamina_mask_arr = np.maximum(lamina_mask_arr, lm)

            _, lamina_crop_sitk = crop_and_resample(
                ct_norm, lamina_mask_arr, bbox, ct_sitk,
                interpolator_mask=sitk.sitkNearestNeighbor,
            )

            # Guarda a imagesTr (canal 0: CT, canal 1: vèrtebra) + labelsTr
            sitk.WriteImage(ct_crop_sitk,
                            str(dirs["imagesTr"] / f"{case_id}_0000.nii.gz"))
            sitk.WriteImage(vert_crop_sitk,
                            str(dirs["imagesTr"] / f"{case_id}_0001.nii.gz"))
            sitk.WriteImage(lamina_crop_sitk,
                            str(dirs["labelsTr"] / f"{case_id}.nii.gz"))

            stats["train"] += 1

        else:
            # Guarda a imagesTs (inferència futura)
            sitk.WriteImage(ct_crop_sitk,
                            str(dirs["imagesTs"] / f"{case_id}_0000.nii.gz"))
            sitk.WriteImage(vert_crop_sitk,
                            str(dirs["imagesTs"] / f"{case_id}_0001.nii.gz"))
            stats["test"] += 1

        case_counter[0] += 1
        print(f"    ✓ {vert_seg['name']}  →  {case_id}")


# ─── CREA dataset.json ────────────────────────────────────────────────────────

def create_dataset_json(output_dir: Path, n_train: int) -> None:
    """Crea el fitxer dataset.json que necessita nnUNet."""
    dataset = {
        "name": "LaminaVertebral",
        "description": "Segmentació de làmina vertebral - TFG Sara Masdeus",
        "tensorImageSize": "3D",
        "reference": "",
        "licence": "private",
        "release": "0.1",
        "channel_names": {
            "0": "CT_normalitzat",
            "1": "Mascara_vertebra",
        },
        "labels": {
            "background": 0,
            "lamina": 1,
        },
        "numTraining": n_train,
        "file_ending": ".nii.gz",
    }
    json_path = output_dir / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\ndataset.json creat: {json_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocessat pipeline làmina vertebral")
    parser.add_argument(
        "--ct_dir", type=Path, default=None,
        help="Carpeta amb els CTs (sub-XXXX/ct/sub-XXXX_ct.nii.gz). "
             "Si no s'especifica, usa el valor per defecte del script."
    )
    parser.add_argument(
        "--seg_dir", type=Path, default=None,
        help="Carpeta segmentacio_totalseg. "
             "Si no s'especifica, usa el valor per defecte del script."
    )
    parser.add_argument(
        "--out_dir", type=Path, default=None,
        help="Carpeta de sortida per al dataset nnUNet. "
             "Si no s'especifica, usa el valor per defecte del script."
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Mostra els noms dels segments de cada fitxer i surt (per diagnòstic)"
    )
    args = parser.parse_args()

    # Sobreescriu les rutes per defecte si s'han especificat per arguments
    global CT_BASE, SEG_BASE, OUT_BASE
    if args.ct_dir:
        CT_BASE = args.ct_dir
    if args.seg_dir:
        SEG_BASE = args.seg_dir
    if args.out_dir:
        OUT_BASE = args.out_dir

    # Mode inspecció: mostra noms de segments per ajudar a configurar el matching
    if args.inspect:
        print("\n=== MODE INSPECCIÓ: noms de segments als fitxers .seg.nrrd ===")
        for subj in SUBJECTS_WITH_VERTEBRA[:3]:  # Mostra primer 3 subjectes
            vert_path = (SEG_BASE / subj.upper() /
                         f"{subj}_ct segmentation.seg.nrrd")
            if vert_path.exists():
                print_segments_info(str(vert_path))
        for subj in SUBJECTS_WITH_LAMINA[:2]:
            lam_path = (SEG_BASE / subj.upper() /
                        f"{subj}_ct_LAMINES.seg.nrrd")
            if lam_path.exists():
                print_segments_info(str(lam_path))
        return

    # Crea estructura de carpetes nnUNet
    dirs = {
        "imagesTr": OUT_BASE / "imagesTr",
        "labelsTr": OUT_BASE / "labelsTr",
        "imagesTs": OUT_BASE / "imagesTs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    stats       = {"train": 0, "test": 0, "skipped": 0}
    case_counter = [0]

    print("=" * 60)
    print("PIPELINE DE PREPROCESSAT - LÀMINA VERTEBRAL")
    print("=" * 60)
    print(f"Sortida: {OUT_BASE}")
    print(f"Subjectes amb làmina (training): {SUBJECTS_WITH_LAMINA}")
    print(f"Subjectes sense làmina (test):   "
          f"{[s for s in SUBJECTS_WITH_VERTEBRA if s not in SUBJECTS_WITH_LAMINA]}")

    # ── Processa subjectes amb làmina (→ training) ───────────────
    for subj in SUBJECTS_WITH_VERTEBRA:
        has_lamina = subj in SUBJECTS_WITH_LAMINA
        process_subject(subj, has_lamina, case_counter, dirs, stats)

    # ── Crea dataset.json ─────────────────────────────────────────
    create_dataset_json(OUT_BASE, n_train=stats["train"])

    # ── Resum ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESUM FINAL")
    print(f"  Mostres d'entrenament (imagesTr + labelsTr): {stats['train']}")
    print(f"  Mostres d'inferència  (imagesTs):            {stats['test']}")
    print(f"  Subjectes saltats:                           {stats['skipped']}")
    print(f"  Total crops generats:                        {case_counter[0]}")
    print("=" * 60)
    print(f"\nDataset llest a: {OUT_BASE}")
    print("Proper pas: pujar la carpeta a Google Drive i entrenar amb nnUNet a Colab.")


if __name__ == "__main__":
    main()