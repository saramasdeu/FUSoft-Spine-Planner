"""
Utilitats per llegir fitxers .seg.nrrd de 3D Slicer
=====================================================
Els fitxers .seg.nrrd poden tenir dos formats interns:
  - Format 4D: cada segment és una capa (layer) binària independent
  - Format 3D: tots els segments comparteixen una capa, cadascun amb un label_value diferent

Aquesta funció gestiona els dos casos automàticament.
"""

import numpy as np
import nrrd


def read_seg_nrrd(filepath: str) -> tuple[np.ndarray, list[dict], dict]:
    """
    Llegeix un fitxer .seg.nrrd de 3D Slicer.

    Retorna:
        data    : array numpy tal com el retorna pynrrd
        segments: llista de dicts amb info de cada segment:
                  {'name': str, 'layer': int, 'label_value': int}
        header  : header complet del fitxer nrrd (necessari per metadata espacial)
    """
    data, header = nrrd.read(str(filepath))

    # Extrau informació de cada segment del header
    segments = []
    i = 0
    while f"Segment{i}_Name" in header:
        seg = {
            "name":        header[f"Segment{i}_Name"],
            "layer":       int(header.get(f"Segment{i}_Layer", 0)),
            "label_value": int(header.get(f"Segment{i}_LabelValue", i + 1)),
        }
        segments.append(seg)
        i += 1

    if not segments:
        raise ValueError(f"No s'han trobat segments al fitxer: {filepath}")

    return data, segments, header


def get_segment_mask(data: np.ndarray, segment: dict) -> np.ndarray:
    """
    Extreu la màscara binària (0/1) d'un segment concret.

    Gestiona automàticament format 3D (labelmap) i 4D (capes separades).
    """
    if data.ndim == 4:
        # Format 4D: (N_layers, Z, Y, X)
        layer = segment["layer"]
        mask = (data[layer] == segment["label_value"]).astype(np.uint8)
    elif data.ndim == 3:
        # Format 3D labelmap: (Z, Y, X)
        mask = (data == segment["label_value"]).astype(np.uint8)
    else:
        raise ValueError(f"Format de dades no suportat: {data.ndim}D")

    return mask


def get_combined_mask(data: np.ndarray, segments: list[dict]) -> np.ndarray:
    """
    Combina tots els segments en una sola màscara binària.
    Útil per obtenir la màscara total de vèrtebra o làmina sense distingir labels.
    """
    combined = np.zeros(data.shape[-3:], dtype=np.uint8)
    for seg in segments:
        combined = np.maximum(combined, get_segment_mask(data, seg))
    return combined


def get_labelmap(data: np.ndarray, segments: list[dict]) -> np.ndarray:
    """
    Crea un labelmap 3D on cada segment té un valor únic (1, 2, 3...).
    Útil per visualitzar tots els segments junts.
    """
    labelmap = np.zeros(data.shape[-3:], dtype=np.uint8)
    for idx, seg in enumerate(segments, start=1):
        mask = get_segment_mask(data, seg)
        labelmap[mask > 0] = idx
    return labelmap


def find_matching_lamina_segments(
    vertebra_name: str,
    lamina_segments: list[dict],
) -> list[dict]:
    """
    Troba els segments de làmina que corresponen a una vèrtebra concreta.

    Estratègia: comparació flexible de noms (case-insensitive, sense espais).
    Exemples de coincidències:
        vertebra "L1"  ↔  làmina "Lamina_L1", "L1_lamina", "l1"
        vertebra "T5"  ↔  làmina "T5", "Th5", "t5_right"
    """
    v_name = vertebra_name.lower().replace(" ", "").replace("_", "")
    matches = []
    for lseg in lamina_segments:
        l_name = lseg["name"].lower().replace(" ", "").replace("_", "")
        if v_name in l_name or l_name in v_name:
            matches.append(lseg)
    return matches


def print_segments_info(filepath: str) -> None:
    """
    Funció de diagnòstic: imprimeix els segments trobats en un fitxer .seg.nrrd.
    Útil per verificar que els noms dels segments són correctes.
    """
    data, segments, header = read_seg_nrrd(filepath)
    print(f"\nFitxer: {filepath}")
    print(f"Dimensions array: {data.shape}")
    print(f"Space directions: {header.get('space directions')}")
    print(f"Space origin:     {header.get('space origin')}")
    print(f"Segments trobats ({len(segments)}):")
    for i, seg in enumerate(segments):
        mask = get_segment_mask(data, seg)
        n_voxels = mask.sum()
        print(f"  [{i}] nom='{seg['name']}' | layer={seg['layer']} | "
              f"label={seg['label_value']} | vòxels={n_voxels:,}")