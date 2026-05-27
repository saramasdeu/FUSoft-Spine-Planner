#!/usr/bin/env python3
"""
Inferència nnUNet — Làmina Vertebral MRI
=========================================
Executa la predicció sobre els crops de test (imagesTs) del Dataset002_Lamina.

Ús:
    python run_inference_mri.py
"""

import os
import subprocess
import sys
from pathlib import Path

# ─── CONFIGURACIÓ ─────────────────────────────────────────────────────────────

VENV_BIN    = Path("/home/sara/FUSoft-Spine-Planner/.venv/bin")
DATASET_DIR = Path("/home/sara")                    # nnUNet_raw (conté Dataset002_Lamina)
PREPROC_DIR = Path("/home/sara/FUSoft-Spine-Planner/MRI_segmentation/NUNNET_WORK/MRI/nnunet_preprocessed")
RESULTS_DIR = Path("/home/sara/FUSoft-Spine-Planner/MRI_segmentation/NUNNET_WORK/MRI/nnunet_results")

INPUT_DIR   = Path("/home/sara/Dataset002_Lamina/imagesTs")   # crops sense làmina
OUTPUT_DIR  = Path("/home/sara/RESULTATS_NNUNET/MRI")         # on es guardaran les prediccions

DATASET_ID  = "002"
CONFIG      = "3d_fullres"
FOLD        = "0"

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def run(cmd: list, env=None):
    print(f"\n▶ {' '.join(str(c) for c in cmd)}\n{'─'*60}")
    result = subprocess.run([str(c) for c in cmd], env=env or os.environ.copy())
    if result.returncode != 0:
        print(f"\n✗ Error (codi {result.returncode})")
        sys.exit(result.returncode)

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("INFERÈNCIA nnUNet — Làmina Vertebral MRI")
    print("=" * 60)

    # 1. Comprova que existeix la carpeta d'entrada
    if not INPUT_DIR.exists():
        print(f"✗ Carpeta d'entrada no trobada: {INPUT_DIR}")
        print("  Executa primer preprocess_lamina_mri.py")
        sys.exit(1)

    n_crops = len(list(INPUT_DIR.glob("*_0000.nii.gz")))
    print(f"\n✓ Crops de test trobats: {n_crops}")
    print(f"  Entrada: {INPUT_DIR}")
    print(f"  Sortida: {OUTPUT_DIR}")

    # 2. Crea carpeta de sortida
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Variables d'entorn nnUNet
    env = os.environ.copy()
    env["nnUNet_raw"]          = str(DATASET_DIR)
    env["nnUNet_preprocessed"] = str(PREPROC_DIR)
    env["nnUNet_results"]      = str(RESULTS_DIR)
    print(f"\n   nnUNet_raw          = {DATASET_DIR}")
    print(f"   nnUNet_preprocessed = {PREPROC_DIR}")
    print(f"   nnUNet_results      = {RESULTS_DIR}")

    # 4. Inferència
    print(f"\n▶ Iniciant inferència (fold {FOLD}, config {CONFIG})...")
    run([
        VENV_BIN / "nnUNetv2_predict",
        "-i", INPUT_DIR,
        "-o", OUTPUT_DIR,
        "-d", DATASET_ID,
        "-c", CONFIG,
        "-f", FOLD,
    ], env=env)

    # 5. Resum
    n_pred = len(list(OUTPUT_DIR.glob("*.nii.gz")))
    print("\n" + "=" * 60)
    print("✓ INFERÈNCIA COMPLETADA!")
    print(f"  Prediccions generades: {n_pred}")
    print(f"  Guardades a: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()