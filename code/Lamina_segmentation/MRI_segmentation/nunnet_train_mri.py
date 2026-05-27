#!/usr/bin/env python3
"""
Entrenament nnUNet — Làmina Vertebral MRI
==========================================
Executa tot el pipeline: genera plans, preprocessa i entrena el model MRI.

Ús:
    python train_nnunet_mri.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ─── CONFIGURACIÓ ─────────────────────────────────────────────────────────────

VENV_BIN    = Path("/Users/saramasdeusans/Desktop/TFG_FUSOFT/.venv/bin")
DATASET_DIR = Path("/Users/saramasdeusans/Desktop/NUNNET_WORK/MRI")
PREPROC_DIR = Path("/Users/saramasdeusans/Desktop/NUNNET_WORK/MRI/nnunet_preprocessed")
RESULTS_DIR = Path("/Users/saramasdeusans/Desktop/NUNNET_WORK/MRI/nnunet_results")

DATASET_ID   = "002"
DATASET_NAME = "Dataset002_Lamina"
CONFIG       = "3d_fullres"
FOLD         = "0"

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
    print("ENTRENAMENT nnUNet — Làmina Vertebral MRI")
    print("=" * 60)

    # 1. Comprova dataset
    dataset_path = DATASET_DIR / DATASET_NAME
    if not dataset_path.exists():
        print(f"✗ Dataset no trobat: {dataset_path}")
        print("  Executa primer preprocess_lamina_mri.py")
        sys.exit(1)

    images_tr = dataset_path / "imagesTr"
    n_cases = len(list(images_tr.glob("*_0000.nii.gz")))
    print(f"\n✓ Dataset trobat: {DATASET_NAME}")
    print(f"  Casos entrenament: {n_cases}")

    # 2. Crea carpetes
    for d in [PREPROC_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 3. Variables d'entorn nnUNet
    env = os.environ.copy()
    env["nnUNet_raw"]          = str(DATASET_DIR)
    env["nnUNet_preprocessed"] = str(PREPROC_DIR)
    env["nnUNet_results"]      = str(RESULTS_DIR)
    print(f"\n   nnUNet_raw          = {DATASET_DIR}")
    print(f"   nnUNet_preprocessed = {PREPROC_DIR}")
    print(f"   nnUNet_results      = {RESULTS_DIR}")

    # 4. Plan and preprocess
    print(f"\n1. Generant plans nnUNet (2-5 min)...")
    run([
        VENV_BIN / "nnUNetv2_plan_and_preprocess",
        "-d", DATASET_ID,
        "--verify_dataset_integrity"
    ], env=env)

    # 5. Entrenament
    print(f"\n2. Iniciant entrenament (fold {FOLD})...")
    print(f"   Config: {CONFIG}")
    print(f"   Això pot trigar hores — millor a l'ordinador extern amb GPU!\n")
    run([
        VENV_BIN / "nnUNetv2_train",
        DATASET_ID,
        CONFIG,
        FOLD,
        "--npz"
    ], env=env)

    # 6. Localitza checkpoint
    ckpt_dir = (RESULTS_DIR /
                f"Dataset{DATASET_ID}_Lamina" /
                f"nnUNetTrainer__nnUNetPlans__{CONFIG}" /
                f"fold_{FOLD}")
    checkpoint = ckpt_dir / "checkpoint_best.pth"

    print("\n" + "=" * 60)
    print("✓ ENTRENAMENT COMPLETAT!")
    if checkpoint.exists():
        print(f"  Checkpoint: {checkpoint}")
    else:
        print(f"  Busca el checkpoint a: {ckpt_dir}")
    print("\nPropera passa: executa run_inference_mri.py")
    print("=" * 60)


if __name__ == "__main__":
    main()