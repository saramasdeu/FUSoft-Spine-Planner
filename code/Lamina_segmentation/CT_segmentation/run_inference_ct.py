#!/usr/bin/env python3
"""
Inferència local amb nnUNet
============================
Executa tot el pipeline: copia checkpoints, genera plans i fa inferència d'1 cas.

Ús:
    python run_inference_local.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# CONFIGURACIÓ 

VENV_BIN     = Path("/Users/saramasdeusans/Desktop/TFG_FUSOFT/.venv/bin")
CHECKPOINT   = Path("/Users/saramasdeusans/Desktop/102_20.50_checkpoint_best (1).pth") # ara amb el model que he entrenat amb volab
DATASET_DIR  = Path("/Users/saramasdeusans/Desktop/nnunet_lamina")
RESULTS_DIR  = Path("/Users/saramasdeusans/Desktop/nunnet_lamina/nnunet_results")
PREPROC_DIR  = Path("/Users/saramasdeusans/Desktop/nunnet_lamina/nnunet_preprocessed")
PRED_DIR     = Path("/Users/saramasdeusans/Desktop/nunnet_lamina/nnunet_predictions")
TEST_DIR     = Path("/tmp/lamina_1cas")

CKPT_DIR = (RESULTS_DIR /
            "Dataset001_Lamina" /
            "nnUNetTrainer__nnUNetPlans__3d_fullres" /
            "fold_0")

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def run(cmd: list, env=None):
    """Executa una comanda i mostra la sortida en temps real."""
    print(f"\n▶ {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(
        [str(c) for c in cmd],
        env=env or os.environ.copy(),
    )
    if result.returncode != 0:
        print(f"\n✗ Error (codi {result.returncode})")
        sys.exit(result.returncode)

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():

    # 1. Comprova que els fitxers necessaris existeixen
    print("=" * 60)
    print("INFERÈNCIA LOCAL nnUNet — Làmina Vertebral")
    print("=" * 60)

    if not CHECKPOINT.exists():
        print(f"✗ Checkpoint no trobat: {CHECKPOINT}")
        print("  Comprova que el fitxer és al Desktop amb aquest nom exacte.")
        sys.exit(1)

    images_ts = DATASET_DIR / "Dataset001_Lamina" / "imagesTs"
    if not images_ts.exists():
        print(f"✗ Dataset no trobat: {images_ts}")
        sys.exit(1)

    # 2. Crea carpetes
    print("\n1. Creant estructura de carpetes...")
    for d in [CKPT_DIR, PREPROC_DIR, PRED_DIR, TEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print("   ✓ Carpetes creades")

    # 3. Copia checkpoints i dataset.json
    print("\n2. Copiant checkpoints i dataset.json...")
    for nom in ["checkpoint_best.pth", "checkpoint_final.pth"]:
        dst = CKPT_DIR / nom
        shutil.copy(CHECKPOINT, dst)
        print(f"   ✓ {nom}")

    # dataset.json ha d'estar a la carpeta del model (no només a raw)
    dataset_json_src = DATASET_DIR / "Dataset001_Lamina" / "dataset.json"
    dataset_json_dst = CKPT_DIR.parent / "dataset.json"
    shutil.copy(dataset_json_src, dataset_json_dst)
    print(f"   ✓ dataset.json")

    # plans.json es genera amb plan_and_preprocess i cal copiar-lo al model
    # (es copia després del pas 4, see below)

    # 4. Configura variables d'entorn
    print("\n3. Configurant variables d'entorn...")
    env = os.environ.copy()
    env["nnUNet_raw"]          = str(DATASET_DIR)
    env["nnUNet_preprocessed"] = str(PREPROC_DIR)
    env["nnUNet_results"]      = str(RESULTS_DIR)
    print(f"   nnUNet_raw          = {DATASET_DIR}")
    print(f"   nnUNet_preprocessed = {PREPROC_DIR}")
    print(f"   nnUNet_results      = {RESULTS_DIR}")

    # 5. Genera plans (ràpid, no entrena)
    print("\n4. Generant plans nnUNet (2-3 min)...")
    run([VENV_BIN / "nnUNetv2_plan_and_preprocess",
         "-d", "001",
         "--verify_dataset_integrity"], env=env)

    # Copia plans.json al lloc on nnUNetv2_predict l'espera
    plans_src = PREPROC_DIR / "Dataset001_Lamina" / "nnUNetPlans.json"
    plans_dst = CKPT_DIR.parent / "plans.json"
    if plans_src.exists():
        shutil.copy(plans_src, plans_dst)
        print(f"   ✓ plans.json copiat a {plans_dst}")
    else:
        print(f"✗ plans.json no trobat a {plans_src}")
        sys.exit(1)

    # 6. Selecciona tots els crops de sub0006
    SUBJECTE = "sub0006"
    print(f"\n5. Seleccionant tots els crops de {SUBJECTE}...")
    casos = sorted(images_ts.glob(f"{SUBJECTE}*_0000.nii.gz"))
    if not casos:
        print(f"✗ No s'han trobat casos de {SUBJECTE} a {images_ts}")
        sys.exit(1)

    # Esborra contingut anterior del TEST_DIR
    for f in TEST_DIR.iterdir():
        f.unlink()

    for cas in casos:
        cas_nom = cas.name.replace("_0000.nii.gz", "")
        for canal in ["_0000.nii.gz", "_0001.nii.gz"]:
            src = images_ts / f"{cas_nom}{canal}"
            if src.exists():
                shutil.copy(src, TEST_DIR / f"{cas_nom}{canal}")
        print(f"   ✓ {cas_nom}")

    print(f"   Total: {len(casos)} vèrtebres")

    # 7. Inferència
    print("\n6. Fent inferència (10-20 min a CPU)...")
    run([VENV_BIN / "nnUNetv2_predict",
         "-i", str(TEST_DIR),
         "-o", str(PRED_DIR),
         "-d", "001",
         "-c", "3d_fullres",
         "-f", "0",
         "-device", "cpu"], env=env)

    # 8. Resultat
    print("\n" + "=" * 60)
    print("✓ INFERÈNCIA COMPLETADA!")
    print(f"Predicció guardada a: {PRED_DIR}")
    preds = list(PRED_DIR.glob("*.nii.gz"))
    for p in preds:
        print(f"  → {p.name}")
    print("\nObre el fitxer al 3D Slicer per veure la segmentació!")
    print("=" * 60)


if __name__ == "__main__":
    main()