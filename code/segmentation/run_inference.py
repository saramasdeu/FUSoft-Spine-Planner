#!/usr/bin/env python3
"""
Inferència local amb nnUNet
============================
Obre un selector d'arxius per triar una o múltiples imatges NIfTI
i fa la inferència amb el model entrenat.

Ús:
    python run_interference.py
"""

import os
import shutil
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path


# ─── CONFIGURACIÓ ─────────────────────────────────────────────────────────────

VENV_BIN    = Path("/Users/saramasdeusans/Desktop/TFG_FUSOFT/.venv/bin")
CHECKPOINT  = Path("/Users/saramasdeusans/Desktop/checkpoint_best.pth")
DATASET_DIR = Path("/Users/saramasdeusans/Desktop/nnunet_lamina")
RESULTS_DIR = Path("/Users/saramasdeusans/Desktop/nnunet_lamina/nnunet_results")
PREPROC_DIR = Path("/Users/saramasdeusans/Desktop/nnunet_lamina/nnunet_preprocessed")
PRED_DIR    = Path("/Users/saramasdeusans/Desktop/nnunet_predictions")
TEST_DIR    = Path("/tmp/lamina_inferencia")

CKPT_DIR = (RESULTS_DIR /
            "Dataset001_Lamina" /
            "nnUNetTrainer__nnUNetPlans__3d_fullres" /
            "fold_0")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def run(cmd: list, env=None):
    """Executa una comanda i mostra la sortida en temps real."""
    print(f"\n▶ {' '.join(str(c) for c in cmd)}\n{'─'*60}")
    result = subprocess.run(
        [str(c) for c in cmd],
        env=env or os.environ.copy(),
    )
    if result.returncode != 0:
        print(f"\n✗ Error (codi {result.returncode})")
        sys.exit(result.returncode)


def seleccionar_imatges() -> list[Path]:
    """
    Obre un selector d'arxius natiu del sistema.
    Retorna la llista de fitxers _0000.nii.gz seleccionats.
    """
    root = tk.Tk()
    root.withdraw()  # amaga la finestra principal de tkinter
    root.lift()
    root.attributes("-topmost", True)

    resposta = messagebox.askyesno(
        "Mode de selecció",
        "Vols seleccionar MÚLTIPLES imatges?\n\n"
        "Sí  → selecció múltiple\n"
        "No → selecció d'una sola imatge"
    )

    if resposta:
        # Selecció múltiple
        fitxers = filedialog.askopenfilenames(
            title="Selecciona les imatges NIfTI (_0000.nii.gz)",
            filetypes=[("NIfTI", "*_0000.nii.gz *.nii.gz"), ("Tots", "*")],
            initialdir=str(DATASET_DIR),
        )
    else:
        # Selecció d'una sola imatge
        fitxer = filedialog.askopenfilename(
            title="Selecciona una imatge NIfTI (_0000.nii.gz)",
            filetypes=[("NIfTI", "*_0000.nii.gz *.nii.gz"), ("Tots", "*")],
            initialdir=str(DATASET_DIR),
        )
        fitxers = [fitxer] if fitxer else []

    root.destroy()

    if not fitxers:
        print("✗ No s'ha seleccionat cap imatge. Sortint.")
        sys.exit(0)

    return [Path(f) for f in fitxers]


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("INFERÈNCIA LOCAL nnUNet — Làmina Vertebral")
    print("=" * 60)

    # 1. Comprova checkpoint
    if not CHECKPOINT.exists():
        print(f"✗ Checkpoint no trobat: {CHECKPOINT}")
        print("  Comprova que el fitxer és al Desktop amb aquest nom exacte.")
        sys.exit(1)

    # 2. Selector d'imatges
    print("\nObrint selector d'imatges...")
    imatges = seleccionar_imatges()
    print(f"\n✓ {len(imatges)} imatge(s) seleccionada(s):")
    for im in imatges:
        print(f"   → {im.name}")

    # 3. Crea carpetes
    print("\n1. Creant estructura de carpetes...")
    for d in [CKPT_DIR, PREPROC_DIR, PRED_DIR, TEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    # Buida el TEST_DIR d'execucions anteriors
    for f in TEST_DIR.iterdir():
        f.unlink()
    print("   ✓ Carpetes creades")

    # 4. Copia les imatges seleccionades al TEST_DIR
    # nnUNet necessita canal _0000 i _0001 per cada cas
    print("\n2. Preparant imatges per a la inferència...")
    for im in imatges:
        nom_base = im.name.replace("_0000.nii.gz", "").replace(".nii.gz", "")
        carpeta_origen = im.parent
        for canal in ["_0000.nii.gz", "_0001.nii.gz"]:
            src = carpeta_origen / f"{nom_base}{canal}"
            if src.exists():
                shutil.copy(src, TEST_DIR / f"{nom_base}{canal}")
                print(f"   ✓ {nom_base}{canal}")
            elif canal == "_0000.nii.gz":
                # Si el fitxer ja es diu sense canal, copia tal qual
                shutil.copy(im, TEST_DIR / f"{nom_base}_0000.nii.gz")
                print(f"   ✓ {nom_base}_0000.nii.gz")

    # 5. Copia checkpoint
    print("\n3. Copiant checkpoint...")
    shutil.copy(CHECKPOINT, CKPT_DIR / "checkpoint_best.pth")
    shutil.copy(CHECKPOINT, CKPT_DIR / "checkpoint_final.pth")
    print("   ✓ checkpoint_best.pth i checkpoint_final.pth")

    # Copia dataset.json
    dataset_json_src = DATASET_DIR / "Dataset001_Lamina" / "dataset.json"
    if dataset_json_src.exists():
        shutil.copy(dataset_json_src, CKPT_DIR.parent / "dataset.json")
        print("   ✓ dataset.json")

    # 6. Variables d'entorn
    env = os.environ.copy()
    env["nnUNet_raw"]          = str(DATASET_DIR)
    env["nnUNet_preprocessed"] = str(PREPROC_DIR)
    env["nnUNet_results"]      = str(RESULTS_DIR)

    # 7. Genera plans
    print("\n4. Generant plans nnUNet (2-3 min)...")
    run([VENV_BIN / "nnUNetv2_plan_and_preprocess",
         "-d", "001",
         "--verify_dataset_integrity"], env=env)

    plans_src = PREPROC_DIR / "Dataset001_Lamina" / "nnUNetPlans.json"
    if plans_src.exists():
        shutil.copy(plans_src, CKPT_DIR.parent / "plans.json")
        print(f"   ✓ plans.json copiat")
    else:
        print(f"✗ plans.json no trobat a {plans_src}")
        sys.exit(1)

    # 8. Detecta dispositiu
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n   Device: {device}")

    # 9. Inferència
    print(f"\n5. Fent inferència sobre {len(imatges)} imatge(s)...")
    run([VENV_BIN / "nnUNetv2_predict",
         "-i", str(TEST_DIR),
         "-o", str(PRED_DIR),
         "-d", "001",
         "-c", "3d_fullres",
         "-f", "0",
         "-device", device], env=env)

    # 10. Resultat
    print("\n" + "=" * 60)
    print("✓ INFERÈNCIA COMPLETADA!")
    print(f"  Prediccions guardades a: {PRED_DIR}")
    preds = sorted(PRED_DIR.glob("*.nii.gz"))
    for p in preds:
        print(f"  → {p.name}")
    print("\nObre els fitxers al 3D Slicer per veure les segmentacions!")
    print("=" * 60)


if __name__ == "__main__":
    main()
