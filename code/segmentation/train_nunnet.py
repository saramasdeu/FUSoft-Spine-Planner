#!/usr/bin/env python3
"""
Entrenament nnUNet — Segmentació Làmina Vertebral
==================================================
Script autònom per entrenar el model nnUNet en un ordinador potent (GPU).
Substitueix el notebook de Google Colab.

Requisits previs
----------------
1. Python ≥ 3.10 amb venv creat i nnunetv2 instal·lat:
       python -m venv .venv
       source .venv/bin/activate          # Linux/Mac
       .venv\\Scripts\\activate             # Windows
       pip install nnunetv2

2. Arxiu ZIP del dataset preprocessat (Dataset001_Lamina.zip) a la mateixa
   carpeta que aquest script, o especificar la ruta amb --dataset_zip.

3. GPU NVIDIA amb drivers CUDA instal·lats (recomanat ≥ 8 GB VRAM).
   Si no hi ha GPU disponible, utilitza --device cpu (molt més lent).

Ús
---
    python train_nnunet.py                              # opcions per defecte
    python train_nnunet.py --dataset_zip /ruta/Dataset001_Lamina.zip
    python train_nnunet.py --epochs 500 --fold 0
    python train_nnunet.py --resume                     # continua des del checkpoint_latest
    python train_nnunet.py --device cpu                 # sense GPU

Estructura de carpetes generada
--------------------------------
    nnunet_raw/           ← dataset descomprimit
    nnunet_preprocessed/  ← plans + preprocessat nnUNet
    nnunet_results/       ← checkpoints i logs d'entrenament
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


# ─── CONFIGURACIÓ PER DEFECTE ─────────────────────────────────────────────────

# Canvia BASE_DIR a la carpeta on vols guardar tot el treball nnUNet
BASE_DIR     = Path.home() / "nnunet_work"

DATASET_ID   = "001"
DATASET_NAME = "Dataset001_Lamina"
CONFIG       = "3d_fullres"
TRAINER      = "nnUNetTrainer"
PLANS        = "nnUNetPlans"


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def run(cmd: list, env: dict | None = None) -> None:
    """Executa una comanda i mostra la sortida en temps real. Surt si hi ha error."""
    print(f"\n▶ {' '.join(str(c) for c in cmd)}\n{'─'*60}")
    result = subprocess.run(
        [str(c) for c in cmd],
        env=env or os.environ.copy(),
    )
    if result.returncode != 0:
        print(f"\n✗ Error (codi de sortida: {result.returncode})")
        sys.exit(result.returncode)


def find_nnunet_bin(venv_bin: Path | None) -> Path:
    """
    Troba el binari nnUNetv2_train.
    Prioritza el venv especificat, llavors el PATH del sistema.
    """
    candidates = []
    if venv_bin:
        candidates.append(venv_bin / "nnUNetv2_train")
    # Cerca al PATH
    system_bin = shutil.which("nnUNetv2_train")
    if system_bin:
        candidates.append(Path(system_bin).parent)

    for c in candidates:
        if isinstance(c, Path) and c.exists():
            return c.parent   # retorna la carpeta bin
        if isinstance(c, Path) and (c / "nnUNetv2_train").exists():
            return c

    print(
        "✗ No s'ha trobat nnUNetv2_train.\n"
        "  Assegura't que el venv amb nnunetv2 estigui activat, o especifica\n"
        "  la ruta amb --venv_bin /ruta/a/.venv/bin"
    )
    sys.exit(1)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena nnUNet per segmentació de làmina vertebral",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_zip", type=Path,
        default=Path(__file__).parent / f"{DATASET_NAME}.zip",
        help="Ruta al ZIP del dataset preprocessat",
    )
    parser.add_argument(
        "--base_dir", type=Path, default=BASE_DIR,
        help="Carpeta arrel on es crearan nnunet_raw, _preprocessed i _results",
    )
    parser.add_argument(
        "--venv_bin", type=Path, default=None,
        help="Carpeta bin del venv (ex: /ruta/.venv/bin). Si no s'especifica, "
             "cerca nnUNetv2_train al PATH del sistema.",
    )
    parser.add_argument(
        "--epochs", type=int, default=1000,
        help="Nombre d'epochs d'entrenament",
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Fold de cross-validation (0–4)",
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "mps"], default="cuda",
        help="Dispositiu de còmput (cuda=GPU NVIDIA, mps=GPU Apple Silicon)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continua l'entrenament des del checkpoint_latest (si existeix)",
    )
    parser.add_argument(
        "--skip_preprocess", action="store_true",
        help="Salta plan_and_preprocess (si ja s'ha executat abans)",
    )
    args = parser.parse_args()

    # ── Configuració de carpetes ──────────────────────────────────────────────
    raw_dir    = args.base_dir / "nnunet_raw"
    preproc_dir = args.base_dir / "nnunet_preprocessed"
    results_dir = args.base_dir / "nnunet_results"

    for d in [raw_dir, preproc_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Variables d'entorn necessàries per nnUNet
    env = os.environ.copy()
    env["nnUNet_raw"]          = str(raw_dir)
    env["nnUNet_preprocessed"] = str(preproc_dir)
    env["nnUNet_results"]      = str(results_dir)

    print("=" * 60)
    print("ENTRENAMENT nnUNet — Làmina Vertebral")
    print("=" * 60)
    print(f"  nnUNet_raw          = {raw_dir}")
    print(f"  nnUNet_preprocessed = {preproc_dir}")
    print(f"  nnUNet_results      = {results_dir}")
    print(f"  Device              = {args.device}")
    print(f"  Epochs              = {args.epochs}")
    print(f"  Fold                = {args.fold}")
    print(f"  Reprendre           = {args.resume}")

    # ── Troba binaris nnUNet ──────────────────────────────────────────────────
    bin_dir = find_nnunet_bin(args.venv_bin)
    print(f"\n  Binaris nnUNet a: {bin_dir}")

    # ── Descomprimeix dataset ─────────────────────────────────────────────────
    dataset_dest = raw_dir / DATASET_NAME
    if not dataset_dest.exists():
        if not args.dataset_zip.exists():
            print(f"\n✗ No s'ha trobat el ZIP del dataset: {args.dataset_zip}")
            print("  Especifica la ruta amb --dataset_zip /ruta/Dataset001_Lamina.zip")
            sys.exit(1)
        print(f"\n1. Descomprimint dataset a {raw_dir} ...")
        with zipfile.ZipFile(args.dataset_zip, "r") as zf:
            zf.extractall(raw_dir)
        print("   ✓ Dataset descomprimit")
    else:
        print(f"\n1. Dataset ja present a {dataset_dest} — saltant descompressió")

    # Verifica estructura mínima
    for subdir in ["imagesTr", "labelsTr", "imagesTs", "dataset.json"]:
        path = dataset_dest / subdir
        if not path.exists():
            print(f"\n✗ Estructura del dataset incorrecta: falta {path}")
            sys.exit(1)
    print("   ✓ Estructura del dataset verificada")

    # ── Plan & preprocess ─────────────────────────────────────────────────────
    plans_file = preproc_dir / DATASET_NAME / "nnUNetPlans.json"
    if args.skip_preprocess and plans_file.exists():
        print("\n2. Saltant plan_and_preprocess (--skip_preprocess activat)")
    else:
        print(f"\n2. Executant plan_and_preprocess (pot trigar 5-10 min) ...")
        run([
            bin_dir / "nnUNetv2_plan_and_preprocess",
            "-d", DATASET_ID,
            "--verify_dataset_integrity",
        ], env=env)
        print("   ✓ Plans i preprocessat generats")

    # ── Entrenament ───────────────────────────────────────────────────────────
    print(f"\n3. Iniciant entrenament (fold {args.fold}, {args.epochs} epochs) ...")
    print("   Els checkpoints es guardaran automàticament a:")
    ckpt_dir = (results_dir / DATASET_NAME /
                f"{TRAINER}__{PLANS}__{CONFIG}" / f"fold_{args.fold}")
    print(f"   {ckpt_dir}\n")

    train_cmd = [
        bin_dir / "nnUNetv2_train",
        DATASET_ID,
        CONFIG,
        str(args.fold),
        "--npz",                   # guarda softmax per a possibles ensembles
        "-device", args.device,
        "--num_epochs", str(args.epochs),  # funciona a nnunetv2 ≥ 2.3
    ]
    if args.resume:
        train_cmd.append("--c")    # continua des de checkpoint_latest

    run(train_cmd, env=env)

    # ── Resum final ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✓ ENTRENAMENT COMPLETAT!")
    print(f"  Checkpoints a: {ckpt_dir}")
    if ckpt_dir.exists():
        for ckpt in sorted(ckpt_dir.glob("*.pth")):
            size_mb = ckpt.stat().st_size / 1024 / 1024
            print(f"    → {ckpt.name}  ({size_mb:.1f} MB)")
    print("\nProper pas:")
    print("  1. Copia checkpoint_best.pth i checkpoint_latest.pth al Desktop de Sara")
    print("  2. Executa run_inference_local.py per fer la inferència")
    print("=" * 60)


if __name__ == "__main__":
    main()