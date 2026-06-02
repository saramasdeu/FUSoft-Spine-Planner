# FUSoft: Spine Planner

**FUSoft** is a surgical planning platform designed to optimize **Transcranial MR-guided Focused Ultrasound (TcMRgFUS)** treatments in the spine. The system automates image processing to enable precise ultrasound focusing, minimizing risks and maximizing thermal efficiency.

Developed as a Bachelor's Thesis (TFG) — 2026.

---

## Repository Structure

```
TFG_FUSOFT/
├── code/
│   ├── coregistration/          # MRI–CT co-registration (ANTs + SimpleITK)
│   ├── Pseudo_CTS/              # Pseudo-CT generation pipeline
│   │   ├── stinity_avaluator.py         # Interactive local tool (Mac)
│   │   ├── stinity_mr-to-pct/           # Sitiny model (submodule/clone)
│   │   ├── SynCT_TcMRgFUS/              # Han-Liu model (submodule/clone)
│   │   ├── dataset/
│   │   │   ├── creation_patches.py      # Generate MRI/CT patch pairs (.npy)
│   │   │   ├── finetune_sitiny.py       # Fine-tune ShuffleUNet on vertebrae patches
│   │   │   ├── evaluate_finetuned.py    # Compare pretrained vs fine-tuned (DB1)
│   │   │   └── evaluate_vertebrae.py    # Evaluate on vertebrae-only subject
│   │   ├── scripts/
│   │   │   ├── evaluate_sitiny.py       # Batch evaluation of Sitiny model (server)
│   │   │   ├── evaluate_hanliu.py       # Batch evaluation of Han-Liu model (server)
│   │   │   └── run_inference_fixed.py   # Fixed Han-Liu inference script (MONAI ≥1.0)
│   │   └── results/
│   │       ├── sitiny_eval/             # Figures + metrics (36 DB1 subjects)
│   │       └── finetuned_eval/          # Fine-tuning comparison CSV
│   ├── Lamina_segmentation/     # Vertebral lamina segmentation (nnU-Net)
│   └── Trajectories/            # FUS trajectory optimisation
└── models/                      # Model weights (not tracked — see below)
```

---

## Modules

### 1. Co-registration
MRI–CT rigid co-registration using **ANTs** and **SimpleITK**.

### 2. Pseudo-CT Generation

Two models were evaluated for MRI→pseudo-CT synthesis:

**2.1 Sitiny (ShuffleUNet)**
- Architecture: 3D ShuffleUNet (MONAI-based)
- Reference: [sitiny/mr-to-pct](https://github.com/sitiny/mr-to-pct)
- Pretrained weights: `pretrained_net_final_20220825.pth` (download from repo above)
- Results on DB1 (36 subjects): MAE = 190 HU, SSIM = 0.729

**2.2 Han-Liu (Pix2Pix 3D)**
- Architecture: ResNet-9blocks pix2pix with sliding window inference
- Reference: [han-liu/SynCT_TcMRgFUS](https://github.com/han-liu/SynCT_TcMRgFUS)
- Pretrained weights: download from repo above
- Note: Excluded from final evaluation due to domain gap (MAE > 250 HU on DB1)

**Fine-tuning**
- Attempted on 276 vertebrae patches (37 subjects DB1 + 5 subjects DB2)
- Did not improve results due to limited data and domain shift
- Proposed as future work

### 3. Lamina Segmentation
Vertebral lamina detection using **nnU-Net**, trained on paired MRI and CT data.

### 4. Trajectory Optimisation
FUS transducer trajectory planning based on pseudo-CT bone maps.

---

## Installation

Python 3.10+ recommended. Install dependencies:

```bash
pip install torch torchvision monai antspyx SimpleITK dipy nipype scikit-image matplotlib
```

Or using the project environment:

```bash
uv sync   # if using uv
# or
pip install -r code/Pseudo_CTS/requirements.txt
```

---

## Model Weights

Model weights are **not tracked** in this repository due to size constraints. Download them manually:

| Model | File | Source |
|-------|------|--------|
| Sitiny (pretrained) | `pretrained_net_final_20220825.pth` | [sitiny/mr-to-pct](https://github.com/sitiny/mr-to-pct) |
| Han-Liu | `best_net_G.pth` | [han-liu/SynCT_TcMRgFUS](https://github.com/han-liu/SynCT_TcMRgFUS) |
| nnU-Net MRI | `MRI_checkpoint_best.pth` | Trained locally |
| nnU-Net CT | `CT_checkpoint_best.pth` | Trained locally |

Place all weights in the `models/` directory.

---

## Usage

### Run pseudo-CT generation (local Mac)
```bash
cd code/Pseudo_CTS
python stinity_avaluator.py
```

### Batch evaluation on server
```bash
cd ~/FUSoft-Spine-Planner
python code/Pseudo_CTS/scripts/evaluate_sitiny.py
```

### Generate training patches
```bash
python code/Pseudo_CTS/dataset/creation_patches.py
```

### Fine-tune model
```bash
python code/Pseudo_CTS/dataset/finetune_sitiny.py
```

---

## References

- Han Liu et al., *SynCT: Synthetic CT Generation from MRI for TcMRgFUS Planning*, 2022. [GitHub](https://github.com/han-liu/SynCT_TcMRgFUS)
- Stiny et al., *MR-to-pCT: MRI to pseudo-CT synthesis using ShuffleUNet*, 2022. [GitHub](https://github.com/sitiny/mr-to-pct)
- Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*, Nature Methods, 2021.
