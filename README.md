# FUSoft: Spine Planner

**FUSoft** is a surgical planning platform designed to optimize **Transcranial MR-guided Focused Ultrasound (TcMRgFUS)** treatments in the spine. The system automates image processing to enable precise ultrasound focusing, minimizing risks and maximizing thermal efficiency.

Developed as a Bachelor's Thesis (TFG) — 2026.

---

## Repository Structure

```
TFG_FUSOFT/
├── code/
│   ├── coregistration/                  # MRI–CT co-registration (ANTs + SimpleITK)
│   │   ├── Co-registre.py               # Core co-registration script. Aligns MRI
│   │   │                                # and CT images using rigid registration
│   │   │                                # (ANTs + SimpleITK).
│   │   └── Coregistration(pyqt6).py     # Graphical user interface (PyQt6) for the
│   │                                    # co-registration pipeline. Intended as the
│   │                                    # end-user tool: allows loading MRI and CT
│   │                                    # files, launching registration, and saving
│   │                                    # the result without using the command line.
│   ├── Pseudo_CTS/              # Pseudo-CT generation pipeline
│   │   │
│   │   ├── stinity_avaluator.py         # Main local inference tool (Mac).
│   │   │                                # Loads a T1 MRI, runs ShuffleUNet and
│   │   │                                # saves the pseudo-CT as a NIfTI file.
│   │   │
│   │   ├── stinity_mr-to-pct/           # Original Sitiny repository (cloned).
│   │   │   │                            # Do not modify — used as a library.
│   │   │   └── utils/
│   │   │       ├── netdef.py            # ShuffleUNet architecture definition
│   │   │       ├── infer_funcs.py       # Inference pipeline (with ANTs preprocessing)
│   │   │       └── infer_funcs_noants.py# Inference pipeline (server, without ANTs)
│   │   │
│   │   ├── SynCT_TcMRgFUS/             # Original Han-Liu repository (cloned).
│   │   │                                # Pix2Pix 3D model evaluated but excluded
│   │   │                                # from final results due to domain gap.
│   │   │
│   │   ├── dataset/                     # Scripts for patch generation and fine-tuning
│   │   │   ├── creation_patches.py      # Extracts paired MRI/CT patches (.npy) from
│   │   │   │                            # DS1 (37 subjects) and DS2 (5 vertebrae subjects),
│   │   │   │                            # centred on the vertebrae segmentation mask.
│   │   │   │
│   │   │   ├── finetune_sitiny.py       # Fine-tunes the pretrained ShuffleUNet on the
│   │   │   │                            # generated vertebrae patches. Uses knowledge
│   │   │   │                            # distillation (frozen teacher = pretrained model)
│   │   │   │                            # to prevent catastrophic forgetting.
│   │   │   │
│   │   │   ├── evaluate_finetuned.py    # Compares pretrained vs fine-tuned model on all
│   │   │   │                            # DB1 subjects (36). Computes MAE and SSIM both
│   │   │   │                            # globally and in the vertebrae region specifically.
│   │   │   │                            # Saves comparison figures and a CSV with results.
│   │   │   │
│   │   │   └── evaluate_vertebrae.py    # Evaluates both models on a single vertebrae-only
│   │   │                                # subject (DS2: CT_1.nii + MRI_1.nii).
│   │   │
│   │   ├── scripts/                     # Batch scripts designed to run on the GPU server
│   │   │   └── evaluate_sitiny.py       # Runs full inference + metrics for the pretrained
│   │   │                                # Sitiny model over all 36 DB1 subjects.
│   │   │                                # Saves per-subject figures and a summary CSV.
│   │   │
│   │   └── results/                     # Generated outputs (not tracked in Git)
│   │       ├── sitiny_eval/             # Figures and metrics for the 36 DB1 subjects
│   │       └── finetuned_eval/          # Pretrained vs fine-tuned comparison CSV and figures
│   │
│   ├── Lamina_segmentation/             # Vertebral lamina segmentation (nnU-Net)
│   │   │
│   │   ├── segmenta_lamina.py           # Main end-to-end pipeline (CT or MRI).
│   │   │                                # Auto-detects modality from HU values,
│   │   │                                # runs TotalSegmentator for vertebra
│   │   │                                # detection, crops per vertebra, runs
│   │   │                                # the appropriate nnUNet model, and
│   │   │                                # reconstructs the full lamina mask in
│   │   │                                # the original image space (.nii.gz).
│   │   │
│   │   ├── CT_segmentation/             # CT branch — Dataset001_Lamina
│   │   │   ├── preprocess_lamina_ct.py  # Builds the nnUNet CT dataset from raw
│   │   │   │                            # CTs + 3D Slicer annotations (.seg.nrrd).
│   │   │   │                            # Generates per-vertebra crops with 2
│   │   │   │                            # channels (normalised CT + vertebra mask),
│   │   │   │                            # resampled to 1 mm isotropic. 20 subjects:
│   │   │   │                            # 5 with lamina → imagesTr/labelsTr,
│   │   │   │                            # 15 without → imagesTs.
│   │   │   │
│   │   │   ├── train_nunnet_ct.py       # Standalone training script for the CT
│   │   │   │                            # model. Unpacks Dataset001_Lamina.zip,
│   │   │   │                            # runs nnUNetv2_plan_and_preprocess, then
│   │   │   │                            # nnUNetv2_train (3d_fullres, fold 0).
│   │   │   │                            # Supports GPU auto-detection (CUDA / MPS /
│   │   │   │                            # CPU) and resume from checkpoint.
│   │   │   │
│   │   │   ├── run_inference_ct.py      # Local inference script for CT crops.
│   │   │   │                            # Copies checkpoint_best.pth, generates
│   │   │   │                            # nnUNet plans, and runs nnUNetv2_predict
│   │   │   │                            # on imagesTs cases (CPU or GPU).
│   │   │   │
│   │   │   └── seg_nrrd_utils_ct.py     # Utility library for reading 3D Slicer
│   │   │                                # .seg.nrrd files (3D labelmap and 4D
│   │   │                                # layer formats). Provides read_seg_nrrd,
│   │   │                                # get_segment_mask, get_combined_mask and
│   │   │                                # find_matching_lamina_segments helpers.
│   │   │                                # Used by preprocess_lamina_ct.py.
│   │   │
│   │   └── MRI_segmentation/            # MRI branch — Dataset002_Lamina
│   │       ├── preprocess_mri.py        # Builds the nnUNet MRI dataset from FLAIR
│   │       │                            # images + 3D Slicer annotations. Generates
│   │       │                            # per-vertebra crops with 2 channels (FLAIR
│   │       │                            # + vertebra mask), 1 mm isotropic. 20
│   │       │                            # subjects: 10 with lamina → imagesTr/
│   │       │                            # labelsTr, 10 without → imagesTs.
│   │       │
│   │       ├── nunnet_train_mri.py      # Training script for the MRI model.
│   │       │                            # Runs plan_and_preprocess and
│   │       │                            # nnUNetv2_train on Dataset002_Lamina
│   │       │                            # (3d_fullres, fold 0).
│   │       │
│   │       └── run_inference_mri.py     # Runs nnUNetv2_predict on the MRI test
│   │                                    # crops (Dataset002_Lamina/imagesTs) and
│   │                                    # saves per-vertebra predictions.
│   │
│   └── Trajectories/                    # FUS trajectory optimisation
│       └── optimization.py              # Computes the optimal FUS transducer
│                                        # trajectory to reach a spinal cord target.
│                                        # Uses a heuristic approach over the
│                                        # pseudo-CT bone map to find the acoustic
│                                        # path with minimal bone obstruction.
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
- Stiny et al., *MR-to-pCT: MRI to pseudo-CT synthesis using ShuffleUNet*, 2022. [GitHub](https://github.com/sitiny/mr-to-pct)
- Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*, Nature Methods, 2021.
