# FUSoft: Spine Planner

**FUSoft** is a comprehensive surgical planning platform designed to optimize **Transcranial Magnetic Resonance-guided Focused Ultrasound (TcMRgFUS)** treatments in the spine. The main objective is to automate image processing to allow precise ultrasound focusing, minimizing risks and maximizing thermal efficiency.

---

## Workflow Structure (Modules)

### 1. Co registration`
* **Tools**: Based in **ANTs** and **SimpleITK**.

### 2.1 Pseudo-CT (SynCT): 
* **Description**: Generation of synthetic Computed Tomography images from MRI for bone density calculation and phase correction.
* **Model**: Arquitectura GAN (Pix2Pix)
* **Carpeta**: `code/SynCT_TcMRgFUS/
https://github.com/han-liu/SynCT_TcMRgFUS.
### 2.2 Pseudo-CT (Stinity):
https://github.com/sitiny/mr-to-pct

### 3. Segmentation and Detection of Intervertebral Windows
`
### 4. Trajectory Calculation


### Data Requirements (Model Weights)
Due to GitHub size limitations, AI model weight files (.pth) must be downloaded externally:

1. **Model SynCT**:`best_net_G.pth` from [(https://github.com/han-liu/SynCT_TcMRgFUS.)].
2. **Model Stinity** [(https://github.com/sitiny/mr-to-pct)]

### General Installation

**Python 3.10** environment is recommended.

```bash
# Install global dependencies
pip install torch torchvision monai antspyx SimpleITK dipy nibype scikit-image matplotlib
