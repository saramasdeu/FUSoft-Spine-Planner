"""
creation_patches.py
-------------------
Generates paired MRI/CT training patches from two sources:

  1. coregistration_and_segmentation/
       - Selected subjects only (SELECTED_IDS)
       - Each subject folder contains:
           *mri_reference.nii.gz
           *ct_registered.nii.gz
           *segmentation.seg.nrrd   (vertebra mask → patch centres)
       - Patches are centred on the vertebra segmentation mask.

  2. Dataset vertebres format ok/
       - CT_n.nii + MRI_n.nii  (n = 1..5)
       - Images are already cropped to the vertebra region.
       - Patches are sampled randomly across the volume.

All patches are saved as .npy pairs:
    <output_dir>/<subject_id>_patch_<i>_mri.npy
    <output_dir>/<subject_id>_patch_<i>_ct.npy
"""

import os
import glob
import numpy as np
import SimpleITK as sitk

# ===========================================================
# CONFIGURATION
# ===========================================================
COREGISTRATION_DIR = "/Users/saramasdeusans/Desktop/coregistration_and_segmentation"
VERTEBRAE_DIR      = "/Users/saramasdeusans/Desktop/Dataset vertebres format ok"
OUTPUT_DIR         = "/Users/saramasdeusans/Desktop/DATASET_NET/patches"

# Subjects selected from coregistration_and_segmentation (no metal artefacts)
SELECTED_IDS = [
    "sub0003", "sub0005", "sub0006", "sub0007", "sub0008", "sub0009",
    "sub0011", "sub0012", "sub0013", "sub0019", "sub0021", "sub0022",
    "sub0026", "sub0027", "sub0029", "sub0030", "sub0032", "sub0034",
    "sub0036", "sub0037",
]

PATCH_SIZE              = (128, 128, 64)   # (X, Y, Z) in voxels
PATCHES_PER_SUBJECT_DS1 = 6               # centred on segmentation mask
PATCHES_PER_SUBJECT_DS2 = 12              # random (already-cropped vertebrae)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================
# NORMALISATION
# ===========================================================
def normalise_mri(image: sitk.Image) -> np.ndarray:
    """
    Z-score normalisation (foreground only) followed by
    min-max rescaling to [-1, 1].
    This matches the input range expected by the pre-trained model.
    """
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    foreground = arr > 0
    if foreground.sum() == 0:
        return arr
    mean = arr[foreground].mean()
    std  = arr[foreground].std()
    arr  = (arr - mean) / (std + 1e-5)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 2.0 - 1.0
    return arr


def resample_to_reference(moving: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """
    Resample `moving` image into the physical space of `reference`
    (same origin, spacing, direction, and size).
    Uses linear interpolation — suitable for continuous-valued images (MRI, CT).
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(sitk.GetArrayFromImage(moving).min()))
    return resampler.Execute(moving)


def normalise_ct(image: sitk.Image) -> np.ndarray:
    """
    Clip to clinically relevant HU range [-1000, 2000],
    then rescale linearly to [-1, 1].
    Used for Dataset 1 (standard HU values).
    """
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    arr = np.clip(arr, -1000, 2000)
    arr = (arr + 1000) / 1500.0 - 1.0
    return arr


def normalise_ct_uint8(image: sitk.Image) -> np.ndarray:
    """
    Normalise CT saved as uint8 (range 0-255) to [-1, 1].
    Used for Dataset 2, where CTs were exported from raw files as 8-bit.
    Mapping: 0 → -1,  255 → 1
    """
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    arr = np.clip(arr, 0, 255)
    arr = arr / 127.5 - 1.0
    return arr


# ===========================================================
# PATCH EXTRACTION — centred on segmentation mask (Dataset 1)
# ===========================================================
def extract_patches_from_mask(
    mri: np.ndarray,
    ct: np.ndarray,
    mask: np.ndarray,
    subject_id: str,
    n_patches: int,
) -> int:
    """
    Sample `n_patches` patch centres uniformly from non-zero mask voxels.
    Each patch is clipped to the volume boundary and zero-padded if needed.
    Returns the number of patches successfully saved.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        print(f"      [WARN] Empty segmentation mask for {subject_id} — skipping.")
        return 0

    pz, py, px = PATCH_SIZE[2], PATCH_SIZE[1], PATCH_SIZE[0]
    saved = 0

    for i in range(n_patches):
        centre = coords[np.random.randint(len(coords))]

        z0 = int(np.clip(centre[0] - pz // 2, 0, mri.shape[0] - pz))
        y0 = int(np.clip(centre[1] - py // 2, 0, mri.shape[1] - py))
        x0 = int(np.clip(centre[2] - px // 2, 0, mri.shape[2] - px))

        mri_patch = mri[z0:z0+pz, y0:y0+py, x0:x0+px].copy()
        ct_patch  = ct [z0:z0+pz, y0:y0+py, x0:x0+px].copy()

        # Edge padding if the patch is smaller than expected
        def _pad(arr, target):
            pad = [(0, max(0, t - s)) for s, t in zip(arr.shape, target)]
            return np.pad(arr, pad, mode="edge") if any(p[1] > 0 for p in pad) else arr

        mri_patch = _pad(mri_patch, (pz, py, px))
        ct_patch  = _pad(ct_patch,  (pz, py, px))

        stem = os.path.join(OUTPUT_DIR, f"{subject_id}_patch_{i}")
        np.save(stem + "_mri.npy", mri_patch)
        np.save(stem + "_ct.npy",  ct_patch)
        saved += 1

    return saved


# ===========================================================
# PATCH EXTRACTION — random sampling (Dataset 2)
# ===========================================================
def resize_volume(arr: np.ndarray, target: tuple) -> np.ndarray:
    """
    Resize a 3D numpy array to `target` (Z, Y, X) using SimpleITK linear
    interpolation. Preserves content without padding artefacts.
    """
    tz, ty, tx = target
    img = sitk.GetImageFromArray(arr)
    resampler = sitk.ResampleImageFilter()
    orig_size    = np.array(img.GetSize(),    dtype=float)   # (X, Y, Z)
    orig_spacing = np.array(img.GetSpacing(), dtype=float)
    new_size     = [tx, ty, tz]
    new_spacing  = orig_spacing * orig_size / np.array([tx, ty, tz], dtype=float)
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(arr.min()))
    return sitk.GetArrayFromImage(resampler.Execute(img))


def extract_whole_volume(
    mri: np.ndarray,
    ct: np.ndarray,
    subject_id: str,
    n_augmented: int,
) -> int:
    """
    For already-cropped vertebra volumes (Dataset 2):
    resize the whole volume to PATCH_SIZE and save it as one patch.
    Optionally generate additional augmented versions via random flips.
    Returns the number of patches saved.
    """
    pz, py, px = PATCH_SIZE[2], PATCH_SIZE[1], PATCH_SIZE[0]
    target = (pz, py, px)

    mri_r = resize_volume(mri, target).astype(np.float32)
    ct_r  = resize_volume(ct,  target).astype(np.float32)

    saved = 0

    # Base (no augmentation)
    stem = os.path.join(OUTPUT_DIR, f"{subject_id}_patch_0")
    np.save(stem + "_mri.npy", mri_r)
    np.save(stem + "_ct.npy",  ct_r)
    saved += 1

    # Light augmentation: random axis flips (geometric only — consistent for both)
    for i in range(1, n_augmented):
        mri_aug = mri_r.copy()
        ct_aug  = ct_r.copy()
        for axis in range(3):
            if np.random.rand() > 0.5:
                mri_aug = np.flip(mri_aug, axis=axis).copy()
                ct_aug  = np.flip(ct_aug,  axis=axis).copy()
        stem = os.path.join(OUTPUT_DIR, f"{subject_id}_patch_{i}")
        np.save(stem + "_mri.npy", mri_aug)
        np.save(stem + "_ct.npy",  ct_aug)
        saved += 1

    return saved


# ===========================================================
# DATASET 1 — coregistration_and_segmentation (selected IDs)
# ===========================================================
def process_dataset1() -> int:
    print("\n" + "=" * 60)
    print("DATASET 1 — coregistration_and_segmentation")
    print(f"  Source : {COREGISTRATION_DIR}")
    print(f"  Subjects: {len(SELECTED_IDS)}  |  Patches/subject: {PATCHES_PER_SUBJECT_DS1}")
    print("=" * 60)

    total = 0
    for subject_id in SELECTED_IDS:
        subject_dir = os.path.join(COREGISTRATION_DIR, subject_id)

        if not os.path.isdir(subject_dir):
            print(f"  [WARN] Directory not found: {subject_dir} — skipping.")
            continue

        # Locate the three required files
        try:
            mri_path  = glob.glob(os.path.join(subject_dir, "*mri_reference.nii.gz"))[0]
            ct_path   = glob.glob(os.path.join(subject_dir, "*ct_registered.nii.gz"))[0]
            mask_path = glob.glob(os.path.join(subject_dir, "*segmentation.seg.nrrd"))[0]
        except IndexError:
            print(f"  [WARN] Missing file(s) in {subject_dir} — skipping.")
            continue

        print(f"\n  → {subject_id}")
        try:
            mri_arr  = normalise_mri(sitk.ReadImage(mri_path))
            ct_arr   = normalise_ct(sitk.ReadImage(ct_path))
            mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))

            if mri_arr.shape != ct_arr.shape:
                print(f"    [WARN] Shape mismatch — MRI {mri_arr.shape} vs CT {ct_arr.shape} — skipping.")
                continue

            n = extract_patches_from_mask(mri_arr, ct_arr, mask_arr, subject_id, PATCHES_PER_SUBJECT_DS1)
            print(f"    Saved: {n} patches")
            total += n

        except Exception as exc:
            print(f"    [ERROR] {exc}")

    print(f"\n  Dataset 1 total: {total} patches")
    return total


# ===========================================================
# DATASET 2 — vertebrae (already cropped, random patches)
# ===========================================================
def process_dataset2() -> int:
    print("\n" + "=" * 60)
    print("DATASET 2 — Dataset vertebres format ok")
    print(f"  Source : {VERTEBRAE_DIR}")
    print(f"  Patches/subject: {PATCHES_PER_SUBJECT_DS2}")
    print("=" * 60)

    # --- diagnostics ---
    print(f"  Checking directory: {VERTEBRAE_DIR}")
    print(f"  Directory exists  : {os.path.isdir(VERTEBRAE_DIR)}")
    if os.path.isdir(VERTEBRAE_DIR):
        all_files = os.listdir(VERTEBRAE_DIR)
        print(f"  Files found ({len(all_files)}): {all_files}")
    # --------------------

    ct_files = sorted(glob.glob(os.path.join(VERTEBRAE_DIR, "CT_*.nii")))
    if not ct_files:
        # Also try .nii.gz in case extension differs
        ct_files = sorted(glob.glob(os.path.join(VERTEBRAE_DIR, "CT_*.nii.gz")))
        if ct_files:
            print("  [INFO] Found .nii.gz files — updating glob pattern.")
    if not ct_files:
        print("  [ERROR] No CT files found in Dataset 2.")
        return 0

    total = 0
    for ct_path in ct_files:
        n_str      = os.path.basename(ct_path).replace("CT_", "").replace(".nii", "")
        mri_path   = os.path.join(VERTEBRAE_DIR, f"MRI_{n_str}.nii")
        subject_id = f"vertebrae_sub{n_str.zfill(2)}"

        if not os.path.exists(mri_path):
            print(f"  [WARN] MRI_{n_str}.nii not found — skipping.")
            continue

        print(f"\n  → {subject_id}  (CT_{n_str}.nii + MRI_{n_str}.nii)")
        try:
            ct_itk  = sitk.ReadImage(ct_path)
            mri_itk = sitk.ReadImage(mri_path)

            mri_raw = sitk.GetArrayFromImage(mri_itk)
            ct_raw  = sitk.GetArrayFromImage(ct_itk)
            print(f"    MRI native shape: {mri_raw.shape}  CT native shape: {ct_raw.shape}")

            # Normalise each modality from its raw values.
            # Each will be independently resized to PATCH_SIZE inside extract_whole_volume().
            # We do NOT resample MRI into CT space — that would leave large empty regions
            # because MRI has a smaller FOV than CT in this dataset.
            mri_arr = normalise_mri(mri_itk)
            ct_arr  = normalise_ct_uint8(ct_itk)   # Dataset 2 CTs are uint8 (0-255)

            n = extract_whole_volume(mri_arr, ct_arr, subject_id, PATCHES_PER_SUBJECT_DS2)
            print(f"    Saved: {n} patches")
            total += n

        except Exception as exc:
            print(f"    [ERROR] {exc}")

    print(f"\n  Dataset 2 total: {total} patches")
    return total


# ===========================================================
# MAIN
# ===========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PATCH GENERATION")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  Patch size : {PATCH_SIZE}  (X, Y, Z)")
    print("=" * 60)

    n1 = 0  # process_dataset1()
    n2 = process_dataset2()

    on_disk = len(glob.glob(os.path.join(OUTPUT_DIR, "*_mri.npy")))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Dataset 1 (segmentation-centred) : {n1:>4} patches")
    print(f"  Dataset 2 (vertebrae, random)    : {n2:>4} patches")
    print(f"  Total patches on disk            : {on_disk:>4}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 60)