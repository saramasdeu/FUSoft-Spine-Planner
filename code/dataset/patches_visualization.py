"""
patches_visualization.py
------------------------
Interactive viewer for MRI/CT patch pairs.

Controls:
  - Slice slider   : scroll through Z slices
  - Mouse scroll   : same as slider
  - Next / Prev    : navigate between patches
  - Filter buttons : show All / Dataset 1 / Dataset 2 patches
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ===========================================================
# CONFIGURATION — update this path after running creation_patches.py
# ===========================================================
PATCH_DIR = "/Users/saramasdeusans/Desktop/DATASET_NET/patches"


# ===========================================================
# VIEWER
# ===========================================================
class PatchViewer:
    def __init__(self, patch_bases: list[str]):
        self.all_patches = patch_bases
        self.filtered    = patch_bases   # active list (may be filtered)
        self.idx         = 0

        self._load_patch()
        self._build_ui()
        self._draw()

    # -------------------------------------------------------
    def _load_patch(self):
        base = self.filtered[self.idx]
        self.current_name = base
        self.mri = np.load(os.path.join(PATCH_DIR, base + "_mri.npy"))
        self.ct  = np.load(os.path.join(PATCH_DIR, base + "_ct.npy"))
        self.n_slices = self.mri.shape[0]

    # -------------------------------------------------------
    def _build_ui(self):
        self.fig = plt.figure(figsize=(14, 7))
        self.fig.patch.set_facecolor("#1e1e1e")

        # --- image axes ---
        self.ax_mri = self.fig.add_axes([0.05, 0.25, 0.40, 0.65])
        self.ax_ct  = self.fig.add_axes([0.55, 0.25, 0.40, 0.65])
        for ax in (self.ax_mri, self.ax_ct):
            ax.axis("off")
            ax.set_facecolor("#1e1e1e")

        self.im_mri = self.ax_mri.imshow(
            np.zeros((128, 128)), cmap="gray", vmin=-1.5, vmax=1.5)
        self.im_ct  = self.ax_ct.imshow(
            np.zeros((128, 128)), cmap="gray", vmin=-1.0, vmax=1.0)

        self.ax_mri.set_title("MRI", color="white", fontsize=12)
        self.ax_ct.set_title("CT  (target)", color="white", fontsize=12)

        # --- slice slider ---
        ax_sl = self.fig.add_axes([0.25, 0.14, 0.50, 0.025])
        self.slider = Slider(
            ax_sl, "Slice Z", 0, self.n_slices - 1,
            valinit=self.n_slices // 2, valstep=1, color="#4a90d9"
        )
        self.slider.label.set_color("white")
        self.slider.valtext.set_color("white")
        self.slider.on_changed(self._on_slider)

        # --- Prev / Next buttons ---
        ax_prev = self.fig.add_axes([0.05, 0.04, 0.12, 0.06])
        ax_next = self.fig.add_axes([0.83, 0.04, 0.12, 0.06])
        self.btn_prev = Button(ax_prev, "◀  Prev", color="#333", hovercolor="#555")
        self.btn_next = Button(ax_next, "Next  ▶", color="#333", hovercolor="#555")
        self.btn_prev.label.set_color("white")
        self.btn_next.label.set_color("white")
        self.btn_prev.on_clicked(lambda _: self._navigate(-1))
        self.btn_next.on_clicked(lambda _: self._navigate(+1))

        # --- Filter buttons ---
        ax_all  = self.fig.add_axes([0.25, 0.04, 0.12, 0.06])
        ax_ds1  = self.fig.add_axes([0.39, 0.04, 0.12, 0.06])
        ax_ds2  = self.fig.add_axes([0.53, 0.04, 0.12, 0.06])
        self.btn_all  = Button(ax_all,  "All",        color="#4a4a4a", hovercolor="#666")
        self.btn_ds1  = Button(ax_ds1,  "Dataset 1",  color="#4a4a4a", hovercolor="#666")
        self.btn_ds2  = Button(ax_ds2,  "Dataset 2",  color="#4a4a4a", hovercolor="#666")
        for b in (self.btn_all, self.btn_ds1, self.btn_ds2):
            b.label.set_color("white")
        self.btn_all.on_clicked(lambda _: self._filter("all"))
        self.btn_ds1.on_clicked(lambda _: self._filter("sub"))
        self.btn_ds2.on_clicked(lambda _: self._filter("vertebrae"))

        # --- title text ---
        self.title = self.fig.text(
            0.5, 0.93, "", ha="center", va="center",
            color="white", fontsize=11
        )
        self.info = self.fig.text(
            0.5, 0.89, "", ha="center", va="center",
            color="#aaaaaa", fontsize=9
        )

        # --- mouse scroll ---
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

    # -------------------------------------------------------
    def _draw(self):
        z = int(self.slider.val)
        self.im_mri.set_data(np.flipud(self.mri[z]))
        self.im_ct.set_data(np.flipud(self.ct[z]))

        source = "Dataset 2 — vertebrae" if self.current_name.startswith("vertebrae") \
                 else "Dataset 1 — coregistration"
        self.title.set_text(
            f"Patch {self.idx + 1} / {len(self.filtered)}  —  {self.current_name}"
        )
        self.info.set_text(
            f"Source: {source}   |   "
            f"Shape: {self.mri.shape}   |   "
            f"MRI [{self.mri.min():.2f}, {self.mri.max():.2f}]   "
            f"CT [{self.ct.min():.2f}, {self.ct.max():.2f}]"
        )
        self.fig.canvas.draw_idle()

    # -------------------------------------------------------
    def _on_slider(self, _):
        self._draw()

    def _on_scroll(self, event):
        if event.button == "up":
            new = min(self.slider.val + 1, self.n_slices - 1)
        elif event.button == "down":
            new = max(self.slider.val - 1, 0)
        else:
            return
        self.slider.set_val(new)

    def _navigate(self, direction: int):
        self.idx = (self.idx + direction) % len(self.filtered)
        self._load_patch()
        # Reset slider range for new patch
        self.slider.valmax = self.n_slices - 1
        self.slider.set_val(self.n_slices // 2)
        self._draw()

    def _filter(self, mode: str):
        if mode == "all":
            self.filtered = self.all_patches
        elif mode == "sub":
            self.filtered = [p for p in self.all_patches if p.startswith("sub")]
        elif mode == "vertebrae":
            self.filtered = [p for p in self.all_patches if p.startswith("vertebrae")]

        if not self.filtered:
            print(f"[WARN] No patches match filter '{mode}'.")
            self.filtered = self.all_patches

        self.idx = 0
        self._load_patch()
        self.slider.valmax = self.n_slices - 1
        self.slider.set_val(self.n_slices // 2)
        self._draw()


# ===========================================================
# MAIN
# ===========================================================
if __name__ == "__main__":
    if not os.path.isdir(PATCH_DIR):
        print(f"[ERROR] Patch directory not found: {PATCH_DIR}")
        sys.exit(1)

    mri_files = sorted(glob.glob(os.path.join(PATCH_DIR, "*_mri.npy")))
    if not mri_files:
        print(f"[ERROR] No .npy patch files found in {PATCH_DIR}")
        sys.exit(1)

    patch_bases = [os.path.basename(f).replace("_mri.npy", "") for f in mri_files]
    print(f"Found {len(patch_bases)} patches  ({PATCH_DIR})")
    ds1 = sum(1 for p in patch_bases if p.startswith("sub"))
    ds2 = sum(1 for p in patch_bases if p.startswith("vertebrae"))
    print(f"  Dataset 1 (sub*)       : {ds1}")
    print(f"  Dataset 2 (vertebrae*) : {ds2}")

    plt.close("all")
    viewer = PatchViewer(patch_bases)
    plt.show()