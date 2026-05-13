import sys
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import map_coordinates
from skimage import measure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QTabWidget, QSlider, QGroupBox,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QProgressDialog,
    QSplitter,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import pyvista as pv
from pyvistaqt import QtInteractor
from skimage.measure import marching_cubes

BONE_HU = 300

# Heuristic weights
W_DIST    = 2.0   # distance
W_THICK   = 1.1   # bone thickness
W_ANGLE   = 1.9   # incidence angle
W_DENSITY = 1.1   # density
W_CHANGES = 2.5   # density changes

N_STEPS = 300

_PANEL_STYLE = """
    QGroupBox {
        color: #a0c4ff; border: 1px solid #2d4a7a; border-radius: 6px;
        margin-top: 10px; font-weight: bold; font-size: 11px;
    }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
    QLabel { color: #c8d8f0; font-size: 11px; }
    QPushButton {
        background: #16213e; color: #a0c4ff; border: 1px solid #2d4a7a;
        border-radius: 4px; padding: 7px; font-size: 11px;
    }
    QPushButton:hover { background: #0f3460; border-color: #5b9bd5; }
    QSlider::groove:horizontal { height: 4px; background: #2d4a7a; border-radius: 2px; }
    QSlider::handle:horizontal {
        background: #5b9bd5; width: 14px; height: 14px;
        margin: -5px 0; border-radius: 7px;
    }
"""

class MeshWorker(QThread):
    # Emet: bone_verts, bone_faces, skin_verts, skin_faces
    finished = pyqtSignal(object, object, object, object)

    def __init__(self, ct_array, spacing):
        super().__init__()
        self.ct_array = ct_array
        self.spacing  = spacing

    def _mc(self, threshold, step):
        sp = (self.spacing[2], self.spacing[1], self.spacing[0])
        return marching_cubes(self.ct_array, level=threshold,
                              spacing=sp, step_size=step,
                              allow_degenerate=False)

    def run(self):
        try:
            bv, bf, _, _ = self._mc(threshold=200,  step=2) 
            sv, sf, _, _ = self._mc(threshold=-100, step=3)  
            self.finished.emit(bv, bf, sv, sf)
        except Exception as e:
            print(f"[MeshWorker] Error: {e}")
            self.finished.emit(None, None, None, None)


# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FUSpine - Precision Planner")
        self.resize(1600, 950)

        self.mri_array   = None
        self.ct_array    = None
        self.spacing     = (1.0, 1.0, 1.0)
        self.target      = None
        self._cz         = 0
        self.top10       = []
        self.all_results = []
        self._bone_mesh  = None  
        self._mesh_worker = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        lay = QHBoxLayout(central)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)
        self._build_tab1()
        self._build_tab2()
        self._build_tab3()
        self._build_tab4()

    # TAB 1
    def _build_tab1(self):
        tab = QWidget()
        tab.setStyleSheet("background-color: #1a1a2e;")
        lay = QHBoxLayout(tab)
        lay.setContentsMargins(8, 8, 8, 8)

        panel = QVBoxLayout()
        panel.setSpacing(10)
        panel.setContentsMargins(0, 0, 6, 0)

        g1 = QGroupBox("Upload Data")
        g1.setStyleSheet(_PANEL_STYLE)
        gl = QVBoxLayout(g1)
        gl.setSpacing(6)
        b_mri = QPushButton("Upload MRI")
        b_mri.clicked.connect(self._load_mri)
        b_ct  = QPushButton("Upload CT")
        b_ct.clicked.connect(self._load_ct)
        gl.addWidget(b_mri); gl.addWidget(b_ct)
        panel.addWidget(g1)

        g2 = QGroupBox("Axial Navigation")
        g2.setStyleSheet(_PANEL_STYLE)
        g2l = QVBoxLayout(g2)
        g2l.setSpacing(4)
        self.lbl_slice = QLabel("Slice Z:  —")
        self.sld_z = QSlider(Qt.Orientation.Horizontal)
        self.sld_z.setMinimum(0); self.sld_z.setMaximum(100)
        self.sld_z.valueChanged.connect(self._on_z)
        g2l.addWidget(self.lbl_slice); g2l.addWidget(self.sld_z)
        panel.addWidget(g2)

        g3 = QGroupBox("Target selected")
        g3.setStyleSheet(_PANEL_STYLE)
        g3l = QVBoxLayout(g3)
        self.lbl_target = QLabel("Click on the Axial MRI view to mark the target.")
        self.lbl_target.setWordWrap(True)
        g3l.addWidget(self.lbl_target)
        panel.addWidget(g3)

        self.btn_run = QPushButton("CALCULATE 3D TRAJECTORIES")
        self.btn_run.setStyleSheet(
            "background:#1a6b3c; color:white; font-weight:bold; padding:12px;"
            "border-radius:5px; font-size:12px; border:none;")
        self.btn_run.clicked.connect(self._confirm)
        panel.addWidget(self.btn_run)
        panel.addStretch()
        lay.addLayout(panel, 1)

        self.fig1 = Figure(facecolor='#0d0d1a')
        self.canvas1 = FigureCanvas(self.fig1)
        self.canvas1.setStyleSheet("background-color: #0d0d1a;")
        ax_kw = dict(facecolor='#111827')
        self.ax1_mri_ax  = self.fig1.add_subplot(2, 2, 1, **ax_kw)
        self.ax1_mri_sag = self.fig1.add_subplot(2, 2, 2, **ax_kw)
        self.ax1_mri_cor = self.fig1.add_subplot(2, 2, 3, **ax_kw)
        self.ax1_ct_ax   = self.fig1.add_subplot(2, 2, 4, **ax_kw)
        self.fig1.tight_layout(pad=1.5)
        self.canvas1.mpl_connect('button_press_event', self._on_click)
        lay.addWidget(self.canvas1, 4)
        self.tabs.addTab(tab, "1. Target Selection")

    #TAB 2
    def _build_tab2(self):
        tab = QWidget()
        lay = QHBoxLayout(tab)
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["#", "Dist (mm)", "Bone (mm)", "Incidence°", "Density", "Score"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(self._on_table_click)
        lay.addWidget(self.table, 1)
        self.fig2 = Figure(figsize=(10, 7))
        self.canvas2 = FigureCanvas(self.fig2)
        self.ax2_axial   = self.fig2.add_subplot(1, 2, 1)
        self.ax2_sagital = self.fig2.add_subplot(1, 2, 2)
        for ax in (self.ax2_axial, self.ax2_sagital):
            ax.set_facecolor('#111'); ax.axis('off')
        lay.addWidget(self.canvas2, 2)
        self.tabs.addTab(tab, "2. Top 10 Results")

    # ── TAB 3 ────────────────────────────────────────────────────────────────
    def _build_tab3(self):
        tab = QWidget()
        main_lay = QVBoxLayout(tab)
        main_lay.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_w = QWidget()
        left_lay = QVBoxLayout(left_w)
        left_lay.setContentsMargins(0, 0, 2, 0)

        self.fig3 = Figure(figsize=(7, 7), facecolor='#0d0d1a')
        self.canvas3 = FigureCanvas(self.fig3)
        gs = self.fig3.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
        self.ax3_axial   = self.fig3.add_subplot(gs[:, 0])
        self.ax3_sagital = self.fig3.add_subplot(gs[0, 1])
        self.ax3_pr      = self.fig3.add_subplot(gs[1, 1])
        for ax in (self.ax3_axial, self.ax3_sagital):
            ax.set_facecolor('#111'); ax.axis('off')
        self.ax3_pr.set_facecolor('#111')
        self.ax3_pr.tick_params(colors='#aaa', labelsize=7)

        self.fig3b = Figure(figsize=(7, 2), facecolor='#0d0d1a')
        self.canvas3b = FigureCanvas(self.fig3b)
        self.ax3_gr = self.fig3b.add_subplot(1, 1, 1)
        self.ax3_gr.set_facecolor('#111')
        self.ax3_gr.tick_params(colors='#aaa', labelsize=7)

        left_lay.addWidget(self.canvas3, 4)
        left_lay.addWidget(self.canvas3b, 1)
        splitter.addWidget(left_w)

        right_w = QWidget()
        right_lay = QVBoxLayout(right_w)
        right_lay.setContentsMargins(2, 0, 0, 0)

        lbl3d = QLabel("  3D View — Bone + Trajectory (rotate: left click · zoom: wheel)")
        lbl3d.setStyleSheet("color:#7eb8f7; font-weight:bold; font-size:11px;"
                            "background:#0d0d1a; padding:5px;")
        right_lay.addWidget(lbl3d)

        self.pv_plotter = QtInteractor(right_w, auto_update=False)
        self.pv_plotter.set_background('#0d0d1a')
        right_lay.addWidget(self.pv_plotter.interactor)

        splitter.addWidget(right_w)
        splitter.setSizes([520, 780])
        main_lay.addWidget(splitter)
        self.tabs.addTab(tab, "3. Verification and 3D Profile")

    # ── TAB 4 ────────────────────────────────────────────────────────────────
    def _build_tab4(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(10, 10, 10, 10)

        lbl = QLabel("All calculated trajectories — sorted by score (best first)")
        lbl.setStyleSheet("color:#7eb8f7; font-weight:bold; font-size:12px; padding:6px;")
        lay.addWidget(lbl)

        self.full_table = QTableWidget()
        self.full_table.setColumnCount(7)
        self.full_table.setHorizontalHeaderLabels(
            ["#", "Dist (mm)", "Bone (mm)", "Incidence°", "Density", "ΔDensitt", "Score"])
        self.full_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.full_table.setAlternatingRowColors(True)
        self.full_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.full_table.cellClicked.connect(self._on_full_table_click)
        lay.addWidget(self.full_table)

        self.tabs.addTab(tab, "4. All Trajectories")

    # Upload
    def _load_mri(self):
        path, _ = QFileDialog.getOpenFileName(self, "MRI", "", "*.nii *.nii.gz")
        if path:
            img = sitk.ReadImage(path)
            self.mri_array = sitk.GetArrayFromImage(img)
            self.spacing = img.GetSpacing()
            self.sld_z.setMaximum(self.mri_array.shape[0] - 1)
            self.sld_z.setValue(self.mri_array.shape[0] // 2)
            self._redraw1()

    def _load_ct(self):
        path, _ = QFileDialog.getOpenFileName(self, "CT", "", "*.nii *.nii.gz")
        if path:
            img = sitk.ReadImage(path)
            mask    = sitk.BinaryThreshold(img, lowerThreshold=-200, upperThreshold=3000)
            cleaned = sitk.RelabelComponent(sitk.ConnectedComponent(mask))
            head    = sitk.BinaryThreshold(cleaned, lowerThreshold=1, upperThreshold=1)
            img     = sitk.Mask(img, head, outsideValue=-1000)
            self.ct_array = sitk.GetArrayFromImage(img)
            self._bone_mesh = None 
            self._redraw1()

    # TAB 1 Navegation
    def _on_z(self, v):
        self._cz = v
        self.lbl_slice.setText(f"Slice Z:  {v}")
        self._redraw1()

    def _on_click(self, event):
        if event.inaxes != self.ax1_mri_ax or self.mri_array is None:
            return
        nz, ny, nx = self.mri_array.shape
        tx = int(np.clip(event.xdata, 0, nx - 1))
        ty = int(np.clip(event.ydata, 0, ny - 1))
        self.target = (tx, ty, self._cz)
        self.lbl_target.setText(f"X = {tx}\nY = {ty}\nZ = {self._cz}")
        self._redraw1()

    def _redraw1(self):
        for ax in (self.ax1_mri_ax, self.ax1_mri_sag,
                   self.ax1_mri_cor, self.ax1_ct_ax):
            ax.clear(); ax.set_facecolor('#111827'); ax.axis('off')

        s = self._cz
        if self.target:
            tx, ty, tz = self.target
        else:
            arr = self.mri_array if self.mri_array is not None else self.ct_array
            if arr is None:
                self.canvas1.draw(); return
            tz, ty, tx = arr.shape[0]//2, arr.shape[1]//2, arr.shape[2]//2

        title_kw = dict(color='#7eb8f7', fontsize=9, pad=4)
        ch_kw    = dict(color='#ff4d4d', linewidth=0.8, alpha=0.7)

        if self.mri_array is not None:
            nz = self.mri_array.shape[0]
            self.ax1_mri_ax.imshow(self.mri_array[s], cmap='gray')
            self.ax1_mri_ax.set_title(f"MRI · Axial  z={s}", **title_kw)
            self.ax1_mri_sag.imshow(np.flipud(self.mri_array[:, :, tx]), cmap='gray', aspect='auto')
            self.ax1_mri_sag.set_title(f"MRI · Sagital  x={tx}", **title_kw)
            self.ax1_mri_cor.imshow(np.flipud(self.mri_array[:, ty, :]), cmap='gray', aspect='auto')
            self.ax1_mri_cor.set_title(f"MRI · Coronal  y={ty}", **title_kw)
            if self.target:
                tz_flip = nz - 1 - tz
                self.ax1_mri_ax.axhline(ty, **ch_kw); self.ax1_mri_ax.axvline(tx, **ch_kw)
                self.ax1_mri_ax.plot(tx, ty, 'o', color='#ff4d4d', ms=7,
                                     markerfacecolor='none', markeredgewidth=1.5)
                self.ax1_mri_sag.axhline(tz_flip, **ch_kw); self.ax1_mri_sag.axvline(ty, **ch_kw)
                self.ax1_mri_sag.plot(ty, tz_flip, 'o', color='#ff4d4d', ms=7,
                                      markerfacecolor='none', markeredgewidth=1.5)
                self.ax1_mri_cor.axhline(tz_flip, **ch_kw); self.ax1_mri_cor.axvline(tx, **ch_kw)
                self.ax1_mri_cor.plot(tx, tz_flip, 'o', color='#ff4d4d', ms=7,
                                      markerfacecolor='none', markeredgewidth=1.5)

        if self.ct_array is not None:
            self.ax1_ct_ax.imshow(self.ct_array[s], cmap='gray', vmin=-1000, vmax=1500)
            self.ax1_ct_ax.set_title(f"CT · Axial  z={s}", **title_kw)
            if self.target:
                self.ax1_ct_ax.axhline(ty, **ch_kw); self.ax1_ct_ax.axvline(tx, **ch_kw)
                self.ax1_ct_ax.plot(tx, ty, 'o', color='#ff4d4d', ms=7,
                                    markerfacecolor='none', markeredgewidth=1.5)

        self.fig1.tight_layout(pad=1.5)
        self.canvas1.draw()

    # Skin detection
    def _on_skin(self, x, y, z):
        xi = int(round(x))
        yi = int(round(y))
        zi = int(round(z))
        nz, ny, nx = self.ct_array.shape
        if not (0 <= xi < nx and 0 <= yi < ny and 0 <= zi < nz):
            return False
        if self.ct_array[zi, yi, xi] < -300:
            return False 

        def _has_air_neighbor_2d(slice_2d, row, col):
            h, w = slice_2d.shape
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if slice_2d[nr, nc] < -300:
                        return True
            return False

        axial_ok   = _has_air_neighbor_2d(self.ct_array[zi],         yi, xi)
        sagital_ok = _has_air_neighbor_2d(self.ct_array[:, :, xi],   zi, yi)
        coronal_ok = _has_air_neighbor_2d(self.ct_array[:, yi, :],   zi, xi)

        return axial_ok and sagital_ok and coronal_ok

    # CONFIRMATION + ENGINE
    def _confirm(self):
        if self.target and self.ct_array is not None:
            self._run()
            self.tabs.setCurrentIndex(1)

    def _run(self):
        xt, yt, zt = self.target
        nz, ny, nx = self.ct_array.shape
        sx, sy, sz = self.spacing
        results = []

        MAX_BONE_MM = 10.0   #>10mm of bone → always discarded

        print(f"[NCE] Target: {self.target}, spacing: {self.spacing}")

        body_mask          = (self.ct_array > -300).astype(np.float32)
        gz_vol, gy_vol, gx_vol = np.gradient(body_mask)   # [dA/dz, dA/dy, dA/dx]

        # Instead of 2D slices, we shoot rays in all directions of space.
        #phi = elevation (0=up, π=down), theta = azimuth (0…2π)
        n_phi   = 60
        n_theta = 120
        phis   = np.linspace(0.05 * np.pi, 0.95 * np.pi, n_phi)
        thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

        for phi in phis:
            for theta in thetas:
                # Unit direction vector in mm (outward from the target)
                dx_mm = np.sin(phi) * np.cos(theta)
                dy_mm = np.sin(phi) * np.sin(theta)
                dz_mm = np.cos(phi)

                # Vectorized beam: steps from 1mm to 200mm
                steps  = np.arange(5.0, 200.0, 1.0)
                xi_arr = np.round(xt + dx_mm * steps / sx).astype(int)
                yi_arr = np.round(yt + dy_mm * steps / sy).astype(int)
                zi_arr = np.round(zt + dz_mm * steps / sz).astype(int)

                valid  = ((xi_arr >= 0) & (xi_arr < nx) &
                          (yi_arr >= 0) & (yi_arr < ny) &
                          (zi_arr >= 0) & (zi_arr < nz))
                xi_v, yi_v, zi_v = xi_arr[valid], yi_arr[valid], zi_arr[valid]
                if len(xi_v) == 0: continue

                # First air voxel = skin surface
                air_idx = np.where(self.ct_array[zi_v, yi_v, xi_v] < -300)[0]
                if len(air_idx) == 0: continue
                si      = air_idx[0]
                ex, ey, ez = float(xi_v[si]), float(yi_v[si]), float(zi_v[si])

                dist_mm = float(np.sqrt(((xt - ex) * sx) ** 2 +
                                        ((yt - ey) * sy) ** 2 +
                                        ((zt - ez) * sz) ** 2))
                if dist_mm < 20: continue

                #HU profile of the 3D skin → target beam 
                t  = np.linspace(0, 1, N_STEPS)
                xr = ex + (xt - ex) * t
                yr = ey + (yt - ey) * t
                zr = ez + (zt - ez) * t

                valid3d = ((xr >= 0) & (xr < nx) &
                           (yr >= 0) & (yr < ny) &
                           (zr >= 0) & (zr < nz))
                if not np.any(valid3d): continue

                profile = map_coordinates(
                    self.ct_array, [zr[valid3d], yr[valid3d], xr[valid3d]],
                    order=1, cval=-1000).astype(float)

                step_mm         = dist_mm / len(profile)
                bone            = float(np.sum(profile > BONE_HU)) * step_mm
                mean_density    = float(np.mean(profile))
                density_changes = int(np.sum(np.abs(np.diff(profile)) > 50))

                # Absolute bone threshold (always, regardless of the rest)
                if bone > MAX_BONE_MM: continue

                # angle = 0° → perpendicular ray (ideal for FUS, no reflection)
                # angle = 90° → grazing ray (total reflection, unacceptable)
                xi_c = int(np.clip(round(ex), 0, nx - 1))
                yi_c = int(np.clip(round(ey), 0, ny - 1))
                zi_c = int(np.clip(round(ez), 0, nz - 1))

                # Normal in mm space from precomputed gradient
                normal_vox = np.array([
                    gx_vol[zi_c, yi_c, xi_c] / sx,
                    gy_vol[zi_c, yi_c, xi_c] / sy,
                    gz_vol[zi_c, yi_c, xi_c] / sz,
                ])
                n_norm = float(np.linalg.norm(normal_vox))

                if n_norm < 1e-6:
                    incidence_angle = 45.0 
                else:
                    normal_unit = normal_vox / n_norm
                    traj_mm     = np.array([(xt - ex) * sx,
                                            (yt - ey) * sy,
                                            (zt - ez) * sz])
                    traj_unit   = traj_mm / np.linalg.norm(traj_mm)
                    cos_a       = abs(float(np.dot(traj_unit, normal_unit)))
                    incidence_angle = float(np.degrees(np.arccos(np.clip(cos_a, 0, 1))))

                if ey > yt: 
                    continue

                results.append({
                    'skin_vox':        (ex, ey, ez),
                    'dist':            dist_mm,
                    'bone':            bone,
                    'incidence':      incidence_angle,
                    'profile':         profile,
                    'mean_density':    mean_density,
                    'density_changes': density_changes,
                    'score':           0,
                })

        print(f"[NCE] Candidate Trajectories: {len(results)}")
        if not results: return

        DIST_REF    = 150.0   # mm – maximum reasonable distance
        BONE_REF    = MAX_BONE_MM   # mm – limit already applied, scale 0–10mm
        ANGLE_REF   = 90.0    # degrees – 0°=ideal, 90°=flat
        CHANGES_REF = 100.0   #number of abrupt changes of reference

        distances           = np.array([r['dist']            for r in results])
        thicknesses         = np.array([r['bone']            for r in results])
        angles_arr          = np.array([r['incidence']      for r in results])
        mean_densities      = np.array([r['mean_density']    for r in results])
        density_changes_arr = np.array([r['density_changes'] for r in results])

        def abs_norm(arr, ref):
            """0 = best possible, 10 = at the reference limit (Keysha)."""
            return np.clip(arr / ref, 0, 1) * 10

        def rel_norm(arr):
            """Relative normalization (for metrics without clear clinical ref.)"""
            rng = np.max(arr) - np.min(arr)
            return ((arr - np.min(arr)) / rng * 10) if rng > 0 else np.zeros_like(arr)

        dist_norm    = abs_norm(distances,           DIST_REF)
        thick_norm   = abs_norm(thicknesses,         BONE_REF)
        angle_norm   = abs_norm(angles_arr,          ANGLE_REF)
        density_norm = rel_norm(mean_densities)      
        changes_norm = abs_norm(density_changes_arr, CHANGES_REF)

        sum_weights = W_DIST + W_THICK + W_ANGLE + W_DENSITY + W_CHANGES

        print(f"[NCE] Dist {distances.min():.0f}–{distances.max():.0f}mm | "
              f"Bone {thicknesses.min():.1f}–{thicknesses.max():.1f}mm | "
              f"Angle {angles_arr.min():.0f}–{angles_arr.max():.0f}°")

        for i, r in enumerate(results):
            r['score'] = (
                W_DIST    * dist_norm[i]    +
                W_THICK   * thick_norm[i]   +
                W_ANGLE   * angle_norm[i]   +
                W_DENSITY * density_norm[i] +
                W_CHANGES * changes_norm[i]
            ) / sum_weights

        results.sort(key=lambda x: x['score'])
        self.top10       = results[:10]
        self.all_results = results

        # ── Top 10 (tab 2) ────────────────────────────────────────────────
        self.table.setRowCount(len(self.top10))
        for i, r in enumerate(self.top10):
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QTableWidgetItem(f"{r['dist']:.1f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{r['bone']:.1f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{r['incidence']:.1f}°"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{r['mean_density']:.1f}"))
            self.table.setItem(i, 5, QTableWidgetItem(f"{r['score']:.3f}"))

        # ── Totes les trajectòries (tab 4) ───────────────────────────────
        self.full_table.setRowCount(len(results))
        for i, r in enumerate(results):
            self.full_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.full_table.setItem(i, 1, QTableWidgetItem(f"{r['dist']:.1f}"))
            self.full_table.setItem(i, 2, QTableWidgetItem(f"{r['bone']:.1f}"))
            self.full_table.setItem(i, 3, QTableWidgetItem(f"{r['incidence']:.1f}°"))
            self.full_table.setItem(i, 4, QTableWidgetItem(f"{r['mean_density']:.1f}"))
            self.full_table.setItem(i, 5, QTableWidgetItem(str(r['density_changes'])))
            self.full_table.setItem(i, 6, QTableWidgetItem(f"{r['score']:.4f}"))

        self._draw2_trajectories(highlight=0)
        self._draw3(self.top10[0])

    # TAB 2
    def _draw2_trajectories(self, highlight=0):
        xt, yt, zt = self.target
        self.ax2_axial.clear()
        self.ax2_axial.imshow(self.ct_array[zt], cmap='gray', vmin=-1000, vmax=1500)
        self.ax2_axial.set_title(f"Axial (z={zt}) — Top 10", color='white', fontsize=10)
        self.ax2_axial.axis('off')
        self.ax2_axial.plot(xt, yt, 'r+', ms=16, markeredgewidth=2, label='Target')
        self.ax2_sagital.clear()
        sag = self.ct_array[:, :, xt]
        self.ax2_sagital.imshow(sag, cmap='gray', vmin=-1000, vmax=1500, origin='lower')
        self.ax2_sagital.set_title(f"Sagital (x={xt}) — Top 10", color='white', fontsize=10)
        self.ax2_sagital.axis('off')
        self.ax2_sagital.plot(yt, zt, 'r+', ms=16, markeredgewidth=2, label='Target')
        for i, r in enumerate(self.top10):
            ex, ey, ez = r['skin_vox']
            color = '#2ecc71' if i == highlight else '#e74c3c'
            lw    = 2.5 if i == highlight else 1.0
            alpha = 1.0 if i == highlight else 0.5
            self.ax2_axial.plot([ex, xt], [ey, yt], color=color, lw=lw, alpha=alpha)
            self.ax2_axial.plot(ex, ey, 'o', color=color, ms=5, alpha=alpha)
            self.ax2_sagital.plot([ey, yt], [ez, zt], color=color, lw=lw, alpha=alpha)
            self.ax2_sagital.plot(ey, ez, 'o', color=color, ms=5, alpha=alpha)
        ex, ey, ez = self.top10[highlight]['skin_vox']
        self.ax2_axial.plot(ex, ey, 'go', ms=10, label='Better Entrance')
        self.ax2_axial.legend(loc='upper right', fontsize=7, facecolor='#222', labelcolor='white')
        self.ax2_sagital.plot(ey, ez, 'go', ms=10, label='Better Entrance')
        self.ax2_sagital.legend(loc='upper right', fontsize=7, facecolor='#222', labelcolor='white')
        self.fig2.tight_layout(pad=2.0)
        self.canvas2.draw_idle()

    def _on_table_click(self, r, c):
        self._draw2_trajectories(highlight=r)
        self._draw3(self.top10[r])
        self.tabs.setCurrentIndex(2)

    def _on_full_table_click(self, r, c):
        if r < len(self.all_results):
            self._draw3(self.all_results[r])
            self.tabs.setCurrentIndex(2)

    # TAB 3
    def _draw3(self, traj):
        #2d
        self.ax3_axial.clear(); self.ax3_sagital.clear()
        self.ax3_pr.clear();    self.ax3_gr.clear()

        xt, yt, zt = self.target
        ex, ey, ez = traj['skin_vox']
        profile = traj['profile']

        self.ax3_axial.imshow(self.ct_array[int(ez)], cmap='gray', vmin=-1000, vmax=1500)
        self.ax3_axial.plot([ex, xt], [ey, yt], color='yellow', lw=1.5, ls='--')
        self.ax3_axial.scatter(ex, ey, color='#2ecc71', s=100, edgecolors='white', label='Transductor')
        self.ax3_axial.scatter(xt, yt, color='red', marker='+', s=200, zorder=5, label='Target')
        self.ax3_axial.set_title(f"Axial Entrance (Z={int(ez)})", color='white')
        self.ax3_axial.axis('off')
        self.ax3_axial.legend(loc='upper right', fontsize=7, facecolor='#222', labelcolor='white')

        sag = self.ct_array[:, :, int(ex)]
        self.ax3_sagital.imshow(sag, cmap='gray', vmin=-1000, vmax=1500, origin='lower')
        self.ax3_sagital.plot([ey, yt], [ez, zt], color='yellow', lw=2.5)
        self.ax3_sagital.scatter(ey, ez, color='#2ecc71', s=100, edgecolors='white')
        self.ax3_sagital.scatter(yt, zt, color='red', marker='+', s=200, zorder=5)
        self.ax3_sagital.set_title("Sagital projection", color='white')
        self.ax3_sagital.axis('off')

        self.ax3_pr.plot(profile, color='#5dade2', lw=2)
        self.ax3_pr.fill_between(range(len(profile)), 0, profile,
                                  where=profile > 300, color='red', alpha=0.3, label='Os (>300HU)')
        self.ax3_pr.set_facecolor('#111')
        self.ax3_pr.set_title("HU Profile", color='white', fontsize=9)
        self.ax3_pr.tick_params(colors='#aaa', labelsize=7)
        self.ax3_pr.legend(fontsize=7, facecolor='#222', labelcolor='white')

        self.ax3_gr.plot(np.diff(profile), color='#e67e22', lw=1.5)
        self.ax3_gr.axhline(0, color='#555', lw=0.8, ls='--')
        self.ax3_gr.set_facecolor('#111')
        self.ax3_gr.set_title("ΔHU Gradient", color='white', fontsize=9)
        self.ax3_gr.tick_params(colors='#aaa', labelsize=7)

        self.canvas3.draw_idle()
        self.canvas3b.draw_idle()

        # 3D PyVista
        self._update_3d(traj)

        if self._bone_mesh is None and self.ct_array is not None:
            self._build_bone_mesh_async()

    # Bone mesh
    def _build_bone_mesh_async(self):
        if self._mesh_worker is not None and self._mesh_worker.isRunning():
            return

        self._progress = QProgressDialog(
            "Generating 3D mesh of bone…\n(may take a few seconds)", None, 0, 0, self)
        self._progress.setWindowTitle("FUSpine 3D")
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setCancelButton(None)
        self._progress.show()

        self._mesh_worker = MeshWorker(self.ct_array, self.spacing)
        self._mesh_worker.finished.connect(self._on_mesh_ready)
        self._mesh_worker.start()

    def _on_mesh_ready(self, bv, bf, sv, sf):
        self._progress.close()
        if bv is None:
            return

        def make_pv(verts, faces, smooth_iter=20):
            # skimage: (z_mm, y_mm, x_mm) → PyVista: (x, y, z)
            pts = np.column_stack([verts[:, 2], verts[:, 1], verts[:, 0]])
            n = len(faces)
            pv_faces = np.hstack([np.full((n, 1), 3), faces]).ravel()
            mesh = pv.PolyData(pts, pv_faces)
            return mesh.smooth(n_iter=smooth_iter, relaxation_factor=0.1)

        self._bone_mesh = make_pv(bv, bf, smooth_iter=20)
        skin_mesh       = make_pv(sv, sf, smooth_iter=15)

        # Bone
        self.pv_plotter.add_mesh(
            self._bone_mesh,
            color='#ecdfc8',
            opacity=0.85,
            smooth_shading=True,
            name='bone',
        )
        # Skin
        self.pv_plotter.add_mesh(
            skin_mesh,
            color='#a8c8e8',
            opacity=0.18,
            smooth_shading=True,
            name='skin',
        )
        self.pv_plotter.reset_camera()
        self.pv_plotter.render()

    # update 3d trajectory
    def _update_3d(self, traj):
        sx, sy, sz = self.spacing
        xt, yt, zt = self.target
        ex, ey, ez = traj['skin_vox']

        # Coordenates in mm (x, y, z)
        p_target = np.array([xt*sx, yt*sy, zt*sz])
        p_entry  = np.array([ex*sx, ey*sy, ez*sz])

        # Tube of trajectory
        line = pv.Line(p_entry, p_target, resolution=1)
        tube = line.tube(radius=1.5, n_sides=20)
        self.pv_plotter.add_mesh(tube, color='#2ecc71', opacity=0.95,
                                  smooth_shading=True, name='trajectory')

        # Transductor
        entry_sphere = pv.Sphere(radius=4.0, center=p_entry)
        self.pv_plotter.add_mesh(entry_sphere, color='#2ecc71', opacity=1.0,
                                  smooth_shading=True, name='entry')

        # Target
        tgt_sphere = pv.Sphere(radius=4.0, center=p_target)
        self.pv_plotter.add_mesh(tgt_sphere, color='#e74c3c', opacity=1.0,
                                  smooth_shading=True, name='target_sphere')

        # text
        label_pts  = pv.PolyData(np.array([p_entry, p_target]))
        self.pv_plotter.add_point_labels(
            label_pts,
            ['Transductor', 'Target'],
            font_size=12,
            text_color='white',
            bold=True,
            show_points=False,
            name='labels',
            always_visible=True,
        )

        self.pv_plotter.render()

    def closeEvent(self, event):
        self.pv_plotter.close()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())