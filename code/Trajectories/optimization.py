import sys
import numpy as np
import SimpleITK as sitk
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QLabel,
                             QTabWidget, QSlider, QGroupBox, QMessageBox, 
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt
from scipy.signal import find_peaks

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FUSpine - Advanced Multimodal Planner")
        self.resize(1600, 950)

        # State variables
        self.mri_image = None
        self.ct_image = None
        self.mri_array = None
        self.ct_array = None
        self.num_slices = 0
        self.current_slice = 0
        self.target_point = None
        self.top_trajectories = []
        self.all_results = []
        self.spacing = (1.0, 1.0, 1.0)

        # Heuristic weights
        self.W_DIST = 1.3
        self.W_THICK = 2.8 
        self.W_ANGLE = 1.5
        self.W_DENSITY = 1.1
        self.W_CHANGES = 1.5

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        self.tabs = QTabWidget()


        # SCREEN 1: Target Selection
        self.tab1 = QWidget()
        tab1_layout = QHBoxLayout(self.tab1)
        left_tab1 = QVBoxLayout()

        load_group = QGroupBox("1. Load & Preprocessing")
        load_layout = QVBoxLayout()
        self.btn_load_mri = QPushButton("Load MRI")
        self.btn_load_mri.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; border-radius: 4px; padding: 6px;")
        self.btn_load_mri.clicked.connect(self.load_mri)
        self.btn_load_ct = QPushButton("Load CT")
        self.btn_load_ct.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; border-radius: 4px; padding: 6px;")
        self.btn_load_ct.clicked.connect(self.load_ct)
        load_layout.addWidget(self.btn_load_mri)
        load_layout.addWidget(self.btn_load_ct)
        load_group.setLayout(load_layout)
        left_tab1.addWidget(load_group)

        slice_group = QGroupBox("2. Slice Viewer")
        slice_layout = QVBoxLayout()
        self.lbl_slice = QLabel("Slice: 0")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.valueChanged.connect(self.change_slice)
        slice_layout.addWidget(self.lbl_slice)
        slice_layout.addWidget(self.slider)
        slice_group.setLayout(slice_layout)
        left_tab1.addWidget(slice_group)

        action_group = QGroupBox("3. Confirm Target")
        action_layout = QVBoxLayout()
        self.lbl_target = QLabel("Selected Target: None")
        self.btn_confirm = QPushButton("Confirm & Calculate Phase 2")
        self.btn_confirm.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 8px; border-radius: 4px;")
        self.btn_confirm.clicked.connect(self.confirm_target)
        action_layout.addWidget(self.lbl_target)
        action_layout.addWidget(self.btn_confirm)
        action_group.setLayout(action_layout)
        left_tab1.addWidget(action_group)

        left_tab1.addStretch()
        tab1_layout.addLayout(left_tab1, stretch=1)

        right_tab1 = QVBoxLayout()
        self.fig_tab1 = Figure(figsize=(12, 8))
        self.canvas_tab1 = FigureCanvas(self.fig_tab1)
        self.ax_mri = self.fig_tab1.add_subplot(1, 2, 1)
        self.ax_ct = self.fig_tab1.add_subplot(1, 2, 2)
        right_tab1.addWidget(self.canvas_tab1)
        tab1_layout.addLayout(right_tab1, stretch=3)

        self.tabs.addTab(self.tab1, "Screen 1: Target Selection")

        # SCREEN 2: Top 10 Trajectory Results
        self.tab2 = QWidget()
        tab2_layout = QHBoxLayout(self.tab2)

        data_layout = QVBoxLayout()
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(["No.", "Distance (mm)", "Bone Thickness (mm)", "Angle", "Density", "Score"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.cellClicked.connect(self.on_table_click)
        data_layout.addWidget(self.results_table)
        
        self.result_info = QLabel("Optimal trajectory results will appear here.\n- Select a target to begin -")
        self.result_info.setStyleSheet("font-size: 13px; font-weight: bold; padding: 12px; background-color: #f2f4f4; border: 1px solid #bdc3c7;")
        data_layout.addWidget(self.result_info)
        
        tab2_layout.addLayout(data_layout, stretch=1)

        self.fig_tab2 = Figure(figsize=(14, 8))
        self.canvas_tab2 = FigureCanvas(self.fig_tab2)
        
        self.ax_axial = self.fig_tab2.add_subplot(1, 2, 1)
        self.ax_sagittal = self.fig_tab2.add_subplot(1, 2, 2)
        
        tab2_layout.addWidget(self.canvas_tab2, stretch=3)

        self.tabs.addTab(self.tab2, "Screen 2: Top 10 Trajectories")

        # SCREEN 3: Optimal Trajectory & Density Profile
        self.tab3 = QWidget()
        tab3_layout = QHBoxLayout(self.tab3)

        self.fig_tab3 = Figure(figsize=(14, 8))
        self.canvas_tab3 = FigureCanvas(self.fig_tab3)
        self.ax_best_axial = self.fig_tab3.add_subplot(1, 2, 1)
        self.ax_profile = self.fig_tab3.add_subplot(2, 2, 2)
        self.ax_gradient = self.fig_tab3.add_subplot(2, 2, 4)
        
        tab3_layout.addWidget(self.canvas_tab3)
        self.tabs.addTab(self.tab3, "Screen 3: Optimal Trajectory & Graphs")

        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

        self.canvas_tab1.mpl_connect('button_press_event', self.on_click_mri)
        self.show()

    def load_mri(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select MRI File", "", "NIfTI (*.nii *.nii.gz)")
        if path:
            self.mri_image = sitk.ReadImage(path)
            self.mri_array = sitk.GetArrayFromImage(self.mri_image)
            self.num_slices = self.mri_array.shape[0]
            self.spacing = self.mri_image.GetSpacing()
            self.slider.setMaximum(self.num_slices - 1)
            self.slider.setValue(self.num_slices // 2)
            self.redraw_tab1()

    def load_ct(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CT File", "", "NIfTI (*.nii *.nii.gz)")
        if path:
            self.ct_image = sitk.ReadImage(path)
            self.ct_array = sitk.GetArrayFromImage(self.ct_image)
            self.redraw_tab1()

    def change_slice(self, val):
        self.current_slice = val
        self.lbl_slice.setText(f"Slice: {self.current_slice}")
        self.redraw_tab1()

    def redraw_tab1(self):
        if self.mri_array is None and self.ct_array is None:
            return
        self.ax_mri.clear()
        self.ax_ct.clear()

        if self.mri_array is not None:
            self.ax_mri.imshow(self.mri_array[self.current_slice], cmap='gray')
            self.ax_mri.set_title(f"MRI Slice {self.current_slice}")
            self.ax_mri.axis('off')

        if self.ct_array is not None:
            self.ax_ct.imshow(self.ct_array[self.current_slice], cmap='gray', vmin=-1000, vmax=1500)
            self.ax_ct.set_title(f"CT Slice {self.current_slice}")
            self.ax_ct.axis('off')

        if self.target_point and self.target_point[2] == self.current_slice:
            self.ax_mri.plot(self.target_point[0], self.target_point[1], 'r+', markersize=16, markeredgewidth=3, label='Target Point')
            self.ax_mri.legend(loc='upper right')
            self.ax_ct.plot(self.target_point[0], self.target_point[1], 'r+', markersize=16, markeredgewidth=3, label='Target Point')
            self.ax_ct.legend(loc='upper right')

        self.fig_tab1.canvas.draw_idle()

    def on_click_mri(self, event):
        if event.inaxes != self.ax_mri or self.mri_array is None:
            return
        self.target_point = (int(event.xdata), int(event.ydata), self.current_slice)
        self.lbl_target.setText(f"Target: X:{self.target_point[0]}, Y:{self.target_point[1]}, Slice:{self.target_point[2]}")
        self.redraw_tab1()

    def confirm_target(self):
        if self.target_point is None or self.ct_array is None:
            QMessageBox.warning(self, "Warning", "Please select a target point on the MRI and load the coregistered CT.")
            return
        self.tabs.setCurrentIndex(1)
        self.run_multimodal_optimization()

    def run_multimodal_optimization(self):
        x_target, y_target, z_target = self.target_point
        slice_data = self.ct_array[z_target]
        all_results = []
        spacing_xy = np.mean(self.spacing[:2])

        angles = np.linspace(np.pi, 2 * np.pi, 60)
        for angle in angles:
            max_radius = max(self.ct_array.shape[2], self.ct_array.shape[1])
            x_ray = np.int_(x_target + np.cos(angle) * np.arange(max_radius))
            y_ray = np.int_(y_target + np.sin(angle) * np.arange(max_radius))

            valid_idx = (x_ray >= 0) & (x_ray < self.ct_array.shape[2]) & (y_ray >= 0) & (y_ray < self.ct_array.shape[1])
            x_ray, y_ray = x_ray[valid_idx], y_ray[valid_idx]

            if len(x_ray) == 0: continue
            intensities = slice_data[y_ray, x_ray]

            air_indices = np.where(intensities < -300)[0]
            if len(air_indices) == 0: continue

            skin_idx = air_indices[0]
            if skin_idx < 10: continue

            path_intensities = intensities[:skin_idx][::-1]
            entry_x = x_ray[skin_idx]
            entry_y = y_ray[skin_idx]

            distance = len(path_intensities) * spacing_xy
            skull_thickness = np.sum(path_intensities > 300) * spacing_xy
            angle_deg = np.abs(np.degrees(angle - (1.5 * np.pi)))
            if angle_deg > 90:
                angle_deg = 180 - angle_deg

            mean_density = np.mean(path_intensities)
            derivative = np.abs(np.diff(path_intensities))
            density_changes = np.sum(derivative > 50)

            all_results.append({
                'entry_point': (entry_x, entry_y),
                'distance': distance,
                'skull_thickness': skull_thickness,
                'angle': angle_deg,
                'mean_density': mean_density,
                'density_changes': density_changes,
                'profile': path_intensities,
                'score': 0
            })

        if not all_results:
            return

        distances = np.array([r['distance'] for r in all_results])
        thicknesses = np.array([r['skull_thickness'] for r in all_results])
        angles_arr = np.array([r['angle'] for r in all_results])
        mean_densities = np.array([r['mean_density'] for r in all_results])
        density_changes = np.array([r['density_changes'] for r in all_results])

        def normalize(arr):
            m = np.max(arr)
            return (arr / m) * 10 if m > 0 else arr

        dist_norm = normalize(distances)
        thick_norm = normalize(thicknesses)
        density_norm = normalize(mean_densities)
        changes_norm = normalize(density_changes)

        angle_norm = (angles_arr**4 / 30000) + 1
        sum_weights = self.W_DIST + self.W_THICK + self.W_ANGLE + self.W_DENSITY + self.W_CHANGES

        for i in range(len(all_results)):
            score = (
                (self.W_DIST * dist_norm[i]) +
                (self.W_THICK * thick_norm[i]) +
                (self.W_ANGLE * angle_norm[i]) +
                (self.W_DENSITY * density_norm[i]) +
                (self.W_CHANGES * changes_norm[i])
            ) / sum_weights
            all_results[i]['score'] = score

        all_results.sort(key=lambda x: x['score'])
        self.top_trajectories = all_results[:10]
        self.all_results = all_results

        self.results_table.setRowCount(10)
        for idx in range(10):
            res = self.top_trajectories[idx]
            self.results_table.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))
            self.results_table.setItem(idx, 1, QTableWidgetItem(f"{res['distance']:.1f}"))
            self.results_table.setItem(idx, 2, QTableWidgetItem(f"{res['skull_thickness']:.1f}"))
            self.results_table.setItem(idx, 3, QTableWidgetItem(f"{res['angle']:.1f}°"))
            self.results_table.setItem(idx, 4, QTableWidgetItem(f"{res['mean_density']:.1f}"))
            self.results_table.setItem(idx, 5, QTableWidgetItem(f"{res['score']:.4f}"))

        self.plot_trajectories(self.top_trajectories[0])

    def on_table_click(self, row, column):
        if row < len(self.top_trajectories):
            self.plot_trajectories(self.top_trajectories[row])

    def plot_trajectories(self, trajectory):
        x_target, y_target, z_target = self.target_point

        # SCREEN 2: Trajectory Results View
        self.ax_axial.clear()
        self.ax_axial.imshow(self.ct_array[z_target], cmap='gray', vmin=-1000, vmax=1500)
        self.ax_axial.set_title("Axial View - Top 10 Trajectories", fontsize=12, pad=10)
        self.ax_axial.axis('off')

        self.ax_axial.plot(x_target, y_target, 'r+', markersize=16, markeredgewidth=2, label="dACC Target")

        for i in range(min(10, len(self.top_trajectories))):
            path = self.top_trajectories[i]
            ex, ey = path['entry_point']
            c = '#2ecc71' if i == 0 else '#e74c3c'
            self.ax_axial.plot([ex, x_target], [ey, y_target], color=c, linewidth=1.5)

        ex, ey = trajectory['entry_point']
        self.ax_axial.plot(ex, ey, 'go', markersize=9, label="Optimal Transducer")
        self.ax_axial.legend(loc="upper right", fontsize=8)

        self.result_info.setText(
            f"Results:\n\n"
            f"· Transducer Position: (X:{ex:.0f}, Y:{ey:.0f})\n"
            f"· Distance: {trajectory['distance']:.1f} mm\n"
            f"· Bone Thickness: {trajectory['skull_thickness']:.1f} mm\n"
            f"· Angle: {trajectory['angle']:.1f}°\n"
            f"· Score: {trajectory['score']:.4f}"
        )

        self.ax_sagittal.clear()
        sagittal_slice = self.ct_array[:, :, x_target]
        self.ax_sagittal.imshow(sagittal_slice, cmap='gray', origin='lower', aspect=1.0)
        self.ax_sagittal.set_title("Sagittal View (Profile)", fontsize=12, pad=10)
        self.ax_sagittal.axis('off')

        self.ax_sagittal.plot(y_target, z_target, 'r+', markersize=12, markeredgewidth=2)
        self.ax_sagittal.plot(ey, z_target, 'go', markersize=8, label="Transducer")
        self.ax_sagittal.plot([ey, y_target], [z_target, z_target], 'g-', linewidth=2)
        self.ax_sagittal.legend(loc="upper right", fontsize=8)

        # SCREEN 3: Optimal Trajectory & Density Profile
        self.ax_best_axial.clear()
        self.ax_best_axial.imshow(self.ct_array[z_target], cmap='gray', vmin=-1000, vmax=1500)
        self.ax_best_axial.set_title("Optimal Trajectory", fontsize=12, pad=10)
        self.ax_best_axial.axis('off')
        
        self.ax_best_axial.plot(x_target, y_target, 'r+', markersize=16, markeredgewidth=2, label="Target")
        self.ax_best_axial.plot([ex, x_target], [ey, y_target], color='#27ae60', linewidth=3, label="Trajectory")
        self.ax_best_axial.plot(ex, ey, 'go', markersize=10, label="Transducer")
        self.ax_best_axial.legend(loc="upper right", fontsize=8)

        best_profile = trajectory['profile']
        x_axis = np.arange(len(best_profile))
        
        self.ax_profile.clear()
        self.ax_profile.fill_between(x_axis, -200, 2000, where=(best_profile < -150), color='lightblue', alpha=0.15)
        self.ax_profile.fill_between(x_axis, -200, 2000, where=(best_profile >= -150) & (best_profile <= 300), color='gray', alpha=0.15)
        self.ax_profile.fill_between(x_axis, -200, 2000, where=(best_profile > 300), color='salmon', alpha=0.2)
        
        self.ax_profile.plot(x_axis, best_profile, '#27ae60', linewidth=2, label='Density')
        
        peaks, _ = find_peaks(best_profile, height=400, distance=3)
        if len(peaks) > 0:
            self.ax_profile.plot(x_axis[peaks], best_profile[peaks], "r^", markersize=9, label="Density Peaks")

        self.ax_profile.set_title("Trajectory Density Profile", fontsize=10)
        self.ax_profile.set_ylabel("Intensity (HU)", fontsize=8)
        self.ax_profile.set_ylim(-200, 2000)
        self.ax_profile.grid(True)
        self.ax_profile.legend(loc='upper right', fontsize=7)

        self.ax_gradient.clear()
        gradient = np.diff(best_profile)
        self.ax_gradient.plot(x_axis[:-1], gradient, color='#e67e22', linewidth=1.5, label='Gradient')
        self.ax_gradient.axhline(y=0, color='black', linestyle='--', linewidth=1)
        self.ax_gradient.set_title("Density Change Rate", fontsize=10)
        self.ax_gradient.set_xlabel("Entry Distance", fontsize=8)
        self.ax_gradient.grid(True)

        self.fig_tab2.canvas.draw_idle()
        self.fig_tab3.canvas.draw_idle()


if __name__ == '__main__':
    qt_app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(qt_app.exec())