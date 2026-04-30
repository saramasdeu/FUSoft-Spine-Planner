import sys
import os
import numpy as np
import SimpleITK as sitk
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QSlider, QLabel, 
                             QFileDialog, QStackedWidget, QMessageBox,
                             QDoubleSpinBox, QFrame, QScrollArea, QGridLayout, QCheckBox)
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QPolygonF

pg.setConfigOptions(imageAxisOrder='row-major')
pg.setConfigOptions(useOpenGL=True)

def normalize_intensity(array, is_ct=False):
    array_f = array.astype(np.float32)
    if is_ct:
        vmin, vmax = -200, 1000
    else:
        vmin, vmax = np.percentile(array_f, [2, 98])
    
    if vmax <= vmin:
        return np.zeros_like(array_f)
    return np.clip((array_f - vmin) / (vmax - vmin + 1e-6), 0, 1)

class AnalysisResultWindow(QMainWindow):
    def __init__(self, mri_img, ct_reg_img, mri_mask, ct_mask, dice_val):
        super().__init__()
        self.setWindowTitle(f"FUSOFT - Validation Report (DICE: {dice_val:.4f})")
        self.resize(1400, 750)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(QLabel(f"<h2>Registration Analysis | DICE Score: {dice_val:.4f}</h2>"))
        
        grid_layout = QGridLayout()
        
        mri_data = normalize_intensity(sitk.GetArrayFromImage(mri_img))
        mri_mask_data = sitk.GetArrayFromImage(mri_mask)
        ct_reg_data = normalize_intensity(sitk.GetArrayFromImage(ct_reg_img), is_ct=True)
        ct_mask_data = sitk.GetArrayFromImage(ct_mask)
        
        # Pink Bone
        anatomy_overlay = np.zeros((*mri_data.shape, 3), dtype=np.float32)
        bone_mask = (sitk.GetArrayFromImage(ct_reg_img) > 200).astype(np.float32)
        for i in range(3):
            anatomy_overlay[..., i] = mri_data * 0.7
        anatomy_overlay[..., 0] = np.clip(anatomy_overlay[..., 0] + bone_mask * 0.7, 0, 1)
        anatomy_overlay[..., 2] = np.clip(anatomy_overlay[..., 2] + bone_mask * 0.7, 0, 1)
        
        # Match (Yellow)
        boundary_match = np.zeros_like(anatomy_overlay)
        boundary_match[..., 0] = ct_mask_data  # Red channel: CT
        boundary_match[..., 1] = mri_mask_data # Green channel: MRI
        
        visual_outputs = [
            (mri_data, "1. Original MRI"),
            (mri_mask_data, "2. Solid MRI Mask"),
            (ct_reg_data, "3. Co-registered CT"),
            (ct_mask_data, "4. Clean CT Mask"),
            (anatomy_overlay, "5. Anatomy Validation (Pink)"),
            (boundary_match, "6. Boundary Match (Yellow)")
        ]
        
        self.viewers = []
        for i, (data, title) in enumerate(visual_outputs):
            container = QWidget()
            v_layout = QVBoxLayout(container)
            v_layout.addWidget(QLabel(f"<b>{title}</b>"))
            viewer = pg.ImageView()
            viewer.ui.histogram.hide()
            viewer.ui.roiBtn.hide()
            viewer.ui.menuBtn.hide()
            viewer.setImage(data, autoLevels=(data.ndim == 3))
            self.viewers.append(viewer)
            v_layout.addWidget(viewer)
            grid_layout.addWidget(container, i // 3, i % 3)
            
        main_layout.addLayout(grid_layout)
        
        self.sync_slider = QSlider(Qt.Orientation.Horizontal)
        self.sync_slider.setRange(0, mri_data.shape[0] - 1)
        self.sync_slider.valueChanged.connect(self.synchronize_viewers)
        main_layout.addWidget(QLabel("Synchronized Slice Navigation:"))
        main_layout.addWidget(self.sync_slider)

    def synchronize_viewers(self, index):
        for viewer in self.viewers:
            viewer.setCurrentIndex(index)

class RegistrationGizmo(pg.GraphicsObject):
    def __init__(self, viewport):
        super().__init__()
        self.viewport = viewport
        self.radius = 55
        self.arrow_length = 80
        self.active_handle = None
        self.last_mouse_position = None
        self.setZValue(1000)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

    def boundingRect(self):
        return QRectF(-120, -120, 240, 240)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        alpha_base, alpha_active = 70, 200
        
        color_rot = QColor(255, 255, 0, alpha_active if self.active_handle == 'rot' else alpha_base)
        color_tx = QColor(255, 50, 50, alpha_active if self.active_handle == 'tx' else alpha_base)
        color_ty = QColor(50, 255, 50, alpha_active if self.active_handle == 'ty' else alpha_base)
        
        painter.setPen(QPen(color_rot, 3))
        painter.drawEllipse(QPointF(0, 0), self.radius, self.radius)
        
        painter.setPen(QPen(color_tx, 4))
        painter.drawLine(0, 0, self.arrow_length, 0)
        painter.setBrush(QBrush(color_tx))
        painter.drawPolygon(QPolygonF([QPointF(self.arrow_length, -8), QPointF(self.arrow_length + 15, 0), QPointF(self.arrow_length, 8)]))
        
        painter.setPen(QPen(color_ty, 4))
        painter.drawLine(0, 0, 0, -self.arrow_length)
        painter.setBrush(QBrush(color_ty))
        painter.drawPolygon(QPolygonF([QPointF(-8, -self.arrow_length), QPointF(0, -self.arrow_length - 15), QPointF(8, -self.arrow_length)]))

    def mousePressEvent(self, event):
        pos = event.pos()
        dist = np.sqrt(pos.x()**2 + pos.y()**2)
        if abs(dist - self.radius) < 15:
            self.active_handle = 'rot'
        elif abs(pos.y()) < 25 and 0 < pos.x() < self.arrow_length + 25:
            self.active_handle = 'tx'
        elif abs(pos.x()) < 25 and -self.arrow_length - 25 < pos.y() < 0:
            self.active_handle = 'ty'
        else:
            self.active_handle = None
            event.ignore()
            return
        self.last_mouse_position = event.scenePos()
        event.accept()

    def mouseMoveEvent(self, event):
        if self.active_handle is None:
            return
        
        modifiers = QApplication.keyboardModifiers()
        sensitivity = 0.1 if modifiers == Qt.KeyboardModifier.ShiftModifier else 1.0
        
        image_item = self.viewport.image_view.getImageItem()
        current_pos = image_item.mapFromScene(event.scenePos())
        previous_pos = image_item.mapFromScene(self.last_mouse_position)
        delta = (current_pos - previous_pos) * sensitivity
        self.last_mouse_position = event.scenePos()
        
        spacing = self.viewport.main_window.mri_image.GetSpacing()
        
        if self.active_handle in ['tx', 'ty']:
            self.viewport.main_window.apply_gizmo_transform(
                self.viewport.orientation, 
                dx=delta.x() * spacing[0], 
                dy=-delta.y() * spacing[1], 
                is_live=True
            )
        elif self.active_handle == 'rot':
            angle = np.degrees(np.arctan2(event.pos().y(), event.pos().x()))
            if not hasattr(self, 'prev_angle'): self.prev_angle = angle
            self.viewport.main_window.apply_gizmo_transform(
                self.viewport.orientation, 
                dr=(angle - self.prev_angle) * sensitivity * 1.5, 
                is_live=True
            )
            self.prev_angle = angle
        event.accept()

    def mouseReleaseEvent(self, event):
        self.active_handle = None
        self.prev_angle = 0
        self.update()
        self.viewport.main_window.refresh_registration_view(is_live=False)
        event.accept()

class VolumeSliceViewer(QWidget):
    def __init__(self, label):
        super().__init__()
        layout = QVBoxLayout(self)
        self.title_label = QLabel(f"<b>{label}</b>")
        self.image_view = pg.ImageView()
        self.image_view.ui.histogram.hide()
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_view)
        layout.addWidget(self.slice_slider)
        
        self.volume_data = None
        self.orientation = label

    def load_volume(self, data):
        self.volume_data = data
        axis_map = {'Axial': 0, 'Coronal': 1, 'Sagittal': 2}
        max_slices = data.shape[axis_map.get(self.orientation.split()[-1], 0)]
        self.slice_slider.setRange(0, max_slices - 1)
        self.slice_slider.setValue(max_slices // 2)
        self.slice_slider.valueChanged.connect(self.update_slice)
        self.update_slice()

    def update_slice(self):
        if self.volume_data is None:
            return
        idx = self.slice_slider.value()
        if "Axial" in self.orientation:
            slice_img = self.volume_data[idx]
        elif "Coronal" in self.orientation:
            slice_img = self.volume_data[:, idx, :]
        else:
            slice_img = self.volume_data[:, :, idx]
            
        final_render = slice_img if "Axial" in self.orientation else np.flipud(slice_img)
        self.image_view.setImage(final_render, autoLevels=True)

class RegistrationViewport(QWidget):
    def __init__(self, label, main_window):
        super().__init__()
        self.orientation = label
        self.main_window = main_window
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(QLabel(f"<b>{label}</b>"))
        
        self.image_view = pg.ImageView(view=pg.PlotItem())
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        
        self.plot_item = self.image_view.getView()
        self.plot_item.setMouseEnabled(x=False, y=False)
        self.plot_item.setAspectLocked(True)
        
        self.gizmo = RegistrationGizmo(self)
        self.image_view.addItem(self.gizmo)
        
        self.nav_slider = QSlider(Qt.Orientation.Horizontal)
        layout.addWidget(self.image_view, stretch=10)
        layout.addWidget(self.nav_slider)
        
        self.volume = None
        self.is_first_load = True
        self.nav_slider.valueChanged.connect(self.render_slice)

    def set_volume_data(self, data):
        self.volume = data
        axis = 0 if "Axial" in self.orientation else 1 if "Coronal" in self.orientation else 2
        self.nav_slider.setRange(0, data.shape[axis] - 1)
        self.render_slice()

    def render_slice(self):
        if self.volume is None:
            return
        idx = self.nav_slider.value()
        if "Axial" in self.orientation:
            slice_img = self.volume[idx]
        elif "Coronal" in self.orientation:
            slice_img = self.volume[:, idx]
        else:
            slice_img = self.volume[:, :, idx]
            
        final_render = slice_img if "Axial" in self.orientation else np.flipud(slice_img)
        self.image_view.setImage(final_render, autoLevels=(self.volume.ndim == 3), autoRange=self.is_first_load)
        
        if self.is_first_load:
            self.plot_item.autoRange()
            self.is_first_load = False
            
        self.gizmo.setPos(slice_img.shape[1] / 2, slice_img.shape[0] / 2)

class FusoftApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FUSOFT - Image Registration System")
        self.showMaximized()
        
        self.mri_image = self.ct_full = self.ct_cropped = self.initial_transform = None
        self.ui_stack = QStackedWidget()
        self.setCentralWidget(self.ui_stack)
        self.setup_interface()

    def setup_interface(self):
        # 1: data loading
        p1 = QWidget()
        l1 = QVBoxLayout(p1)
        btn_layout = QHBoxLayout()
        
        self.btn_mri = QPushButton("1. Load MRI")
        self.btn_mri.clicked.connect(self.import_mri)
        self.btn_ct = QPushButton("2. Load CT")
        self.btn_ct.clicked.connect(self.import_ct)
        self.btn_to_crop = QPushButton("Proceed to Cropping")
        self.btn_to_crop.setEnabled(False)
        self.btn_to_crop.clicked.connect(lambda: self.ui_stack.setCurrentIndex(1))
        
        btn_layout.addWidget(self.btn_mri)
        btn_layout.addWidget(self.btn_ct)
        btn_layout.addWidget(self.btn_to_crop)
        l1.addLayout(btn_layout)
        
        view_grid = QHBoxLayout()
        self.mri_previews = {k: VolumeSliceViewer(f"MRI {k}") for k in ['Axial', 'Coronal', 'Sagittal']}
        self.ct_previews = {k: VolumeSliceViewer(f"CT {k}") for k in ['Axial', 'Coronal', 'Sagittal']}
        
        mri_col, ct_col = QVBoxLayout(), QVBoxLayout()
        for v in self.mri_previews.values(): mri_col.addWidget(v)
        for v in self.ct_previews.values(): ct_col.addWidget(v)
        
        view_grid.addLayout(mri_col)
        view_grid.addLayout(ct_col)
        l1.addLayout(view_grid)
        self.ui_stack.addWidget(p1)
        
        # 2: Cropping
        p2 = QWidget()
        l2 = QVBoxLayout(p2)
        crop_view_layout = QHBoxLayout()
        self.crop_axial_view = pg.ImageView()
        self.crop_sagittal_view = pg.ImageView()
        crop_view_layout.addWidget(self.crop_axial_view)
        crop_view_layout.addWidget(self.crop_sagittal_view)
        l2.addLayout(crop_view_layout)
        
        self.crop_slider = QSlider(Qt.Orientation.Horizontal)
        self.crop_slider.valueChanged.connect(self.update_crop_preview)
        l2.addWidget(self.crop_slider)
        
        btn_to_reg = QPushButton("Proceed to Registration")
        btn_to_reg.clicked.connect(self.finalize_cropping)
        l2.addWidget(btn_to_reg)
        self.ui_stack.addWidget(p2)
        
        # 3: Registration
        p3 = QWidget()
        self.reg_main_layout = QHBoxLayout(p3)
        self.reg_viewports = {k: RegistrationViewport(k, self) for k in ['Axial', 'Coronal', 'Sagittal']}
        for v in self.reg_viewports.values():
            self.reg_main_layout.addWidget(v, stretch=3)
            
        control_panel = QScrollArea()
        control_panel.setFixedWidth(240)
        control_panel.setWidgetResizable(True)
        panel_widget = QWidget()
        panel_layout = QVBoxLayout(panel_widget)
        
        self.lbl_dice = QLabel("DICE: 0.0000")
        self.lbl_dice.setStyleSheet("font-size: 18px; color: #f1c40f; font-weight: bold; background: #2c3e50; padding: 5px;")
        panel_layout.addWidget(self.lbl_dice)
        
        btn_auto = QPushButton("Automated Registration")
        btn_auto.clicked.connect(self.run_automated_registration)
        panel_layout.addWidget(btn_auto)
        
        self.chk_overlay = QCheckBox("Show Visual Overlay")
        self.chk_overlay.stateChanged.connect(lambda: self.refresh_registration_view(is_live=False))
        panel_layout.addWidget(self.chk_overlay)
        panel_layout.addWidget(QLabel("<hr>"))
        
        self.spin_boxes = {}
        transform_params = [('tx', 'Trans X', 250), ('ty', 'Trans Y', 250), ('tz', 'Trans Z', 250),
                            ('rx', 'Rot X', 180), ('ry', 'Rot Y', 180), ('rz', 'Rot Z', 180)]
        
        for key, name, limit in transform_params:
            group = QVBoxLayout()
            group.addWidget(QLabel(f"<b>{name}</b>"))
            row = QHBoxLayout()
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-limit * 10, limit * 10)
            spin = QDoubleSpinBox()
            spin.setRange(-limit, limit)
            spin.setSingleStep(0.1)
            
            slider.valueChanged.connect(lambda v, s=spin: s.setValue(v / 10.0))
            spin.valueChanged.connect(lambda v, s=slider: s.setValue(int(v * 10)))
            spin.valueChanged.connect(lambda: self.refresh_registration_view(is_live=False))
            
            self.spin_boxes[key] = spin
            row.addWidget(slider)
            row.addWidget(spin)
            group.addLayout(row)
            panel_layout.addLayout(group)
            
        btn_finalize = QPushButton("Finalize & Save Result")
        btn_finalize.clicked.connect(self.save_final_output)
        btn_finalize.setStyleSheet("background-color: #c0392b; color: white; height: 40px;")
        panel_layout.addWidget(btn_finalize)
        
        control_panel.setWidget(panel_widget)
        self.reg_main_layout.addWidget(control_panel, stretch=1)
        self.ui_stack.addWidget(p3)

    def process_robust_mask(self, mask, radius=5):
        closing = sitk.BinaryMorphologicalClosing(sitk.Cast(mask, sitk.sitkUInt8), [radius, radius, radius])
        return sitk.BinaryFillhole(closing)

    def import_mri(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select MRI Image", "", options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            self.mri_image = sitk.ReadImage(path, sitk.sitkFloat32)
            self.mri_array = normalize_intensity(sitk.GetArrayFromImage(self.mri_image))
            otsu = sitk.Cast(sitk.OtsuThreshold(self.mri_image, 0, 1), sitk.sitkUInt8)
            self.mri_mask = self.process_robust_mask(otsu, 5)
            for v in self.mri_previews.values():
                v.load_volume(np.flip(self.mri_array, axis=1))
            self.validate_input_state()

    def import_ct(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CT Image", "", options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            self.ct_full = sitk.ReadImage(path, sitk.sitkFloat32)
            self.ct_file_path = path
            array = normalize_intensity(sitk.GetArrayFromImage(self.ct_full), is_ct=True)
            for v in self.ct_previews.values():
                v.load_volume(np.flip(array, axis=1))
            self.crop_axial_view.setImage(array)
            self.crop_slider.setRange(0, array.shape[0] - 1)
            self.sagittal_baseline = np.flipud(array[:, :, array.shape[2] // 2])
            self.validate_input_state()

    def validate_input_state(self):
        if self.mri_image and self.ct_full:
            self.btn_to_crop.setEnabled(True)

    def calculate_dice_score(self, registered_ct):
        thresholded_ct = sitk.BinaryThreshold(registered_ct, -200.0, 3000.0, 1, 0)
        components = sitk.ConnectedComponent(sitk.Cast(thresholded_ct, sitk.sitkUInt8))
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(components)
        
        if stats.GetNumberOfLabels() > 0:
            largest_idx = max(range(1, stats.GetNumberOfLabels() + 1), key=lambda l: stats.GetNumberOfPixels(l))
            cleaned_ct = components == largest_idx
        else:
            cleaned_ct = thresholded_ct
            
        self.last_ct_mask = self.process_robust_mask(sitk.Cast(cleaned_ct, sitk.sitkUInt8), 5)
        overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
        try:
            overlap_filter.Execute(self.mri_mask, self.last_ct_mask)
            return overlap_filter.GetDiceCoefficient()
        except:
            return 0.0

    def refresh_registration_view(self, is_live=True):
        if not self.mri_image or not self.ct_cropped:
            return
            
        params = {k: v.value() for k, v in self.spin_boxes.items()}
        transform = sitk.Euler3DTransform()
        transform.SetCenter(self.initial_transform.GetCenter())
        transform.SetRotation(np.radians(params['rx']), np.radians(params['ry']), np.radians(params['rz']))
        transform.SetTranslation((params['tx'], params['ty'], params['tz']))
        
        resampled_ct = sitk.Resample(
            self.ct_cropped, self.mri_image, 
            sitk.CompositeTransform([self.initial_transform, transform]), 
            sitk.sitkLinear, -1000.0
        )
        self.final_ct_output = resampled_ct
        
        if not is_live:
            dice = self.calculate_dice_score(resampled_ct)
            self.current_dice_value = dice
            self.lbl_dice.setText(f"DICE: {dice:.4f}")
            
        mri_arr = self.mri_array
        ct_arr = sitk.GetArrayFromImage(resampled_ct)
        
        if self.chk_overlay.isChecked():
            h, w, d = mri_arr.shape
            rgb_render = np.zeros((h, w, d, 3), dtype=np.float32)
            bone_data = (ct_arr > 200).astype(np.float32)
            for i in range(3):
                rgb_render[..., i] = mri_arr * 0.7
            rgb_render[..., 0] = np.clip(rgb_render[..., 0] + bone_data * 0.7, 0, 1)
            rgb_render[..., 2] = np.clip(rgb_render[..., 2] + bone_data * 0.7, 0, 1)
            output_volume = rgb_render
        else:
            output_volume = (mri_arr * 0.5 + normalize_intensity(ct_arr, is_ct=True) * 0.8)
            
        for view in self.reg_viewports.values():
            view.set_volume_data(output_volume)

    def apply_gizmo_transform(self, orientation, dx=0, dy=0, dr=0, is_live=True):
        params = {k: v.value() for k, v in self.spin_boxes.items()}
        for sb in self.spin_boxes.values(): sb.blockSignals(True)
        
        if "Axial" in orientation:
            self.spin_boxes['tx'].setValue(params['tx'] + dx)
            self.spin_boxes['ty'].setValue(params['ty'] + dy)
            self.spin_boxes['rz'].setValue(params['rz'] + dr)
        elif "Coronal" in orientation:
            self.spin_boxes['tx'].setValue(params['tx'] + dx)
            self.spin_boxes['tz'].setValue(params['tz'] + dy)
            self.spin_boxes['ry'].setValue(params['ry'] + dr)
        else:
            self.spin_boxes['ty'].setValue(params['ty'] + dx)
            self.spin_boxes['tz'].setValue(params['tz'] + dy)
            self.spin_boxes['rx'].setValue(params['rx'] + dr)
            
        for sb in self.spin_boxes.values(): sb.blockSignals(False)
        self.refresh_registration_view(is_live=is_live)

    def update_crop_preview(self):
        z_idx = self.crop_slider.value()
        self.crop_axial_view.setCurrentIndex(z_idx)
        sag_view = self.sagittal_baseline.copy()
        pos = sag_view.shape[0] - z_idx
        sag_view[max(0, pos - 2):pos + 1, :] = 1.0
        self.crop_sagittal_view.setImage(sag_view, autoLevels=False)

    def finalize_cropping(self):
        z_idx = self.crop_slider.value()
        original_size = list(self.ct_full.GetSize())
        original_size[2] -= z_idx
        self.ct_cropped = sitk.RegionOfInterest(self.ct_full, original_size, [0, 0, z_idx])
        self.initial_transform = sitk.CenteredTransformInitializer(self.mri_obj if hasattr(self, 'mri_obj') else self.mri_image, self.ct_cropped, sitk.Euler3DTransform())
        self.ui_stack.setCurrentIndex(2)
        self.refresh_registration_view(is_live=False)

    def run_automated_registration(self):
        reg_method = sitk.ImageRegistrationMethod()
        reg_method.SetMetricAsMattesMutualInformation(50)
        reg_method.SetInterpolator(sitk.sitkLinear)
        reg_method.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-4, 50)
        reg_method.SetInitialTransform(sitk.Euler3DTransform(self.initial_transform))
        
        final_tx = reg_method.Execute(self.mri_image, self.ct_cropped)
        translations = final_tx.GetTranslation()
        for i, key in enumerate(['tx', 'ty', 'tz']):
            self.spin_boxes[key].setValue(translations[i])
        self.refresh_registration_view(is_live=False)

    def save_final_output(self):
        if not hasattr(self, 'final_ct_output'):
            return
        output_path = os.path.join(os.path.dirname(self.ct_file_path), "Registered_CT_Output.nii.gz")
        sitk.WriteImage(self.final_ct_output, output_path)
        
        self.report_window = AnalysisResultWindow(
            self.mri_image, self.final_ct_output, 
            self.mri_mask, self.last_ct_mask, self.current_dice_value
        )
        self.report_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FusoftApp()
    window.show()
    sys.exit(app.exec())