"""
/***************************************************************************
 Serval Interpolation Dialogs
 
    begin            : 2025-07-25
    copyright        : (C) 2025 Enhanced Serval Plugin
    email            : support@serval.co
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                                QComboBox, QLabel, QSpinBox, QDoubleSpinBox, 
                                QCheckBox, QGroupBox, QGridLayout, QSlider)
from qgis.PyQt.QtGui import QFont

class InterpolateFromVerticesDialog(QDialog):
    """Non-modal dialog for Interpolate from Vertices with ArcGIS Pro-like options."""
    
    # Signals for non-modal operation
    interpolation_requested = pyqtSignal(dict)  # Emitted when Apply is clicked
    dialog_closed = pyqtSignal()  # Emitted when dialog is closed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interpolate from Vertices")
        self.setMinimumSize(400, 350)
        self.setModal(False)  # Non-modal dialog
        
        # Set window flags for non-modal behavior
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        
        # Default values
        self.interpolation_method = 'Linear TIN'
        self.blend_width = 0
        self.use_natural_neighbors = False
        self.triangulation_method = 'Delaunay'
        self.edge_length_factor = 2.0
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Interpolate from Vertices")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Interpolation Method Group
        method_group = QGroupBox("Interpolation Method")
        method_layout = QGridLayout()
        
        method_layout.addWidget(QLabel("Method:"), 0, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(['Linear TIN', 'Natural Neighbors', 'IDW'])
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        method_layout.addWidget(self.method_combo, 0, 1)
        
        method_layout.addWidget(QLabel("Triangulation:"), 1, 0)
        self.triangulation_combo = QComboBox()
        self.triangulation_combo.addItems(['Delaunay', 'Constrained Delaunay'])
        method_layout.addWidget(self.triangulation_combo, 1, 1)
        
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        # Advanced Options Group
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QGridLayout()
        
        # Edge Length Factor
        advanced_layout.addWidget(QLabel("Edge Length Factor:"), 0, 0)
        self.edge_length_spin = QDoubleSpinBox()
        self.edge_length_spin.setRange(0.5, 10.0)
        self.edge_length_spin.setValue(2.0)
        self.edge_length_spin.setSingleStep(0.1)
        advanced_layout.addWidget(self.edge_length_spin, 0, 1)
        
        # Blend Options
        self.blend_check = QCheckBox("Enable Blend")
        self.blend_check.toggled.connect(self.on_blend_toggled)
        advanced_layout.addWidget(self.blend_check, 1, 0)
        
        advanced_layout.addWidget(QLabel("Blend Width:"), 2, 0)
        self.blend_spin = QSpinBox()
        self.blend_spin.setRange(0, 50)
        self.blend_spin.setValue(0)
        self.blend_spin.setEnabled(False)
        advanced_layout.addWidget(self.blend_spin, 2, 1)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        # Quality Settings Group
        quality_group = QGroupBox("Quality Settings")
        quality_layout = QGridLayout()
        
        quality_layout.addWidget(QLabel("Interpolation Quality:"), 0, 0)
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(1, 5)
        self.quality_slider.setValue(3)
        self.quality_slider.setTickPosition(QSlider.TicksBelow)
        quality_layout.addWidget(self.quality_slider, 0, 1)
        
        self.quality_label = QLabel("Medium")
        quality_layout.addWidget(self.quality_label, 0, 2)
        self.quality_slider.valueChanged.connect(self.update_quality_label)
        
        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_interpolation)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close_dialog)
        
        button_layout.addStretch()
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.close_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def on_method_changed(self, method):
        self.interpolation_method = method
        # Enable/disable triangulation combo based on method
        self.triangulation_combo.setEnabled(method in ['Linear TIN', 'Natural Neighbors'])
        
    def on_blend_toggled(self, enabled):
        self.blend_spin.setEnabled(enabled)
        
    def update_quality_label(self, value):
        quality_names = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}
        self.quality_label.setText(quality_names[value])
        
    def apply_interpolation(self):
        """Apply interpolation with current parameters (non-modal operation)."""
        parameters = self.get_parameters()
        self.interpolation_requested.emit(parameters)
        
    def close_dialog(self):
        """Close the dialog and emit signal."""
        self.dialog_closed.emit()
        self.close()
        
    def closeEvent(self, event):
        """Handle dialog close event."""
        self.dialog_closed.emit()
        event.accept()
        
    def get_parameters(self):
        """Return current parameter settings."""
        return {
            'method': self.interpolation_method,
            'triangulation': self.triangulation_combo.currentText(),
            'edge_length_factor': self.edge_length_spin.value(),
            'blend_enabled': self.blend_check.isChecked(),
            'blend_width': self.blend_spin.value(),
            'quality': self.quality_slider.value()
        }


class InterpolateFromEdgesDialog(QDialog):
    """Non-modal dialog for Interpolate from Edges with ArcGIS Pro-like options."""
    
    # Signals for non-modal operation
    interpolation_requested = pyqtSignal(dict)  # Emitted when Apply is clicked
    dialog_closed = pyqtSignal()  # Emitted when dialog is closed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interpolate from Edges")
        self.setMinimumSize(400, 320)
        self.setModal(False)  # Non-modal dialog
        
        # Set window flags for non-modal behavior
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        
        # Default values
        self.interpolation_method = 'Linear'
        self.sampling_method = 'Edge-based'
        self.distance_power = 2.0
        self.search_radius = 10
        self.min_points = 3
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Interpolate from Edges")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Interpolation Method Group
        method_group = QGroupBox("Interpolation Method")
        method_layout = QGridLayout()
        
        method_layout.addWidget(QLabel("Method:"), 0, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(['Linear', 'Bilinear', 'Distance Weighted'])
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        method_layout.addWidget(self.method_combo, 0, 1)
        
        method_layout.addWidget(QLabel("Sampling:"), 1, 0)
        self.sampling_combo = QComboBox()
        self.sampling_combo.addItems(['Edge-based', 'Triangle Edge', 'Nearest Edge'])
        method_layout.addWidget(self.sampling_combo, 1, 1)
        
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        # Distance Parameters Group
        distance_group = QGroupBox("Distance Parameters")
        distance_layout = QGridLayout()
        
        distance_layout.addWidget(QLabel("Search Radius:"), 0, 0)
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(1, 100)
        self.radius_spin.setValue(10)
        distance_layout.addWidget(self.radius_spin, 0, 1)
        
        distance_layout.addWidget(QLabel("Distance Power:"), 1, 0)
        self.power_spin = QDoubleSpinBox()
        self.power_spin.setRange(0.1, 5.0)
        self.power_spin.setValue(2.0)
        self.power_spin.setSingleStep(0.1)
        distance_layout.addWidget(self.power_spin, 1, 1)
        
        distance_layout.addWidget(QLabel("Min Points:"), 2, 0)
        self.min_points_spin = QSpinBox()
        self.min_points_spin.setRange(1, 20)
        self.min_points_spin.setValue(3)
        distance_layout.addWidget(self.min_points_spin, 2, 1)
        
        distance_group.setLayout(distance_layout)
        layout.addWidget(distance_group)
        
        # Smoothing Options Group
        smooth_group = QGroupBox("Smoothing Options")
        smooth_layout = QGridLayout()
        
        self.smooth_check = QCheckBox("Enable Smoothing")
        smooth_layout.addWidget(self.smooth_check, 0, 0)
        
        smooth_layout.addWidget(QLabel("Smooth Factor:"), 1, 0)
        self.smooth_spin = QDoubleSpinBox()
        self.smooth_spin.setRange(0.0, 1.0)
        self.smooth_spin.setValue(0.5)
        self.smooth_spin.setSingleStep(0.1)
        self.smooth_spin.setEnabled(False)
        smooth_layout.addWidget(self.smooth_spin, 1, 1)
        
        self.smooth_check.toggled.connect(self.smooth_spin.setEnabled)
        
        smooth_group.setLayout(smooth_layout)
        layout.addWidget(smooth_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_interpolation)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close_dialog)
        
        button_layout.addStretch()
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.close_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def on_method_changed(self, method):
        self.interpolation_method = method
        # Enable/disable power parameter based on method
        self.power_spin.setEnabled(method == 'Distance Weighted')
        
    def apply_interpolation(self):
        """Apply interpolation with current parameters (non-modal operation)."""
        parameters = self.get_parameters()
        self.interpolation_requested.emit(parameters)
        
    def close_dialog(self):
        """Close the dialog and emit signal."""
        self.dialog_closed.emit()
        self.close()
        
    def closeEvent(self, event):
        """Handle dialog close event."""
        self.dialog_closed.emit()
        event.accept()
        
    def get_parameters(self):
        """Return current parameter settings."""
        return {
            'method': self.interpolation_method,
            'sampling': self.sampling_combo.currentText(),
            'search_radius': self.radius_spin.value(),
            'distance_power': self.power_spin.value(),
            'min_points': self.min_points_spin.value(),
            'smooth_enabled': self.smooth_check.isChecked(),
            'smooth_factor': self.smooth_spin.value()
        }