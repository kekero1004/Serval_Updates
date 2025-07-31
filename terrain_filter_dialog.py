"""
/***************************************************************************
 Serval Terrain Filter Dialog
 
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
                                QCheckBox, QGroupBox, QGridLayout, QSlider, QTabWidget, QWidget)
from qgis.PyQt.QtGui import QFont

class TerrainFilterDialog(QDialog):
    """Non-modal dialog for Terrain Filter with ArcGIS Pro PMF algorithm options."""
    
    # Signals for non-modal operation
    filtering_requested = pyqtSignal(dict)  # Emitted when Apply is clicked
    dialog_closed = pyqtSignal()  # Emitted when dialog is closed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Terrain Filter - Progressive Morphological Filter")
        self.setMinimumSize(450, 500)
        self.setModal(False)  # Non-modal dialog
        
        # Set window flags for non-modal behavior
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        
        # Default values based on ArcGIS Pro PMF
        self.filter_algorithm = 'Progressive Morphological Filter'
        self.initial_window_size = 8
        self.window_increase_constant = 2
        self.max_window_size = 32
        self.elevation_threshold = 2.0
        self.slope_threshold = 15.0
        self.iterations = 5
        self.preserve_intensity = 'medium'
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Terrain Filter - Progressive Morphological Filter")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Create tabs for organized options
        tab_widget = QTabWidget()
        
        # Basic Parameters Tab
        basic_tab = QWidget()
        basic_layout = QVBoxLayout()
        
        # Algorithm Selection Group
        algorithm_group = QGroupBox("Filter Algorithm")
        algorithm_layout = QGridLayout()
        
        algorithm_layout.addWidget(QLabel("Algorithm:"), 0, 0)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(['Progressive Morphological Filter', 'Simple Morphological Filter', 'Adaptive Filter'])
        self.algorithm_combo.currentTextChanged.connect(self.on_algorithm_changed)
        algorithm_layout.addWidget(self.algorithm_combo, 0, 1)
        
        algorithm_group.setLayout(algorithm_layout)
        basic_layout.addWidget(algorithm_group)
        
        # PMF Parameters Group
        pmf_group = QGroupBox("Progressive Morphological Filter Parameters")
        pmf_layout = QGridLayout()
        
        # Initial Window Size (wn)
        pmf_layout.addWidget(QLabel("Initial Window Size (wn):"), 0, 0)
        self.initial_window_spin = QSpinBox()
        self.initial_window_spin.setRange(3, 64)
        self.initial_window_spin.setValue(8)
        self.initial_window_spin.setSingleStep(2)
        pmf_layout.addWidget(self.initial_window_spin, 0, 1)
        pmf_layout.addWidget(QLabel("pixels"), 0, 2)
        
        # Window Increase Constant (c)
        pmf_layout.addWidget(QLabel("Increase Constant (c):"), 1, 0)
        self.increase_constant_spin = QSpinBox()
        self.increase_constant_spin.setRange(1, 8)
        self.increase_constant_spin.setValue(2)
        pmf_layout.addWidget(self.increase_constant_spin, 1, 1)
        
        # Maximum Window Size
        pmf_layout.addWidget(QLabel("Max Window Size:"), 2, 0)
        self.max_window_spin = QSpinBox()
        self.max_window_spin.setRange(8, 128)
        self.max_window_spin.setValue(32)
        self.max_window_spin.setSingleStep(8)
        pmf_layout.addWidget(self.max_window_spin, 2, 1)
        pmf_layout.addWidget(QLabel("pixels"), 2, 2)
        
        # Iterations
        pmf_layout.addWidget(QLabel("Iterations:"), 3, 0)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 20)
        self.iterations_spin.setValue(5)
        pmf_layout.addWidget(self.iterations_spin, 3, 1)
        
        pmf_group.setLayout(pmf_layout)
        basic_layout.addWidget(pmf_group)
        
        # Elevation Threshold Group
        elevation_group = QGroupBox("Elevation Difference Filtering")
        elevation_layout = QGridLayout()
        
        elevation_layout.addWidget(QLabel("Elevation Threshold:"), 0, 0)
        self.elevation_threshold_spin = QDoubleSpinBox()
        self.elevation_threshold_spin.setRange(0.1, 20.0)
        self.elevation_threshold_spin.setValue(2.0)
        self.elevation_threshold_spin.setSingleStep(0.1)
        self.elevation_threshold_spin.setDecimals(1)
        elevation_layout.addWidget(self.elevation_threshold_spin, 0, 1)
        elevation_layout.addWidget(QLabel("meters"), 0, 2)
        
        elevation_layout.addWidget(QLabel("Slope Threshold:"), 1, 0)
        self.slope_threshold_spin = QDoubleSpinBox()
        self.slope_threshold_spin.setRange(0.0, 90.0)
        self.slope_threshold_spin.setValue(15.0)
        self.slope_threshold_spin.setSingleStep(1.0)
        self.slope_threshold_spin.setDecimals(1)
        elevation_layout.addWidget(self.slope_threshold_spin, 1, 1)
        elevation_layout.addWidget(QLabel("degrees"), 1, 2)
        
        elevation_group.setLayout(elevation_layout)
        basic_layout.addWidget(elevation_group)
        
        basic_tab.setLayout(basic_layout)
        tab_widget.addTab(basic_tab, "Basic Parameters")
        
        # Advanced Parameters Tab
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout()
        
        # Morphological Operations Group
        morph_group = QGroupBox("Morphological Operations")
        morph_layout = QGridLayout()
        
        # Enable Opening
        self.opening_check = QCheckBox("Enable Opening (Erosion + Dilation)")
        self.opening_check.setChecked(True)
        morph_layout.addWidget(self.opening_check, 0, 0, 1, 2)
        
        # Enable Closing
        self.closing_check = QCheckBox("Enable Closing (Dilation + Erosion)")
        self.closing_check.setChecked(False)
        morph_layout.addWidget(self.closing_check, 1, 0, 1, 2)
        
        # Structure Element Shape
        morph_layout.addWidget(QLabel("Structure Element:"), 2, 0)
        self.structure_combo = QComboBox()
        self.structure_combo.addItems(['Rectangle', 'Ellipse', 'Cross', 'Diamond'])
        morph_layout.addWidget(self.structure_combo, 2, 1)
        
        morph_group.setLayout(morph_layout)
        advanced_layout.addWidget(morph_group)
        
        # Terrain Preservation Group
        preserve_group = QGroupBox("Terrain Preservation")
        preserve_layout = QGridLayout()
        
        preserve_layout.addWidget(QLabel("Preservation Intensity:"), 0, 0)
        self.preserve_combo = QComboBox()
        self.preserve_combo.addItems(['Low', 'Medium', 'High', 'Very High'])
        self.preserve_combo.setCurrentText('Medium')
        preserve_layout.addWidget(self.preserve_combo, 0, 1)
        
        # Edge Enhancement
        self.edge_enhance_check = QCheckBox("Enable Edge Enhancement")
        preserve_layout.addWidget(self.edge_enhance_check, 1, 0, 1, 2)
        
        # Gradient Preservation
        self.gradient_preserve_check = QCheckBox("Preserve Gradient Information")
        self.gradient_preserve_check.setChecked(True)
        preserve_layout.addWidget(self.gradient_preserve_check, 2, 0, 1, 2)
        
        preserve_group.setLayout(preserve_layout)
        advanced_layout.addWidget(preserve_group)
        
        # Quality Settings Group
        quality_group = QGroupBox("Processing Quality")
        quality_layout = QGridLayout()
        
        quality_layout.addWidget(QLabel("Processing Quality:"), 0, 0)
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(1, 5)
        self.quality_slider.setValue(3)
        self.quality_slider.setTickPosition(QSlider.TicksBelow)
        quality_layout.addWidget(self.quality_slider, 0, 1)
        
        self.quality_label = QLabel("Medium")
        quality_layout.addWidget(self.quality_label, 0, 2)
        self.quality_slider.valueChanged.connect(self.update_quality_label)
        
        # Noise Reduction
        self.noise_reduction_check = QCheckBox("Apply Noise Reduction")
        quality_layout.addWidget(self.noise_reduction_check, 1, 0, 1, 2)
        
        quality_group.setLayout(quality_layout)
        advanced_layout.addWidget(quality_group)
        
        advanced_tab.setLayout(advanced_layout)
        tab_widget.addTab(advanced_tab, "Advanced Options")
        
        layout.addWidget(tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Preview button
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self.preview_filter)
        button_layout.addWidget(self.preview_btn)
        
        button_layout.addStretch()
        
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_filter)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_parameters)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close_dialog)
        
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.close_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def on_algorithm_changed(self, algorithm):
        self.filter_algorithm = algorithm
        # Enable/disable PMF specific options
        pmf_enabled = algorithm == 'Progressive Morphological Filter'
        self.increase_constant_spin.setEnabled(pmf_enabled)
        self.max_window_spin.setEnabled(pmf_enabled)
        
    def update_quality_label(self, value):
        quality_names = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}
        self.quality_label.setText(quality_names[value])
        
    def preview_filter(self):
        """Preview the filter effect (placeholder for future implementation)."""
        # This could be implemented to show a preview of the filter effect
        pass
        
    def apply_filter(self):
        """Apply terrain filter with current parameters (non-modal operation)."""
        parameters = self.get_parameters()
        self.filtering_requested.emit(parameters)
        
    def reset_parameters(self):
        """Reset all parameters to default values."""
        self.algorithm_combo.setCurrentText('Progressive Morphological Filter')
        self.initial_window_spin.setValue(8)
        self.increase_constant_spin.setValue(2)
        self.max_window_spin.setValue(32)
        self.elevation_threshold_spin.setValue(2.0)
        self.slope_threshold_spin.setValue(15.0)
        self.iterations_spin.setValue(5)
        self.preserve_combo.setCurrentText('Medium')
        self.quality_slider.setValue(3)
        self.opening_check.setChecked(True)
        self.closing_check.setChecked(False)
        self.edge_enhance_check.setChecked(False)
        self.gradient_preserve_check.setChecked(True)
        self.noise_reduction_check.setChecked(False)
        
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
            'algorithm': self.filter_algorithm,
            'initial_window_size': self.initial_window_spin.value(),
            'increase_constant': self.increase_constant_spin.value(),
            'max_window_size': self.max_window_spin.value(),
            'elevation_threshold': self.elevation_threshold_spin.value(),
            'slope_threshold': self.slope_threshold_spin.value(),
            'iterations': self.iterations_spin.value(),
            'preserve_intensity': self.preserve_combo.currentText().lower(),
            'enable_opening': self.opening_check.isChecked(),
            'enable_closing': self.closing_check.isChecked(),
            'structure_element': self.structure_combo.currentText().lower(),
            'edge_enhancement': self.edge_enhance_check.isChecked(),
            'gradient_preservation': self.gradient_preserve_check.isChecked(),
            'noise_reduction': self.noise_reduction_check.isChecked(),
            'quality': self.quality_slider.value()
        }