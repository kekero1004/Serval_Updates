"""
/***************************************************************************
 Serval Vertex Interpolation Tool
 
    begin            : 2025-07-23
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

import math
import numpy as np
from qgis.PyQt.QtCore import Qt, pyqtSignal, QTimer
from qgis.PyQt.QtGui import QColor, QPen, QBrush, QPixmap, QCursor
from qgis.PyQt.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView
from qgis.core import QgsPointXY, QgsGeometry, QgsFeature, QgsVectorLayer, QgsProject, QgsWkbTypes, QgsField, QgsFields, QgsMessageLog, Qgis
from qgis.gui import QgsMapTool, QgsRubberBand, QgsVertexMarker
from qgis.PyQt.QtCore import QVariant
from .utils import icon_path
try:
    from scipy.interpolate import griddata
    from scipy.spatial import Delaunay
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class VertexInterpolationDialog(QDialog):
    """Dialog for managing interpolation vertices and parameters."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interpolate from Vertices")
        self.setMinimumSize(400, 500)
        self.control_points = []  # List of (x, y, z) tuples
        self.interpolation_method = 'TIN'
        self.preview_enabled = True
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Interpolation Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(['TIN', 'Spline', 'IDW'])
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        method_layout.addWidget(self.method_combo)
        layout.addLayout(method_layout)
        
        # Preview option
        preview_layout = QHBoxLayout()
        self.preview_btn = QPushButton("Enable Real-time Preview")
        self.preview_btn.setCheckable(True)
        self.preview_btn.setChecked(True)
        self.preview_btn.toggled.connect(self.toggle_preview)
        preview_layout.addWidget(self.preview_btn)
        layout.addLayout(preview_layout)
        
        # Control points table
        layout.addWidget(QLabel("Control Points:"))
        self.points_table = QTableWidget()
        self.points_table.setColumnCount(3)
        self.points_table.setHorizontalHeaderLabels(['X', 'Y', 'Z (Value)'])
        header = self.points_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.points_table.itemChanged.connect(self.on_point_value_changed)
        layout.addWidget(self.points_table)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.add_point_btn = QPushButton("Add Point")
        self.add_point_btn.clicked.connect(self.add_point_mode)
        btn_layout.addWidget(self.add_point_btn)
        
        self.delete_point_btn = QPushButton("Delete Selected")
        self.delete_point_btn.clicked.connect(self.delete_selected_point)
        btn_layout.addWidget(self.delete_point_btn)
        
        self.clear_points_btn = QPushButton("Clear All")
        self.clear_points_btn.clicked.connect(self.clear_all_points)
        btn_layout.addWidget(self.clear_points_btn)
        
        layout.addLayout(btn_layout)
        
        # Apply/Cancel buttons
        apply_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Interpolation")
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        apply_layout.addWidget(self.apply_btn)
        apply_layout.addWidget(self.cancel_btn)
        layout.addLayout(apply_layout)
        
        self.setLayout(layout)
        
    def on_method_changed(self, method):
        self.interpolation_method = method
        self.interpolation_changed.emit()
        
    def toggle_preview(self, enabled):
        self.preview_enabled = enabled
        self.preview_btn.setText("Disable Real-time Preview" if enabled else "Enable Real-time Preview")
        if enabled:
            self.interpolation_changed.emit()
            
    def add_point_mode(self):
        self.add_point_signal.emit()
        
    def add_control_point(self, x, y, z=0.0):
        """Add a new control point."""
        self.control_points.append((x, y, z))
        self.update_points_table()
        if self.preview_enabled:
            self.interpolation_changed.emit()
            
    def delete_selected_point(self):
        """Delete selected point from table."""
        current_row = self.points_table.currentRow()
        if 0 <= current_row < len(self.control_points):
            del self.control_points[current_row]
            self.update_points_table()
            self.point_deleted.emit(current_row)
            if self.preview_enabled:
                self.interpolation_changed.emit()
                
    def clear_all_points(self):
        """Clear all control points."""
        self.control_points.clear()
        self.update_points_table()
        self.points_cleared.emit()
        
    def update_points_table(self):
        """Update the points table with current control points."""
        self.points_table.setRowCount(len(self.control_points))
        for i, (x, y, z) in enumerate(self.control_points):
            self.points_table.setItem(i, 0, QTableWidgetItem(f"{x:.3f}"))
            self.points_table.setItem(i, 1, QTableWidgetItem(f"{y:.3f}"))
            self.points_table.setItem(i, 2, QTableWidgetItem(f"{z:.3f}"))
            
    def on_point_value_changed(self, item):
        """Handle changes to point values in the table."""
        if item.column() == 2:  # Z value changed
            row = item.row()
            try:
                new_z = float(item.text())
                if 0 <= row < len(self.control_points):
                    x, y, _ = self.control_points[row]
                    self.control_points[row] = (x, y, new_z)
                    if self.preview_enabled:
                        self.interpolation_changed.emit()
            except ValueError:
                # Restore original value if invalid
                if 0 <= row < len(self.control_points):
                    _, _, z = self.control_points[row]
                    item.setText(f"{z:.3f}")
                    
    def get_interpolation_data(self):
        """Return current interpolation settings and control points."""
        return {
            'method': self.interpolation_method,
            'points': self.control_points.copy(),
            'preview_enabled': self.preview_enabled
        }
    
    # Signals
    interpolation_changed = pyqtSignal()
    add_point_signal = pyqtSignal()
    point_deleted = pyqtSignal(int)
    points_cleared = pyqtSignal()


class VertexInterpolationMapTool(QgsMapTool):
    """Interactive map tool for vertex-based interpolation."""
    
    def __init__(self, canvas, dialog, raster_handler):
        super().__init__(canvas)
        self.canvas = canvas
        self.dialog = dialog
        self.raster_handler = raster_handler
        self.vertex_markers = []
        self.rubber_band = None
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        self.preview_timer.setSingleShot(True)
        self.adding_point = False
        
        # Set cursor
        cursor_pixmap = QPixmap(icon_path('draw_tool.svg'))
        self.setCursor(QCursor(cursor_pixmap, hotX=2, hotY=22))
        
        # Connect dialog signals
        self.dialog.add_point_signal.connect(self.enable_point_adding)
        self.dialog.point_deleted.connect(self.remove_vertex_marker)
        self.dialog.points_cleared.connect(self.clear_vertex_markers)
        self.dialog.interpolation_changed.connect(self.schedule_preview_update)
        
    def enable_point_adding(self):
        """Enable point adding mode."""
        self.adding_point = True
        
    def canvasReleaseEvent(self, event):
        """Handle mouse click to add control points."""
        if event.button() == Qt.LeftButton and self.adding_point:
            # Convert screen coordinates to map coordinates
            map_point = self.toMapCoordinates(event.pos())
            
            # Get elevation value at this point from raster
            try:
                # Transform to raster CRS if needed
                if self.raster_handler.crs_transform:
                    raster_point = self.raster_handler.crs_transform.transform(map_point)
                else:
                    raster_point = map_point
                    
                # Get raster value at point
                ident_vals = self.raster_handler.provider.identify(
                    raster_point, 
                    self.raster_handler.provider.IdentifyFormatValue
                ).results()
                
                if ident_vals:
                    current_value = list(ident_vals.values())[0]
                    if current_value is not None:
                        # Add control point with current raster value
                        self.dialog.add_control_point(map_point.x(), map_point.y(), float(current_value))
                        self.add_vertex_marker(map_point)
                        self.adding_point = False
                        
            except Exception as e:
                # Fallback: add point with zero value
                self.dialog.add_control_point(map_point.x(), map_point.y(), 0.0)
                self.add_vertex_marker(map_point)
                self.adding_point = False
                
    def add_vertex_marker(self, point):
        """Add visual marker for control point."""
        marker = QgsVertexMarker(self.canvas)
        marker.setCenter(point)
        marker.setColor(QColor(255, 0, 0))
        marker.setIconSize(10)
        marker.setIconType(QgsVertexMarker.ICON_CIRCLE)
        marker.setPenWidth(2)
        self.vertex_markers.append(marker)
        
    def remove_vertex_marker(self, index):
        """Remove vertex marker at given index."""
        if 0 <= index < len(self.vertex_markers):
            marker = self.vertex_markers.pop(index)
            self.canvas.scene().removeItem(marker)
            
    def clear_vertex_markers(self):
        """Remove all vertex markers."""
        for marker in self.vertex_markers:
            self.canvas.scene().removeItem(marker)
        self.vertex_markers.clear()
        
    def schedule_preview_update(self):
        """Schedule preview update with delay."""
        if self.dialog.preview_enabled:
            self.preview_timer.start(500)  # 500ms delay
            
    def update_preview(self):
        """Update interpolation preview."""
        # This would show a preview of the interpolation result
        # For now, we'll just prepare the interpolation data
        pass
        
    def deactivate(self):
        """Clean up when tool is deactivated."""
        self.clear_vertex_markers()
        if self.rubber_band:
            self.canvas.scene().removeItem(self.rubber_band)
        super().deactivate()


class VertexInterpolator:
    """Handles different vertex-based interpolation methods."""
    
    @staticmethod
    def interpolate_tin(control_points, grid_x, grid_y):
        """Triangulated Irregular Network interpolation."""
        if len(control_points) < 3:
            raise ValueError("TIN interpolation requires at least 3 control points")
            
        if not SCIPY_AVAILABLE:
            QgsMessageLog.logMessage("SciPy not available, falling back to IDW", "Serval", Qgis.Warning)
            return VertexInterpolator.interpolate_idw(control_points, grid_x, grid_y)
            
        points = np.array([(x, y) for x, y, z in control_points])
        values = np.array([z for x, y, z in control_points])
        
        # Create grid points for interpolation
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        
        # Perform linear interpolation using triangulation
        interpolated = griddata(points, values, grid_points, method='linear')
        
        # Fill NaN values with nearest neighbor
        nan_mask = np.isnan(interpolated)
        if np.any(nan_mask):
            nearest = griddata(points, values, grid_points, method='nearest')
            interpolated[nan_mask] = nearest[nan_mask]
            
        return interpolated.reshape(grid_x.shape)
        
    @staticmethod
    def interpolate_spline(control_points, grid_x, grid_y):
        """Spline interpolation using radial basis functions."""
        if len(control_points) < 3:
            raise ValueError("Spline interpolation requires at least 3 control points")
            
        if not SCIPY_AVAILABLE:
            QgsMessageLog.logMessage("SciPy not available, falling back to IDW", "Serval", Qgis.Warning)
            return VertexInterpolator.interpolate_idw(control_points, grid_x, grid_y)
            
        try:
            from scipy.interpolate import Rbf
            
            points = np.array([(x, y) for x, y, z in control_points])
            values = np.array([z for x, y, z in control_points])
            
            # Create RBF interpolator
            rbf = Rbf(points[:, 0], points[:, 1], values, function='thin_plate')
            
            # Interpolate to grid
            interpolated = rbf(grid_x, grid_y)
            
            return interpolated
            
        except ImportError:
            # Fallback to TIN if scipy RBF not available
            QgsMessageLog.logMessage("SciPy RBF not available, falling back to TIN", "Serval", Qgis.Warning)
            return VertexInterpolator.interpolate_tin(control_points, grid_x, grid_y)
            
    @staticmethod
    def interpolate_idw(control_points, grid_x, grid_y, power=2):
        """Inverse Distance Weighting interpolation."""
        if len(control_points) < 1:
            raise ValueError("IDW interpolation requires at least 1 control point")
            
        points = np.array([(x, y) for x, y, z in control_points])
        values = np.array([z for x, y, z in control_points])
        
        # Initialize result grid
        result = np.zeros_like(grid_x)
        
        # For each grid point, calculate IDW
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                gx, gy = grid_x[i, j], grid_y[i, j]
                
                # Calculate distances to all control points
                distances = np.sqrt((points[:, 0] - gx)**2 + (points[:, 1] - gy)**2)
                
                # Handle points that are exactly on control points
                zero_distance = distances == 0
                if np.any(zero_distance):
                    result[i, j] = values[zero_distance][0]
                else:
                    # Calculate weights
                    weights = 1.0 / (distances ** power)
                    weights /= np.sum(weights)
                    
                    # Weighted average
                    result[i, j] = np.sum(weights * values)
                    
        return result
        
    @staticmethod
    def interpolate(method, control_points, grid_x, grid_y):
        """Main interpolation function that dispatches to specific methods."""
        if method == 'TIN':
            return VertexInterpolator.interpolate_tin(control_points, grid_x, grid_y)
        elif method == 'Spline':
            return VertexInterpolator.interpolate_spline(control_points, grid_x, grid_y)
        elif method == 'IDW':
            return VertexInterpolator.interpolate_idw(control_points, grid_x, grid_y)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")