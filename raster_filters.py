"""
/***************************************************************************
 Serval Advanced Raster Filters
 
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

import numpy as np
import math
from scipy import ndimage, interpolate
from scipy.spatial.distance import cdist
from scipy.stats import zscore, median_abs_deviation
from qgis.core import QgsMessageLog, Qgis

class RasterFilters:
    """Advanced raster filtering functions for Serval plugin."""
    
    @staticmethod
    def fill_voids(data, method='idw', search_radius=10, min_neighbors=4, quality_threshold=0.7):
        """
        Fill NoData areas using interpolation methods.
        
        Args:
            data (numpy.ndarray): Input raster data
            method (str): Interpolation method ('idw', 'kriging', 'natural_neighbor', 'bilinear')
            search_radius (int): Search radius in pixels
            min_neighbors (int): Minimum number of neighbors required
            quality_threshold (float): Quality threshold for interpolation
            
        Returns:
            numpy.ndarray: Filled raster data
        """
        filled_data = data.copy()
        
        # Find NoData pixels (don't treat 0 as NoData for elevation data)
        nodata_mask = np.isnan(data) | (data == -9999)
        if not np.any(nodata_mask):
            return filled_data
            
        # Get coordinates of valid and invalid pixels
        rows, cols = np.indices(data.shape)
        valid_mask = ~nodata_mask
        
        if method == 'idw':
            filled_data = RasterFilters._fill_idw(data, nodata_mask, search_radius, min_neighbors)
        elif method == 'bilinear':
            filled_data = RasterFilters._fill_bilinear(data, nodata_mask)
        elif method in ['kriging', 'natural_neighbor']:
            # Simplified implementation - fallback to IDW
            filled_data = RasterFilters._fill_idw(data, nodata_mask, search_radius, min_neighbors)
            
        return filled_data
    
    @staticmethod
    def _fill_idw(data, nodata_mask, search_radius, min_neighbors, power=2):
        """Inverse Distance Weighting interpolation."""
        filled_data = data.copy()
        rows, cols = np.indices(data.shape)
        
        # Get coordinates
        nodata_coords = np.column_stack((rows[nodata_mask], cols[nodata_mask]))
        valid_coords = np.column_stack((rows[~nodata_mask], cols[~nodata_mask]))
        valid_values = data[~nodata_mask]
        
        for i, (row, col) in enumerate(nodata_coords):
            # Calculate distances to all valid points
            distances = cdist([(row, col)], valid_coords)[0]
            
            # Find points within search radius
            within_radius = distances <= search_radius
            if np.sum(within_radius) < min_neighbors:
                continue
                
            # Get nearest points within radius
            near_distances = distances[within_radius]
            near_values = valid_values[within_radius]
            
            # Avoid division by zero
            near_distances[near_distances == 0] = 1e-10
            
            # Calculate weights (inverse distance)
            weights = 1.0 / (near_distances ** power)
            weights /= np.sum(weights)
            
            # Interpolate value
            filled_data[row, col] = np.sum(weights * near_values)
            
        return filled_data
    
    @staticmethod
    def fill_voids_selective(data, selection_mask, **kwargs):
        """Apply fill_voids only to selected areas."""
        result = data.copy()
        if not np.any(selection_mask):
            return result
            
        # Find NoData areas within selection
        selected_nodata = selection_mask & (np.isnan(data) | (data == -9999))
        if not np.any(selected_nodata):
            return result
            
        # Apply fill_voids to the entire data for context
        filled_data = RasterFilters.fill_voids(data, **kwargs)
        
        # Only update selected NoData cells
        result[selected_nodata] = filled_data[selected_nodata]
        return result
    
    @staticmethod
    def _fill_bilinear(data, nodata_mask):
        """Simple bilinear interpolation fill."""
        filled_data = data.copy()
        
        # Create meshgrid
        rows, cols = np.indices(data.shape)
        valid_mask = ~nodata_mask
        
        if np.sum(valid_mask) < 4:  # Need at least 4 points for interpolation
            return filled_data
            
        # Get valid coordinates and values
        valid_rows = rows[valid_mask]
        valid_cols = cols[valid_mask]
        valid_values = data[valid_mask]
        
        # Interpolate using griddata
        try:
            from scipy.interpolate import griddata
            points = np.column_stack((valid_rows, valid_cols))
            grid_rows, grid_cols = np.mgrid[0:data.shape[0], 0:data.shape[1]]
            interpolated = griddata(points, valid_values, (grid_rows, grid_cols), method='linear')
            
            # Fill only the NoData areas
            filled_data[nodata_mask] = interpolated[nodata_mask]
        except ImportError:
            QgsMessageLog.logMessage("SciPy not available for advanced interpolation", "Serval", Qgis.Warning)
            
        return filled_data
    
    @staticmethod
    def terrain_filter(data, filter_size=5, slope_threshold=15, height_threshold=2, iterations=3, preserve_intensity='medium'):
        """
        Filter terrain data to remove above-ground objects.
        
        Args:
            data (numpy.ndarray): Input DSM data
            filter_size (int): Filter kernel size
            slope_threshold (float): Slope threshold in degrees
            height_threshold (float): Height difference threshold in meters
            iterations (int): Number of filter iterations
            preserve_intensity (str): Terrain preservation intensity ('low', 'medium', 'high')
            
        Returns:
            numpy.ndarray: Filtered DTM data
        """
        filtered_data = data.copy()
        
        # Handle NaN values - replace with interpolated values temporarily
        nan_mask = np.isnan(filtered_data)
        if np.any(nan_mask):
            valid_mask = ~nan_mask
            if np.any(valid_mask):
                # Simple nearest neighbor fill for processing
                filled_data = filtered_data.copy()
                filled_data[nan_mask] = np.nanmean(filtered_data)
                filtered_data = filled_data
        
        # Progressive morphological filter
        for i in range(iterations):
            # Apply morphological opening
            kernel_size = filter_size + i
            kernel = np.ones((kernel_size, kernel_size))
            opened = ndimage.grey_opening(filtered_data, structure=kernel)
            
            # Calculate height differences
            height_diff = filtered_data - opened
            
            # Apply height threshold
            preserve_mask = height_diff <= height_threshold
            filtered_data[~preserve_mask] = opened[~preserve_mask]
        
        # Restore NaN values where they originally were
        if np.any(nan_mask):
            filtered_data[nan_mask] = np.nan
            
        # Apply slope-based filtering
        if slope_threshold > 0:
            filtered_data = RasterFilters._apply_slope_filter(filtered_data, slope_threshold)
            
        # Apply preservation intensity
        preserve_factor = {'low': 0.3, 'medium': 0.5, 'high': 0.7}[preserve_intensity]
        diff = data - filtered_data
        filtered_data = data - diff * (1 - preserve_factor)
        
        return filtered_data
    
    @staticmethod
    def terrain_filter_selective(data, selection_mask, **kwargs):
        """Apply terrain_filter only to selected areas."""
        result = data.copy()
        if not np.any(selection_mask):
            return result
            
        # Apply terrain filter to entire data for context, but only update selected cells
        filtered_data = RasterFilters.terrain_filter(data, **kwargs)
        
        # Ensure we don't overwrite with NaN values
        valid_mask = ~np.isnan(filtered_data) & ~np.isinf(filtered_data)
        update_mask = selection_mask & valid_mask
        result[update_mask] = filtered_data[update_mask]
        return result
    
    @staticmethod
    def _apply_slope_filter(data, slope_threshold):
        """Apply slope-based filtering."""
        # Calculate gradients
        gy, gx = np.gradient(data)
        slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
        
        # Apply median filter to areas with high slope
        high_slope_mask = slope > slope_threshold
        filtered_data = data.copy()
        filtered_data[high_slope_mask] = ndimage.median_filter(data, size=3)[high_slope_mask]
        
        return filtered_data
    
    @staticmethod
    def outlier_filter(data, method='statistical', threshold=2.5, window_size=7, replacement='median'):
        """
        Filter outliers from raster data.
        
        Args:
            data (numpy.ndarray): Input raster data
            method (str): Detection method ('statistical', 'mad', 'iqr')
            threshold (float): Threshold for outlier detection
            window_size (int): Window size for local statistics
            replacement (str): Replacement method ('median', 'mean', 'interpolate', 'nodata')
            
        Returns:
            numpy.ndarray: Filtered data
        """
        filtered_data = data.copy()
        
        if method == 'statistical':
            outliers = RasterFilters._detect_statistical_outliers(data, threshold, window_size)
        elif method == 'mad':
            outliers = RasterFilters._detect_mad_outliers(data, threshold, window_size)
        elif method == 'iqr':
            outliers = RasterFilters._detect_iqr_outliers(data, threshold, window_size)
        else:
            return filtered_data
            
        # Replace outliers
        if replacement == 'median':
            replacement_values = ndimage.median_filter(data, size=window_size)[outliers]
        elif replacement == 'mean':
            replacement_values = ndimage.uniform_filter(data, size=window_size)[outliers]
        elif replacement == 'nodata':
            replacement_values = np.nan
        elif replacement == 'interpolate':
            # Simple interpolation replacement
            replacement_values = RasterFilters._interpolate_outliers(data, outliers)
        else:
            return filtered_data
            
        filtered_data[outliers] = replacement_values
        
        return filtered_data
    
    @staticmethod
    def outlier_filter_selective(data, selection_mask, **kwargs):
        """Apply outlier_filter only to selected areas."""
        result = data.copy()
        if not np.any(selection_mask):
            return result
            
        # Apply outlier filter to entire data for context, but only update selected cells
        filtered_data = RasterFilters.outlier_filter(data, **kwargs)
        
        # Ensure we don't overwrite with NaN values
        valid_mask = ~np.isnan(filtered_data) & ~np.isinf(filtered_data)
        update_mask = selection_mask & valid_mask
        result[update_mask] = filtered_data[update_mask]
        return result
    
    @staticmethod
    def _detect_statistical_outliers(data, threshold, window_size):
        """Detect outliers using Z-score method."""
        # Calculate local statistics using moving window
        local_mean = ndimage.uniform_filter(data, size=window_size)
        local_std = ndimage.generic_filter(data, np.std, size=window_size)
        
        # Calculate z-scores
        z_scores = np.abs((data - local_mean) / (local_std + 1e-10))
        
        return z_scores > threshold
    
    @staticmethod
    def _detect_mad_outliers(data, threshold, window_size):
        """Detect outliers using Median Absolute Deviation."""
        local_median = ndimage.median_filter(data, size=window_size)
        mad = ndimage.generic_filter(data, lambda x: np.median(np.abs(x - np.median(x))), size=window_size)
        
        # Calculate modified z-scores
        modified_z_scores = 0.6745 * (data - local_median) / (mad + 1e-10)
        
        return np.abs(modified_z_scores) > threshold
    
    @staticmethod
    def _detect_iqr_outliers(data, threshold, window_size):
        """Detect outliers using Interquartile Range method."""
        def iqr_outliers(window):
            q1, q3 = np.percentile(window, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            center_value = window[len(window)//2]  # Center pixel
            return (center_value < lower_bound) or (center_value > upper_bound)
        
        outliers = ndimage.generic_filter(data, iqr_outliers, size=window_size)
        return outliers.astype(bool)
    
    @staticmethod
    def _interpolate_outliers(data, outliers):
        """Simple interpolation for outlier replacement."""
        # Use nearest neighbor interpolation
        valid_data = data.copy()
        valid_data[outliers] = np.nan
        
        # Simple nearest neighbor fill
        mask = ~np.isnan(valid_data)
        indices = np.arange(valid_data.size).reshape(valid_data.shape)
        valid_indices = indices[mask]
        valid_values = valid_data[mask]
        
        if len(valid_values) == 0:
            return np.zeros_like(data[outliers])
            
        # For simplicity, return median of valid values
        return np.full_like(data[outliers], np.median(valid_values))
    
    @staticmethod
    def interpolate_from_edges(data, selection_mask):
        """
        Interpolate selected area from edge values.
        
        Args:
            data (numpy.ndarray): Input raster data
            selection_mask (numpy.ndarray): Boolean mask of selected area
            
        Returns:
            numpy.ndarray: Data with interpolated values
        """
        result = data.copy()
        
        # Find edge pixels of selection
        from scipy import ndimage
        edge_mask = selection_mask & ~ndimage.binary_erosion(selection_mask)
        
        if not np.any(edge_mask):
            return result
            
        # Get edge coordinates and values
        edge_coords = np.column_stack(np.where(edge_mask))
        edge_values = data[edge_mask]
        
        # Get coordinates to interpolate
        interp_coords = np.column_stack(np.where(selection_mask))
        
        # Perform TIN-based interpolation
        try:
            from scipy.interpolate import griddata
            interpolated = griddata(edge_coords, edge_values, interp_coords, method='linear')
            
            # Fill any remaining NaN with nearest neighbor
            nan_mask = np.isnan(interpolated)
            if np.any(nan_mask):
                interpolated[nan_mask] = griddata(edge_coords, edge_values, 
                                                interp_coords[nan_mask], method='nearest')
            
            # Apply interpolated values
            rows, cols = interp_coords.T
            result[rows, cols] = interpolated
            
        except ImportError:
            QgsMessageLog.logMessage("SciPy not available for edge interpolation", "Serval", Qgis.Warning)
            
        return result
    
    @staticmethod
    def average_filter(data, kernel_size=3, boundary='reflect', weights='uniform'):
        """
        Apply average filter with various options.
        
        Args:
            data (numpy.ndarray): Input data
            kernel_size (int): Size of averaging kernel
            boundary (str): Boundary handling ('reflect', 'constant', 'nearest')
            weights (str): Weight type ('uniform', 'gaussian', 'distance')
            
        Returns:
            numpy.ndarray: Filtered data
        """
        if weights == 'uniform':
            # Simple uniform averaging
            return ndimage.uniform_filter(data, size=kernel_size, mode=boundary)
        elif weights == 'gaussian':
            # Gaussian weighted averaging
            sigma = kernel_size / 3.0
            return ndimage.gaussian_filter(data, sigma=sigma, mode=boundary)
        elif weights == 'distance':
            # Distance-based weights
            kernel = RasterFilters._create_distance_kernel(kernel_size)
            return ndimage.convolve(data, kernel, mode=boundary)
        else:
            return ndimage.uniform_filter(data, size=kernel_size, mode=boundary)
    
    @staticmethod
    def _create_distance_kernel(size):
        """Create distance-based weighting kernel."""
        center = size // 2
        y, x = np.ogrid[-center:center+1, -center:center+1]
        distances = np.sqrt(x*x + y*y)
        
        # Avoid division by zero at center
        distances[center, center] = 1e-10
        
        # Create inverse distance weights
        kernel = 1.0 / distances
        kernel /= np.sum(kernel)  # Normalize
        
        return kernel
    
    @staticmethod
    def average_filter_selective(data, selection_mask, **kwargs):
        """Apply average_filter only to selected areas."""
        result = data.copy()
        if not np.any(selection_mask):
            return result
            
        # Apply average filter to entire data for context, but only update selected cells
        filtered_data = RasterFilters.average_filter(data, **kwargs)
        
        # Ensure we don't overwrite with NaN values
        valid_mask = ~np.isnan(filtered_data) & ~np.isinf(filtered_data)
        update_mask = selection_mask & valid_mask
        result[update_mask] = filtered_data[update_mask]
        return result
    
    @staticmethod
    def constrained_filter(data, max_change=1.0, adaptive_threshold=True, preserve_steep=True):
        """
        Apply constrained averaging filter.
        
        Args:
            data (numpy.ndarray): Input data
            max_change (float): Maximum allowed change per pixel
            adaptive_threshold (bool): Use adaptive threshold based on local slope
            preserve_steep (bool): Preserve steep areas
            
        Returns:
            numpy.ndarray: Filtered data
        """
        # Apply basic average filter
        filtered = ndimage.uniform_filter(data, size=3)
        
        # Calculate changes
        changes = filtered - data
        
        if adaptive_threshold:
            # Calculate local slope
            gy, gx = np.gradient(data)
            slope = np.sqrt(gx**2 + gy**2)
            
            # Adaptive threshold based on slope
            adaptive_max = max_change * (1 + slope / np.max(slope))
            constraint_mask = np.abs(changes) > adaptive_max
        else:
            constraint_mask = np.abs(changes) > max_change
            
        # Apply constraints
        constrained_changes = changes.copy()
        constrained_changes[constraint_mask] = np.sign(changes[constraint_mask]) * max_change
        
        if preserve_steep:
            # Preserve steep areas (high slope)
            if adaptive_threshold:
                steep_mask = slope > np.percentile(slope, 90)
                constrained_changes[steep_mask] *= 0.3  # Reduce changes in steep areas
        
        return data + constrained_changes
    
    @staticmethod
    def median_filter(data, kernel_size=3, shape='rectangle', iterations=1):
        """
        Apply median filter with various kernel shapes.
        
        Args:
            data (numpy.ndarray): Input data
            kernel_size (int): Size of median kernel
            shape (str): Kernel shape ('rectangle', 'circle', 'cross')
            iterations (int): Number of iterations
            
        Returns:
            numpy.ndarray: Filtered data
        """
        if shape == 'rectangle':
            footprint = np.ones((kernel_size, kernel_size))
        elif shape == 'circle':
            footprint = RasterFilters._create_circular_kernel(kernel_size)
        elif shape == 'cross':
            footprint = RasterFilters._create_cross_kernel(kernel_size)
        else:
            footprint = np.ones((kernel_size, kernel_size))
            
        result = data.copy()
        for _ in range(iterations):
            result = ndimage.median_filter(result, footprint=footprint)
            
        return result
    
    @staticmethod
    def median_filter_selective(data, selection_mask, **kwargs):
        """Apply median_filter only to selected areas."""
        result = data.copy()
        if not np.any(selection_mask):
            return result
            
        # Apply median filter to entire data for context, but only update selected cells
        filtered_data = RasterFilters.median_filter(data, **kwargs)
        
        # Ensure we don't overwrite with NaN values
        valid_mask = ~np.isnan(filtered_data) & ~np.isinf(filtered_data)
        update_mask = selection_mask & valid_mask
        result[update_mask] = filtered_data[update_mask]
        return result
    
    @staticmethod
    def _create_circular_kernel(size):
        """Create circular kernel."""
        center = size // 2
        y, x = np.ogrid[-center:center+1, -center:center+1]
        mask = x*x + y*y <= center*center
        return mask.astype(int)
    
    @staticmethod
    def _create_cross_kernel(size):
        """Create cross-shaped kernel."""
        kernel = np.zeros((size, size))
        center = size // 2
        kernel[center, :] = 1  # Horizontal line
        kernel[:, center] = 1  # Vertical line
        return kernel
    
    @staticmethod
    def _extract_boundary_vertices(data, selection_mask):
        """Extract corner/vertex points from selection boundary."""
        if not np.any(selection_mask):
            return []
            
        # Find the bounding box of the selection
        rows, cols = np.where(selection_mask)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        
        # Extract corner vertices with their values
        vertices = []
        corners = [
            (min_row, min_col),  # Top-left
            (min_row, max_col),  # Top-right
            (max_row, min_col),  # Bottom-left
            (max_row, max_col),  # Bottom-right
        ]
        
        for row, col in corners:
            if row < data.shape[0] and col < data.shape[1]:
                value = data[row, col]
                if not np.isnan(value):
                    vertices.append((col, row, value))  # (x, y, z) format
        
        # If we have less than 3 vertices, add edge midpoints
        if len(vertices) < 3:
            edge_points = [
                (min_row, (min_col + max_col) // 2),  # Top middle
                (max_row, (min_col + max_col) // 2),  # Bottom middle
                ((min_row + max_row) // 2, min_col),  # Left middle
                ((min_row + max_row) // 2, max_col),  # Right middle
            ]
            
            for row, col in edge_points:
                if row < data.shape[0] and col < data.shape[1]:
                    value = data[row, col]
                    if not np.isnan(value):
                        vertices.append((col, row, value))
        
        return vertices[:8]  # Limit to 8 points maximum
    
    @staticmethod
    def interpolate_from_vertices(data, selection_mask, control_points=None, method='TIN'):
        """
        Interpolate selected area from vertex values (corners/edges of selection).
        When no control_points are provided, uses the vertices of the selection boundary.
        
        Args:
            data (numpy.ndarray): Input raster data
            selection_mask (numpy.ndarray): Boolean mask of selected area
            control_points (list, optional): List of (x, y, z) control point tuples
            method (str): Interpolation method ('TIN', 'Spline', 'IDW')
            
        Returns:
            numpy.ndarray: Data with interpolated values
        """
        # If no control points provided, use selection boundary vertices
        if not control_points:
            control_points = RasterFilters._extract_boundary_vertices(data, selection_mask)
        
        if not control_points:
            return data.copy()
            
        result = data.copy()
        
        # Get selected area coordinates
        selected_rows, selected_cols = np.where(selection_mask)
        if len(selected_rows) == 0:
            return result
            
        # Create coordinate grids for selected area
        min_row, max_row = selected_rows.min(), selected_rows.max()
        min_col, max_col = selected_cols.min(), selected_cols.max()
        
        # Create meshgrid for interpolation
        rows = np.arange(min_row, max_row + 1)
        cols = np.arange(min_col, max_col + 1)
        grid_rows, grid_cols = np.meshgrid(rows, cols, indexing='ij')
        
        try:
            # Import interpolation functions
            from .vertex_interpolation_tool import VertexInterpolator
            
            # Control points are already in pixel coordinates (col, row, value)
            relative_points = []
            for x, y, z in control_points:
                # x=col, y=row in pixel coordinates
                if min_row <= y <= max_row and min_col <= x <= max_col:
                    relative_points.append((x - min_col, y - min_row, z))
            
            if len(relative_points) < 1:
                return result
                
            # Perform interpolation
            if method == 'TIN' and len(relative_points) >= 3:
                interpolated = VertexInterpolator.interpolate_tin(
                    relative_points, 
                    grid_cols - min_col, 
                    grid_rows - min_row
                )
            elif method == 'Spline' and len(relative_points) >= 3:
                interpolated = VertexInterpolator.interpolate_spline(
                    relative_points,
                    grid_cols - min_col,
                    grid_rows - min_row
                )
            elif method == 'IDW':
                interpolated = VertexInterpolator.interpolate_idw(
                    relative_points,
                    grid_cols - min_col,
                    grid_rows - min_row
                )
            else:
                # Fallback to simple average if not enough points
                avg_value = np.mean([z for _, _, z in relative_points])
                interpolated = np.full_like(grid_rows, avg_value, dtype=float)
                
            # Apply interpolated values only to selected cells
            for i, row in enumerate(rows):
                for j, col in enumerate(cols):
                    if selection_mask[row, col]:
                        result[row, col] = interpolated[i, j]
                        
        except ImportError:
            QgsMessageLog.logMessage("Vertex interpolation tool not available", "Serval", Qgis.Warning)
            
        return result