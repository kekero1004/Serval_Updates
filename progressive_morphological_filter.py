"""
/***************************************************************************
 Serval Progressive Morphological Filter
 
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

import numpy as np
import math
from qgis.core import QgsMessageLog, Qgis

try:
    from scipy import ndimage
    from scipy.ndimage import binary_erosion, binary_dilation, grey_opening, grey_closing
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class ProgressiveMorphologicalFilter:
    """
    Progressive Morphological Filter (PMF) implementation matching ArcGIS Pro 3.4.2
    Based on Zhang et al. and Whitman methodology for terrain filtering.
    """
    
    @staticmethod
    def apply_pmf_filter(data, selection_mask, parameters):
        """
        Apply Progressive Morphological Filter to extract terrain from DSM.
        
        Args:
            data (numpy.ndarray): Input DSM data
            selection_mask (numpy.ndarray): Boolean mask of selected area
            parameters (dict): Filter parameters
            
        Returns:
            numpy.ndarray: Filtered terrain data (DTM)
        """
        if not SCIPY_AVAILABLE:
            QgsMessageLog.logMessage("SciPy required for PMF", "Serval", Qgis.Warning)
            return data.copy()
            
        result = data.copy()
        
        # Extract parameters
        initial_window = parameters.get('initial_window_size', 8)
        increase_constant = parameters.get('increase_constant', 2)
        max_window = parameters.get('max_window_size', 32)
        elevation_threshold = parameters.get('elevation_threshold', 2.0)
        slope_threshold = parameters.get('slope_threshold', 15.0)
        iterations = parameters.get('iterations', 5)
        enable_opening = parameters.get('enable_opening', True)
        enable_closing = parameters.get('enable_closing', False)
        structure_element = parameters.get('structure_element', 'rectangle')
        preserve_intensity = parameters.get('preserve_intensity', 'medium')
        
        # Apply filter only to selected area
        if np.any(selection_mask):
            # Extract selected data
            selected_data = data.copy()
            
            # Handle NaN values temporarily
            nan_mask = np.isnan(selected_data)
            if np.any(nan_mask):
                # Fill NaN with interpolated values for processing
                selected_data = ProgressiveMorphologicalFilter._fill_nan_values(selected_data)
            
            # Apply Progressive Morphological Filter
            filtered_data = ProgressiveMorphologicalFilter._pmf_core_algorithm(
                selected_data, initial_window, increase_constant, max_window, 
                elevation_threshold, iterations, enable_opening, enable_closing, 
                structure_element)
            
            # Apply slope-based filtering
            if slope_threshold > 0:
                filtered_data = ProgressiveMorphologicalFilter._apply_slope_filter(
                    selected_data, filtered_data, slope_threshold)
            
            # Apply terrain preservation
            filtered_data = ProgressiveMorphologicalFilter._apply_terrain_preservation(
                selected_data, filtered_data, preserve_intensity)
            
            # Apply edge enhancement if enabled
            if parameters.get('edge_enhancement', False):
                filtered_data = ProgressiveMorphologicalFilter._enhance_edges(
                    selected_data, filtered_data)
            
            # Apply noise reduction if enabled
            if parameters.get('noise_reduction', False):
                filtered_data = ProgressiveMorphologicalFilter._reduce_noise(filtered_data)
            
            # Restore NaN values where they originally were
            if np.any(nan_mask):
                filtered_data[nan_mask] = np.nan
            
            # Update only selected cells
            result[selection_mask] = filtered_data[selection_mask]
        
        return result
    
    @staticmethod
    def _pmf_core_algorithm(data, initial_window, increase_constant, max_window, 
                           elevation_threshold, iterations, enable_opening, 
                           enable_closing, structure_element):
        """
        Core PMF algorithm implementing progressive morphological operations.
        """
        # Initialize with original data
        filtered_data = data.copy()
        
        # Progressive window sizes: wn = initial_window + c * k (where k is iteration)
        for iteration in range(iterations):
            window_size = min(initial_window + increase_constant * iteration, max_window)
            
            # Create structure element
            structure_elem = ProgressiveMorphologicalFilter._create_structure_element(
                window_size, structure_element)
            
            # Apply morphological operations
            if enable_opening:
                # Opening: Erosion followed by Dilation
                eroded = ndimage.grey_erosion(filtered_data, structure=structure_elem)
                opened = ndimage.grey_dilation(eroded, structure=structure_elem)
                
                # Apply elevation difference threshold
                height_diff = filtered_data - opened
                terrain_mask = height_diff <= elevation_threshold
                
                # Update terrain points
                filtered_data[terrain_mask] = opened[terrain_mask]
            
            if enable_closing:
                # Closing: Dilation followed by Erosion  
                dilated = ndimage.grey_dilation(filtered_data, structure=structure_elem)
                closed = ndimage.grey_erosion(dilated, structure=structure_elem)
                
                # Apply for gap filling and smoothing
                gap_mask = closed > filtered_data
                filtered_data[gap_mask] = closed[gap_mask]
        
        return filtered_data
    
    @staticmethod
    def _create_structure_element(size, shape):
        """Create morphological structure element based on shape and size."""
        if shape == 'rectangle':
            return np.ones((size, size))
        elif shape == 'ellipse':
            y, x = np.ogrid[-size//2:size//2+1, -size//2:size//2+1]
            mask = x*x + y*y <= (size//2)**2
            return mask.astype(np.uint8)
        elif shape == 'cross':
            element = np.zeros((size, size))
            center = size // 2
            element[center, :] = 1  # Horizontal line
            element[:, center] = 1  # Vertical line
            return element
        elif shape == 'diamond':
            element = np.zeros((size, size))
            center = size // 2
            for i in range(size):
                for j in range(size):
                    if abs(i - center) + abs(j - center) <= center:
                        element[i, j] = 1
            return element
        else:
            return np.ones((size, size))
    
    @staticmethod
    def _apply_slope_filter(original_data, filtered_data, slope_threshold):
        """
        Apply slope-based filtering to preserve steep terrain features.
        Based on elevation difference between adjacent points.
        """
        # Calculate gradients
        gy, gx = np.gradient(original_data)
        slope_degrees = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
        
        # Identify steep areas that should be preserved
        steep_mask = slope_degrees > slope_threshold
        
        # For steep areas, use more conservative filtering
        conservative_filtered = filtered_data.copy()
        
        # Apply median filter to steep areas for noise reduction
        steep_areas = ndimage.median_filter(original_data, size=3)
        conservative_filtered[steep_mask] = steep_areas[steep_mask]
        
        return conservative_filtered
    
    @staticmethod
    def _apply_terrain_preservation(original_data, filtered_data, intensity):
        """
        Apply terrain preservation based on intensity setting.
        """
        preserve_factors = {
            'low': 0.2,
            'medium': 0.5, 
            'high': 0.7,
            'very high': 0.9
        }
        
        preserve_factor = preserve_factors.get(intensity, 0.5)
        
        # Blend original and filtered data
        difference = original_data - filtered_data
        blended_data = original_data - difference * (1 - preserve_factor)
        
        return blended_data
    
    @staticmethod
    def _enhance_edges(original_data, filtered_data):
        """
        Enhance terrain edges using gradient-based methods.
        """
        # Calculate gradients
        gy_orig, gx_orig = np.gradient(original_data)
        gy_filt, gx_filt = np.gradient(filtered_data)
        
        # Calculate gradient magnitudes
        grad_orig = np.sqrt(gx_orig**2 + gy_orig**2)
        grad_filt = np.sqrt(gx_filt**2 + gy_filt**2)
        
        # Identify edge areas (high gradient)
        edge_threshold = np.percentile(grad_orig, 75)  # Top 25% of gradients
        edge_mask = grad_orig > edge_threshold
        
        # Preserve original gradients in edge areas
        enhanced_data = filtered_data.copy()
        enhanced_data[edge_mask] = original_data[edge_mask]
        
        return enhanced_data
    
    @staticmethod
    def _reduce_noise(data):
        """
        Apply noise reduction using adaptive filtering.
        """
        # Use bilateral filter-like approach for noise reduction while preserving edges
        # Since scipy doesn't have bilateral filter, use combination of median and gaussian
        
        # Median filter for impulse noise
        median_filtered = ndimage.median_filter(data, size=3)
        
        # Gaussian filter for gaussian noise (with small sigma to preserve edges)
        gaussian_filtered = ndimage.gaussian_filter(data, sigma=0.5)
        
        # Adaptive combination based on local variance
        local_variance = ndimage.uniform_filter(data**2, size=3) - ndimage.uniform_filter(data, size=3)**2
        variance_threshold = np.percentile(local_variance, 50)
        
        # Use median filter in high variance areas, gaussian in low variance areas
        noise_reduced = np.where(local_variance > variance_threshold, 
                                median_filtered, gaussian_filtered)
        
        return noise_reduced
    
    @staticmethod
    def _fill_nan_values(data):
        """
        Fill NaN values with interpolated values for processing.
        """
        if not np.any(np.isnan(data)):
            return data
            
        filled_data = data.copy()
        nan_mask = np.isnan(data)
        
        # Use nearest neighbor interpolation to fill NaN values
        if np.any(~nan_mask):
            # Simple approach: replace NaN with mean of valid neighbors
            for _ in range(3):  # Multiple iterations for better filling
                for i in range(1, data.shape[0]-1):
                    for j in range(1, data.shape[1]-1):
                        if nan_mask[i, j]:
                            neighbors = filled_data[i-1:i+2, j-1:j+2]
                            valid_neighbors = neighbors[~np.isnan(neighbors)]
                            if len(valid_neighbors) > 0:
                                filled_data[i, j] = np.mean(valid_neighbors)
                                nan_mask[i, j] = False
        
        return filled_data
    
    @staticmethod
    def apply_simple_morphological_filter(data, selection_mask, parameters):
        """
        Apply simple morphological filter (non-progressive).
        """
        if not SCIPY_AVAILABLE:
            return data.copy()
            
        result = data.copy()
        window_size = parameters.get('initial_window_size', 8)
        structure_element = parameters.get('structure_element', 'rectangle')
        
        if np.any(selection_mask):
            selected_data = data.copy()
            
            # Handle NaN values
            nan_mask = np.isnan(selected_data)
            if np.any(nan_mask):
                selected_data = ProgressiveMorphologicalFilter._fill_nan_values(selected_data)
            
            # Create structure element
            structure_elem = ProgressiveMorphologicalFilter._create_structure_element(
                window_size, structure_element)
            
            # Apply single opening operation
            filtered_data = ndimage.grey_opening(selected_data, structure=structure_elem)
            
            # Restore NaN values
            if np.any(nan_mask):
                filtered_data[nan_mask] = np.nan
            
            # Update only selected cells
            result[selection_mask] = filtered_data[selection_mask]
        
        return result
    
    @staticmethod
    def apply_adaptive_filter(data, selection_mask, parameters):
        """
        Apply adaptive morphological filter that adjusts window size based on local terrain.
        """
        if not SCIPY_AVAILABLE:
            return data.copy()
            
        result = data.copy()
        
        if np.any(selection_mask):
            selected_data = data.copy()
            
            # Handle NaN values
            nan_mask = np.isnan(selected_data)
            if np.any(nan_mask):
                selected_data = ProgressiveMorphologicalFilter._fill_nan_values(selected_data)
            
            # Calculate local terrain roughness
            gy, gx = np.gradient(selected_data)
            roughness = np.sqrt(gx**2 + gy**2)
            
            # Adaptive window sizing based on roughness
            min_window = parameters.get('initial_window_size', 8) // 2
            max_window = parameters.get('max_window_size', 32)
            
            # Normalize roughness to [0, 1]
            roughness_norm = (roughness - np.nanmin(roughness)) / (np.nanmax(roughness) - np.nanmin(roughness))
            
            # Calculate adaptive window sizes
            window_sizes = (min_window + roughness_norm * (max_window - min_window)).astype(int)
            
            # Apply adaptive filtering
            filtered_data = selected_data.copy()
            
            # Process in blocks with similar window sizes
            unique_sizes = np.unique(window_sizes)
            for window_size in unique_sizes:
                if window_size < 3:
                    continue
                    
                mask = window_sizes == window_size
                if np.any(mask):
                    structure_elem = ProgressiveMorphologicalFilter._create_structure_element(
                        window_size, parameters.get('structure_element', 'rectangle'))
                    
                    # Apply opening to areas with this window size
                    temp_data = selected_data.copy()
                    temp_filtered = ndimage.grey_opening(temp_data, structure=structure_elem)
                    filtered_data[mask] = temp_filtered[mask]
            
            # Restore NaN values
            if np.any(nan_mask):
                filtered_data[nan_mask] = np.nan
            
            # Update only selected cells
            result[selection_mask] = filtered_data[selection_mask]
        
        return result