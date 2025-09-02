"""
Bolometer Geometry Matrix with Ray Transfer

This module demonstrates basic concepts related to bolometer geometry
and ray transfer calculations commonly used in plasma diagnostics.

Based on concepts from CHERAB plasma spectroscopy library.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class BolometerGeometry:
    """
    Simple bolometer geometry class for demonstrating ray transfer concepts.
    
    A bolometer measures the total radiant power from plasma by integrating
    along sight lines through the plasma volume.
    """
    
    def __init__(self, detector_position: Tuple[float, float, float], 
                 detector_normal: Tuple[float, float, float],
                 aperture_position: Tuple[float, float, float],
                 aperture_size: float = 0.01):
        """
        Initialize bolometer geometry.
        
        Args:
            detector_position: (x, y, z) position of detector
            detector_normal: (nx, ny, nz) normal vector of detector surface
            aperture_position: (x, y, z) position of aperture/pinhole
            aperture_size: size of the aperture in meters
        """
        self.detector_position = np.array(detector_position)
        self.detector_normal = np.array(detector_normal)
        self.detector_normal = self.detector_normal / np.linalg.norm(self.detector_normal)
        self.aperture_position = np.array(aperture_position)
        self.aperture_size = aperture_size
        
    def calculate_sight_line(self, num_points: int = 100) -> np.ndarray:
        """
        Calculate points along the sight line from aperture to detector.
        
        Args:
            num_points: Number of points along the sight line
            
        Returns:
            Array of shape (num_points, 3) containing sight line points
        """
        direction = self.detector_position - self.aperture_position
        direction = direction / np.linalg.norm(direction)
        
        # Create points along the sight line
        distances = np.linspace(0, np.linalg.norm(self.detector_position - self.aperture_position), num_points)
        sight_line = self.aperture_position[:, np.newaxis] + direction[:, np.newaxis] * distances
        
        return sight_line.T
    
    def geometry_matrix_element(self, voxel_center: Tuple[float, float, float], 
                               voxel_size: float = 0.1) -> float:
        """
        Calculate geometry matrix element for a plasma volume voxel.
        
        This represents the contribution of emission from a voxel to the
        detector signal, accounting for geometric factors.
        
        Args:
            voxel_center: (x, y, z) center of the plasma voxel
            voxel_size: size of the voxel in meters
            
        Returns:
            Geometry matrix element (dimensionless)
        """
        voxel_center = np.array(voxel_center)
        
        # Vector from voxel to aperture
        to_aperture = self.aperture_position - voxel_center
        distance_to_aperture = np.linalg.norm(to_aperture)
        
        # Vector from voxel to detector
        to_detector = self.detector_position - voxel_center
        distance_to_detector = np.linalg.norm(to_detector)
        
        # Solid angle subtended by aperture as seen from voxel
        solid_angle_aperture = np.pi * (self.aperture_size / 2)**2 / distance_to_aperture**2
        
        # Geometric factor including distance to detector
        geometry_factor = solid_angle_aperture * voxel_size**3 / distance_to_detector**2
        
        return geometry_factor


class RayTransfer:
    """
    Demonstrates ray transfer calculations for bolometry.
    """
    
    def __init__(self, bolometer: BolometerGeometry):
        self.bolometer = bolometer
        
    def calculate_ray_transfer_matrix(self, plasma_grid: np.ndarray) -> np.ndarray:
        """
        Calculate the ray transfer matrix for a plasma grid.
        
        Args:
            plasma_grid: Array of shape (N, 3) containing voxel centers
            
        Returns:
            Ray transfer matrix of shape (1, N) for single bolometer
        """
        num_voxels = plasma_grid.shape[0]
        transfer_matrix = np.zeros((1, num_voxels))
        
        for i, voxel_center in enumerate(plasma_grid):
            transfer_matrix[0, i] = self.bolometer.geometry_matrix_element(voxel_center)
            
        return transfer_matrix
    
    def forward_model(self, emission_profile: np.ndarray, 
                     transfer_matrix: np.ndarray) -> float:
        """
        Forward model: calculate detector signal from emission profile.
        
        Args:
            emission_profile: Array of emission values for each voxel
            transfer_matrix: Ray transfer matrix
            
        Returns:
            Predicted detector signal
        """
        return np.dot(transfer_matrix, emission_profile)[0]


def create_demo_plasma_grid(nx: int = 10, ny: int = 10, nz: int = 10,
                           extent: float = 2.0) -> np.ndarray:
    """
    Create a simple 3D grid representing plasma volume.
    
    Args:
        nx, ny, nz: Grid dimensions
        extent: Size of the grid in meters
        
    Returns:
        Array of shape (nx*ny*nz, 3) containing voxel centers
    """
    x = np.linspace(-extent/2, extent/2, nx)
    y = np.linspace(-extent/2, extent/2, ny)
    z = np.linspace(-extent/2, extent/2, nz)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return grid


def demo_bolometer_geometry():
    """
    Demonstration of bolometer geometry and ray transfer calculations.
    """
    print("Bolometer Geometry and Ray Transfer Demo")
    print("=" * 40)
    
    # Create a bolometer
    detector_pos = (3.0, 0.0, 0.0)  # 3m from origin along x-axis
    detector_normal = (-1.0, 0.0, 0.0)  # Looking towards origin
    aperture_pos = (2.5, 0.0, 0.0)  # Aperture 0.5m in front of detector
    
    bolometer = BolometerGeometry(detector_pos, detector_normal, aperture_pos)
    ray_transfer = RayTransfer(bolometer)
    
    print(f"Detector position: {detector_pos}")
    print(f"Aperture position: {aperture_pos}")
    print(f"Aperture size: {bolometer.aperture_size} m")
    
    # Create plasma grid
    plasma_grid = create_demo_plasma_grid(5, 5, 5, 2.0)
    print(f"\nPlasma grid: {plasma_grid.shape[0]} voxels")
    
    # Calculate ray transfer matrix
    transfer_matrix = ray_transfer.calculate_ray_transfer_matrix(plasma_grid)
    print(f"Transfer matrix shape: {transfer_matrix.shape}")
    print(f"Max geometry factor: {transfer_matrix.max():.2e}")
    print(f"Sum of geometry factors: {transfer_matrix.sum():.2e}")
    
    # Create a simple emission profile (Gaussian centered at origin)
    distances = np.linalg.norm(plasma_grid, axis=1)
    emission_profile = np.exp(-(distances**2) / (0.5**2))  # Gaussian with σ=0.5m
    
    # Calculate detector signal
    signal = ray_transfer.forward_model(emission_profile, transfer_matrix)
    print(f"\nSimulated detector signal: {signal:.2e}")
    
    # Calculate sight line
    sight_line = bolometer.calculate_sight_line(50)
    print(f"Sight line calculated with {sight_line.shape[0]} points")
    
    return bolometer, plasma_grid, transfer_matrix, emission_profile


if __name__ == "__main__":
    demo_bolometer_geometry()