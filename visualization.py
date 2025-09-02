"""
Visualization tools for bolometer geometry and ray transfer.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bolometer_geometry import BolometerGeometry, RayTransfer, create_demo_plasma_grid


def plot_bolometer_geometry(bolometer: BolometerGeometry, plasma_grid: np.ndarray, 
                           emission_profile: np.ndarray, transfer_matrix: np.ndarray):
    """
    Create visualization plots for bolometer geometry and ray transfer.
    
    Args:
        bolometer: BolometerGeometry instance
        plasma_grid: Array of voxel centers
        emission_profile: Emission values for each voxel
        transfer_matrix: Ray transfer matrix
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 3D plot of geometry
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Plot plasma grid points colored by emission
    scatter = ax1.scatter(plasma_grid[:, 0], plasma_grid[:, 1], plasma_grid[:, 2], 
                         c=emission_profile, cmap='hot', alpha=0.6, s=50)
    plt.colorbar(scatter, ax=ax1, label='Emission Intensity', shrink=0.8)
    
    # Plot detector and aperture
    ax1.scatter(*bolometer.detector_position, color='blue', s=200, marker='s', 
               label='Detector')
    ax1.scatter(*bolometer.aperture_position, color='green', s=100, marker='o', 
               label='Aperture')
    
    # Plot sight line
    sight_line = bolometer.calculate_sight_line(50)
    ax1.plot(sight_line[:, 0], sight_line[:, 1], sight_line[:, 2], 
            'b--', alpha=0.7, label='Sight Line')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Bolometer Geometry and Plasma Grid')
    ax1.legend()
    
    # 2D cross-section (X-Y plane)
    ax2 = fig.add_subplot(222)
    
    # Filter points near z=0
    z_mask = np.abs(plasma_grid[:, 2]) < 0.2
    if np.any(z_mask):
        ax2.scatter(plasma_grid[z_mask, 0], plasma_grid[z_mask, 1], 
                   c=emission_profile[z_mask], cmap='hot', s=100, alpha=0.8)
        
    ax2.scatter(bolometer.detector_position[0], bolometer.detector_position[1], 
               color='blue', s=200, marker='s', label='Detector')
    ax2.scatter(bolometer.aperture_position[0], bolometer.aperture_position[1], 
               color='green', s=100, marker='o', label='Aperture')
    ax2.plot([bolometer.aperture_position[0], bolometer.detector_position[0]],
            [bolometer.aperture_position[1], bolometer.detector_position[1]], 
            'b--', alpha=0.7)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Cross-section (Z ≈ 0)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Geometry matrix elements
    ax3 = fig.add_subplot(223)
    
    geometry_factors = transfer_matrix[0, :]
    distances = np.linalg.norm(plasma_grid - bolometer.aperture_position, axis=1)
    
    ax3.scatter(distances, geometry_factors, alpha=0.6, c=emission_profile, cmap='hot')
    ax3.set_xlabel('Distance from Aperture (m)')
    ax3.set_ylabel('Geometry Matrix Element')
    ax3.set_title('Geometry Matrix vs Distance')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Emission profile radial distribution
    ax4 = fig.add_subplot(224)
    
    origin_distances = np.linalg.norm(plasma_grid, axis=1)
    ax4.scatter(origin_distances, emission_profile, alpha=0.6, color='red')
    ax4.set_xlabel('Distance from Origin (m)')
    ax4.set_ylabel('Emission Intensity')
    ax4.set_title('Radial Emission Profile')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_visualization_demo():
    """
    Create and save visualization plots for the bolometer geometry demo.
    """
    from bolometer_geometry import demo_bolometer_geometry
    
    print("Creating visualization demo...")
    
    # Run the demo to get data
    bolometer, plasma_grid, transfer_matrix, emission_profile = demo_bolometer_geometry()
    
    # Create visualization
    fig = plot_bolometer_geometry(bolometer, plasma_grid, emission_profile, transfer_matrix)
    
    # Save the plot
    fig.savefig('bolometer_geometry_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'bolometer_geometry_visualization.png'")
    
    plt.show()


if __name__ == "__main__":
    create_visualization_demo()