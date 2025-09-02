#!/usr/bin/env python3
"""
Example usage of the bolometer geometry implementation.

This script demonstrates how to set up a bolometer, create a plasma grid,
calculate the geometry matrix, and predict detector signals.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for CI/headless environments
import matplotlib.pyplot as plt

from bolometer_geometry import BolometerGeometry, RayTransfer, create_demo_plasma_grid


def main():
    """Run a complete example of bolometer geometry calculations."""
    print("Bolometer Geometry Example")
    print("=" * 26)
    
    # Step 1: Define bolometer geometry
    print("\n1. Setting up bolometer geometry:")
    detector_position = (3.0, 0.0, 0.0)  # 3 meters from origin
    detector_normal = (-1.0, 0.0, 0.0)   # Looking toward origin
    aperture_position = (2.5, 0.0, 0.0)  # Aperture in front of detector
    aperture_size = 0.01  # 1 cm aperture
    
    bolometer = BolometerGeometry(
        detector_position, detector_normal, aperture_position, aperture_size
    )
    
    print(f"   Detector at: {detector_position}")
    print(f"   Aperture at: {aperture_position}")
    print(f"   Aperture size: {aperture_size} m")
    
    # Step 2: Create plasma grid
    print("\n2. Creating plasma volume grid:")
    nx, ny, nz = 6, 6, 6
    extent = 2.0  # 2m x 2m x 2m plasma volume
    plasma_grid = create_demo_plasma_grid(nx, ny, nz, extent)
    
    print(f"   Grid size: {nx}×{ny}×{nz} = {plasma_grid.shape[0]} voxels")
    print(f"   Volume extent: ±{extent/2} m in each direction")
    
    # Step 3: Calculate ray transfer matrix
    print("\n3. Calculating geometry matrix:")
    ray_transfer = RayTransfer(bolometer)
    transfer_matrix = ray_transfer.calculate_ray_transfer_matrix(plasma_grid)
    
    print(f"   Matrix shape: {transfer_matrix.shape}")
    print(f"   Max element: {transfer_matrix.max():.2e}")
    print(f"   Matrix sum: {transfer_matrix.sum():.2e}")
    
    # Step 4: Create emission profiles and calculate signals
    print("\n4. Testing different emission profiles:")
    
    # Profile 1: Gaussian centered at origin
    distances = np.linalg.norm(plasma_grid, axis=1)
    gaussian_profile = np.exp(-(distances**2) / (0.5**2))
    signal_gaussian = ray_transfer.forward_model(gaussian_profile, transfer_matrix)
    
    # Profile 2: Uniform emission
    uniform_profile = np.ones(plasma_grid.shape[0])
    signal_uniform = ray_transfer.forward_model(uniform_profile, transfer_matrix)
    
    # Profile 3: Edge-localized emission
    edge_profile = np.where(distances > 0.7, 1.0, 0.0)
    signal_edge = ray_transfer.forward_model(edge_profile, transfer_matrix)
    
    print(f"   Gaussian profile signal: {signal_gaussian:.2e}")
    print(f"   Uniform profile signal:  {signal_uniform:.2e}")
    print(f"   Edge profile signal:     {signal_edge:.2e}")
    
    # Step 5: Create a simple comparison plot
    print("\n5. Creating comparison plot:")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot emission profiles vs distance
    ax1.scatter(distances, gaussian_profile, alpha=0.7, label='Gaussian', color='red')
    ax1.scatter(distances, uniform_profile, alpha=0.7, label='Uniform', color='blue')
    ax1.scatter(distances, edge_profile, alpha=0.7, label='Edge', color='green')
    ax1.set_xlabel('Distance from Origin (m)')
    ax1.set_ylabel('Emission Intensity')
    ax1.set_title('Emission Profiles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot geometry factors vs distance from detector
    detector_distances = np.linalg.norm(plasma_grid - bolometer.detector_position, axis=1)
    geometry_factors = transfer_matrix[0, :]
    
    ax2.scatter(detector_distances, geometry_factors, alpha=0.7, color='purple')
    ax2.set_xlabel('Distance from Detector (m)')
    ax2.set_ylabel('Geometry Factor')
    ax2.set_title('Geometry Matrix Elements')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_results.png', dpi=150, bbox_inches='tight')
    print("   Plot saved as 'example_results.png'")
    
    # Step 6: Summary
    print("\n6. Summary:")
    profiles = [
        ("Gaussian (peaked at center)", signal_gaussian),
        ("Uniform (constant)", signal_uniform), 
        ("Edge-localized", signal_edge)
    ]
    
    for name, signal in profiles:
        percentage = (signal / signal_uniform) * 100 if signal_uniform > 0 else 0
        print(f"   {name:25}: {signal:.2e} ({percentage:.1f}% of uniform)")
    
    print("\n" + "="*50)
    print("Example completed successfully!")
    print("Check 'example_results.png' for visualization.")


if __name__ == "__main__":
    main()