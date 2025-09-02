"""
Simple tests for bolometer geometry implementation.
"""

import numpy as np
from bolometer_geometry import BolometerGeometry, RayTransfer, create_demo_plasma_grid


def test_bolometer_geometry():
    """Test basic bolometer geometry functionality."""
    print("Testing BolometerGeometry...")
    
    # Create bolometer
    detector_pos = (1.0, 0.0, 0.0)
    detector_normal = (-1.0, 0.0, 0.0)
    aperture_pos = (0.5, 0.0, 0.0)
    
    bolometer = BolometerGeometry(detector_pos, detector_normal, aperture_pos)
    
    # Test sight line calculation
    sight_line = bolometer.calculate_sight_line(10)
    assert sight_line.shape == (10, 3), f"Expected (10, 3), got {sight_line.shape}"
    
    # Test geometry matrix element calculation
    voxel_center = (0.0, 0.0, 0.0)
    geometry_factor = bolometer.geometry_matrix_element(voxel_center)
    assert geometry_factor > 0, "Geometry factor should be positive"
    
    print("✓ BolometerGeometry tests passed")


def test_ray_transfer():
    """Test ray transfer calculations."""
    print("Testing RayTransfer...")
    
    # Create simple setup
    bolometer = BolometerGeometry((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.5, 0.0, 0.0))
    ray_transfer = RayTransfer(bolometer)
    
    # Create small plasma grid
    plasma_grid = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0]
    ])
    
    # Test transfer matrix calculation
    transfer_matrix = ray_transfer.calculate_ray_transfer_matrix(plasma_grid)
    assert transfer_matrix.shape == (1, 3), f"Expected (1, 3), got {transfer_matrix.shape}"
    assert np.all(transfer_matrix >= 0), "All geometry factors should be non-negative"
    
    # Test forward model
    emission_profile = np.array([1.0, 0.5, 0.5])
    signal = ray_transfer.forward_model(emission_profile, transfer_matrix)
    assert isinstance(signal, (int, float, np.number)), "Signal should be a scalar"
    assert signal >= 0, "Signal should be non-negative"
    
    print("✓ RayTransfer tests passed")


def test_plasma_grid():
    """Test plasma grid creation."""
    print("Testing plasma grid creation...")
    
    grid = create_demo_plasma_grid(3, 3, 3, 1.0)
    expected_size = 3 * 3 * 3
    assert grid.shape == (expected_size, 3), f"Expected ({expected_size}, 3), got {grid.shape}"
    
    # Check that grid spans the expected range
    assert grid[:, 0].min() >= -0.6, "Grid should be within expected X range"
    assert grid[:, 0].max() <= 0.6, "Grid should be within expected X range"
    
    print("✓ Plasma grid tests passed")


def test_numerical_consistency():
    """Test that calculations are numerically consistent."""
    print("Testing numerical consistency...")
    
    # Create reproducible setup
    np.random.seed(42)
    bolometer = BolometerGeometry((2.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (1.5, 0.0, 0.0))
    ray_transfer = RayTransfer(bolometer)
    
    plasma_grid = create_demo_plasma_grid(3, 3, 3, 1.0)
    transfer_matrix = ray_transfer.calculate_ray_transfer_matrix(plasma_grid)
    
    # Test with uniform emission
    uniform_emission = np.ones(plasma_grid.shape[0])
    signal_uniform = ray_transfer.forward_model(uniform_emission, transfer_matrix)
    
    # Signal should equal sum of geometry factors for uniform emission
    expected_signal = transfer_matrix.sum()
    assert np.isclose(signal_uniform, expected_signal, rtol=1e-10), \
           f"Expected {expected_signal}, got {signal_uniform}"
    
    # Test with zero emission
    zero_emission = np.zeros(plasma_grid.shape[0])
    signal_zero = ray_transfer.forward_model(zero_emission, transfer_matrix)
    assert np.isclose(signal_zero, 0.0, atol=1e-15), \
           f"Expected 0.0 for zero emission, got {signal_zero}"
    
    print("✓ Numerical consistency tests passed")


def run_all_tests():
    """Run all tests."""
    print("Running bolometer geometry tests...")
    print("=" * 40)
    
    test_bolometer_geometry()
    test_ray_transfer() 
    test_plasma_grid()
    test_numerical_consistency()
    
    print("=" * 40)
    print("All tests passed! ✓")


if __name__ == "__main__":
    run_all_tests()