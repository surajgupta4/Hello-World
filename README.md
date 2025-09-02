# Hello-World: Bolometer Geometry with Ray Transfer

This repository demonstrates bolometer geometry and ray transfer calculations commonly used in plasma diagnostics and fusion research. The implementation is based on concepts from the CHERAB plasma spectroscopy library.

## Overview

Bolometry is the measurement of radiant energy (power) emitted by plasma. A bolometer detector measures the total power incident on its surface by integrating radiation along sight lines through the plasma volume. This is accomplished through:

1. **Geometry Matrix**: Represents the geometric relationship between plasma volume elements (voxels) and the detector
2. **Ray Transfer**: Calculation of how emission from each plasma voxel contributes to the detector signal
3. **Forward Modeling**: Prediction of detector signals from known emission profiles

## Features

- **BolometerGeometry Class**: Models the geometric configuration of detector, aperture, and sight lines
- **RayTransfer Class**: Calculates geometry matrices and forward models detector signals
- **Visualization Tools**: Creates 3D plots and cross-sections of the bolometer geometry
- **Demo Functions**: Complete working examples with realistic plasma emission profiles

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from bolometer_geometry import BolometerGeometry, RayTransfer, create_demo_plasma_grid
import numpy as np

# Create bolometer geometry
detector_pos = (3.0, 0.0, 0.0)
detector_normal = (-1.0, 0.0, 0.0)
aperture_pos = (2.5, 0.0, 0.0)

bolometer = BolometerGeometry(detector_pos, detector_normal, aperture_pos)

# Create plasma grid and ray transfer
plasma_grid = create_demo_plasma_grid(5, 5, 5, 2.0)
ray_transfer = RayTransfer(bolometer)

# Calculate geometry matrix
transfer_matrix = ray_transfer.calculate_ray_transfer_matrix(plasma_grid)

# Create emission profile (Gaussian)
distances = np.linalg.norm(plasma_grid, axis=1)
emission_profile = np.exp(-(distances**2) / (0.5**2))

# Calculate detector signal
signal = ray_transfer.forward_model(emission_profile, transfer_matrix)
print(f"Detector signal: {signal:.2e}")
```

### Running the Demo

```bash
# Run basic demo
python bolometer_geometry.py

# Create visualization (saves PNG file)
python visualization.py
```

## Key Concepts

### Geometry Matrix
The geometry matrix `G` relates the emission profile `E` to the detector signal `S`:
```
S = G × E
```

Each element `G[i,j]` represents the contribution of voxel `j` to detector `i`, accounting for:
- Solid angle subtended by the aperture
- Distance from voxel to detector
- Voxel volume

### Ray Transfer
Ray transfer calculations model how photons emitted from plasma voxels reach the detector through the optical system. The implementation includes:
- Line-of-sight calculations
- Geometric efficiency factors
- Distance-dependent attenuation

### Applications
This type of modeling is essential for:
- Plasma tomography reconstruction
- Radiation measurement interpretation
- Diagnostic system design
- Fusion plasma monitoring

## Files

- `bolometer_geometry.py`: Core implementation of bolometer geometry and ray transfer
- `visualization.py`: Plotting and visualization tools  
- `requirements.txt`: Python dependencies
- `bolometer_geometry_visualization.png`: Generated visualization plot

## References

This implementation demonstrates concepts similar to those used in:
- CHERAB: Python library for plasma spectroscopy modeling
- Bolometry in fusion plasmas and plasma diagnostics
- Tomographic reconstruction techniques

## Contributing

This is a demonstration repository showing basic bolometry concepts. For production use, consider more sophisticated libraries like CHERAB.
