# Multidimensional Motif Simulator - Complete Documentation

## Overview

The **Multidimensional Motif Simulator** is a comprehensive system for creating synthetic multidimensional time-series datasets with embedded motifs. This tool is designed for researchers and developers who need to test and validate motif detection algorithms on controlled datasets with known ground truth.

### Key Capabilities

- **Base Signal Generation**: Create AR(p) processes or sinusoidal signals with customizable properties
- **Motif Creation**: Place various motif types at specified locations with controlled parameters
- **Multidimensional Support**: Extend signals to multiple dimensions with independent noise levels
- **Dimension-Specific Motifs**: Place motifs on arbitrary subsets of dimensions
- **Interactive Interface**: Use Streamlit for real-time visualization and manipulation
- **Metadata Preservation**: Save complete information about all placed motifs and signal parameters
- **Motif Editing**: Modify motif parameters after placement without regenerating the entire signal

---

## Architecture Overview

The system is organized into three main components:

### 1. **Core Simulation Module** (`multidimensionnal_motifs_simulator.py`)

This module contains the core logic for signal and motif generation. It is structured as follows:

#### Classes:

- **`BaseSignalGenerator`**: Static methods for generating base signals
  - `ar_process()`: Generate AR(p) processes with configurable coefficients and noise
  - `sinusoidal_signal()`: Generate sinusoidal signals with varying amplitude and frequency

- **`MotifGenerator`**: Static methods for generating various motif shapes
  - `morlet_wavelet()`: Morlet wavelet motif
  - `sine_period()`: Single sine period
  - `triangle()`: Triangle wave
  - `square()`: Square wave
  - `exponential_decay()`: Exponential rise and decay
  - `dampened_oscillator()`: Decaying sinusoid
  - `generate_motif()`: Generic method to generate any motif type

- **`MultidimensionalSimulator`**: Main class orchestrating the simulation
  - `generate_base_signal()`: Create the base signal for all dimensions
  - `set_noise_levels()`: Configure dimension-specific noise levels
  - `add_motif()`: Add a single motif with collision detection
  - `add_multiple_motifs()`: Randomly add multiple motifs
  - `save_simulation()`: Serialize to pickle file
  - `load_simulation()`: Load from pickle file
  - `get_metadata()`: Retrieve simulation metadata

- **`MotifInstance`** (dataclass): Stores information about a single motif
  - `motif_type`: Type of motif (e.g., 'morlet', 'sine')
  - `start_index`: Where the motif begins in the signal
  - `dimensions`: List of dimensions this motif occupies
  - `length`: Duration of the motif
  - `amplitude`: Peak amplitude
  - `parameters`: Motif-specific parameters (frequency, damping, etc.)
  - `dimension_offsets`: Time offset for each dimension (in bins)

- **`SimulationMetadata`** (dataclass): Contains all information about a simulation
  - Signal properties (length, dimensions)
  - Base signal configuration
  - Noise levels per dimension
  - List of all motif instances

### 2. **Streamlit Application** (`streamlit_motif_simulator.py`)

An interactive web interface for the simulator with the following sections:

#### Sidebar: Signal Setup
- Signal length and number of dimensions
- Base signal type selection
- AR process configuration (order and coefficients)
- Sinusoidal signal parameters (frequency, amplitude variations)
- Dimension-specific noise levels
- Base signal generation button

#### Main Content: Add Motifs

**Single Motif Addition:**
- Select motif type
- Set motif length and amplitude
- Configure motif-specific parameters
- Select target dimensions
- Set dimension-specific time offsets
- Choose start index via slider
- Add motif to signal

**Multiple Motifs Addition:**
- Specify number of motifs to add
- Select which motif types to sample from
- Define ranges for length and amplitude
- Set maximum dimension offset
- Automatically place multiple non-overlapping motifs

#### Visualization Panel
- Interactive Plotly plots showing all dimensions
- Motif markers and annotations at start locations
- Color-coded motif types
- Hover information for detailed values
- Pan and zoom capabilities

#### Motif Editing
- Table view of all placed motifs
- Edit amplitude and parameters of existing motifs
- Real-time signal update without regenerating
- Subtraction and addition of modified motifs

#### Dataset Saving
- Save signal and metadata to pickle file
- Complete metadata preservation
- Easy loading for further processing

### 3. **Streamlit Interface**

The app provides a user-friendly interface organized into logical sections.

---

## Detailed Feature Description

### 1. Base Signal Generation

#### AR(p) Process

**Parameters:**
- `ar_coeffs`: List of AR coefficients (φ₁, φ₂, ..., φₚ)
- `noise_variance`: Variance of the Gaussian noise term
- `signal_length`: Total duration of the signal

**Stability Check:** The generator automatically checks that the AR process is stationary (all roots of the characteristic polynomial lie outside the unit circle) and warns if coefficients may violate this.

**Formula:**
$$x_t = \sum_{i=1}^{p} \phi_i x_{t-i} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)$$

**Special Case:** When `ar_order=0`, the generator produces pure Gaussian noise.

#### Sinusoidal Signal (EMD-like)

**Parameters:**
- `base_frequency`: Base frequency of oscillation
- `amplitude_mean`: Mean amplitude of the oscillation
- `amplitude_variation`: Standard deviation of random amplitude changes
- `frequency_variation`: Standard deviation of random frequency changes
- `signal_length`: Total duration

**Behavior:** This creates a slowly-varying sinusoid similar to an intrinsic mode function from empirical mode decomposition:
- Amplitude and frequency vary smoothly with random walk components
- Variations are scaled to be slow relative to the oscillation itself

### 2. Motif Types

#### Morlet Wavelet
- **Parameters**: `frequency`, `sigma` (temporal width)
- **Use Case**: Detecting localized oscillations
- **Formula**: $\psi(t) = e^{-t^2/(2\sigma^2)} \cos(2\pi f t)$

#### Sine Period
- **Parameters**: `frequency`
- **Use Case**: Simple periodic patterns
- **Duration**: One complete period

#### Triangle Wave
- **Parameters**: None (fixed shape)
- **Use Case**: Sharp changes and linear transitions

#### Square Wave
- **Parameters**: `duty_cycle` (fraction of period at high level)
- **Use Case**: Binary state transitions

#### Exponential Rise and Decay
- **Parameters**: `decay_rate`
- **Use Case**: Event-like patterns with rapid onset and gradual recovery
- **Formula**: 
  - Rise: $A(1 - e^{-\lambda t})$
  - Decay: $A e^{-\lambda t}$

#### Dampened Oscillator
- **Parameters**: `frequency`, `damping`
- **Use Case**: Damped oscillatory patterns
- **Formula**: $A e^{-\zeta t} \sin(2\pi f t)$

### 3. Multidimensional Support

#### Signal Structure
- Base signals are generated independently for each dimension
- Each dimension has its own noise level multiplier
- Motifs can span an arbitrary subset of dimensions

#### Dimension-Specific Properties
- Each dimension can have a different noise level
- Each dimension maintains its own AR coefficients (if AR process)
- Motifs can occupy different dimensions with different time offsets

#### Dimension Offsets
- Each motif can have a different start time in different dimensions
- Offset is specified in bins (samples)
- Useful for simulating propagation delays or phase shifts

**Example**: A motif might start at index 100 in dimension 0 but at index 105 in dimension 1 (5-bin delay).

### 4. Collision Detection

When adding motifs, the system ensures that:
1. No two motifs overlap in the same dimension
2. Overlaps in different dimensions are allowed
3. Motifs don't extend beyond the signal boundaries
4. When adding multiple motifs, failed placements are retried up to 10× the requested number

### 5. Metadata Storage and Serialization

#### Saved Information
For each motif instance:
- Motif type
- Exact start index
- Dimensions it spans
- Length and amplitude
- All parameters (even if sampled randomly)
- Dimension-specific offsets

For the entire simulation:
- Signal dimensions (length × n_dims)
- Base signal type
- All base signal parameters
- Noise levels per dimension
- Complete list of motif instances

#### File Format
- **Format**: Python pickle (`.pkl`)
- **Contains**: 
  - Signal: numpy array of shape (length, n_dimensions)
  - Metadata: Complete SimulationMetadata object
- **Loading**: Use `MultidimensionalSimulator.load_simulation(filepath)` to reconstruct

### 6. Interactive Editing

After placing motifs, you can:
- **Modify amplitude**: Change the scaling of the motif
- **Adjust parameters**: Modify motif-specific parameters (frequency, damping, etc.)
- **Real-time visualization**: See changes immediately in the plot
- **Signal update**: Old motif is subtracted and new one is added (no signal regeneration)

**Limitations**: 
- Cannot change motif location after placement
- Cannot change dimensions after placement
- Cannot change motif length after placement

---

## User Guide

### Quick Start

1. **Launch the app**:
   ```bash
   streamlit run streamlit_motif_simulator.py
   ```

2. **Generate a base signal**:
   - In the sidebar, set signal length and number of dimensions
   - Choose between AR process or sinusoidal signal
   - Configure parameters and noise levels
   - Click "Generate Base Signal"

3. **Add motifs**:
   - Select motif type and parameters
   - Choose target dimensions
   - Set start position
   - Click "Add Motif" or "Add Multiple Motifs"

4. **Visualize**:
   - View interactive plot in the main panel
   - Hover for detailed information
   - Zoom and pan as needed

5. **Edit (optional)**:
   - Select a motif from the table
   - Adjust amplitude or parameters
   - Click "Apply Changes"

6. **Save**:
   - Enter filename
   - Click "Save Dataset"
   - Metadata is automatically saved with the signal

### Tips and Best Practices

1. **AR Process Stability**: For order > 3, ensure coefficients are small (< 0.7) to avoid instability
2. **Motif Parameters**: Experiment with ranges to get diverse motif instances
3. **Dimension Offsets**: Use small offsets (±5 bins) for realistic propagation delays
4. **Multiple Motifs**: Request more than needed; some will fail due to collisions
5. **Editing**: Make small parameter changes to see the effect incrementally

---

## Code Structure

### Key Algorithms

#### Collision Detection
```
For each new motif:
  For each existing motif:
    If dimensions overlap:
      For each overlapping dimension:
        Check if time ranges [start, start+length] overlap
        If overlap found: Return False (collision)
  Return True (no collision)
```

#### Motif Addition
```
1. Generate motif waveform based on type and parameters
2. For each target dimension:
   a. Calculate actual start index (start_index + offset)
   b. Normalize motif to match amplitude
   c. Add motif to signal via signal[start:start+length, dim] += motif
3. Store MotifInstance with all metadata
4. Return success status
```

#### Multiple Motif Placement
```
For attempt = 1 to max_attempts:
  Sample random motif parameters from configured ranges
  If add_motif() succeeds:
    Increment added counter
  If added >= n_motifs:
    Break
Return number of successfully added motifs
```

### Dependencies

Core Module:
- `numpy`: Signal processing and array operations
- `scipy`: Signal processing functions (wavelets, filtering)
- `dataclasses`: Metadata structure definitions
- `pickle`: Serialization

Streamlit App:
- `streamlit`: Web framework
- `plotly`: Interactive visualizations
- `pandas`: Data table display
- Plus all core dependencies

---

## Advanced Usage

### Custom Motif Types

To add custom motif types:

1. Add a new static method to `MotifGenerator`:
   ```python
   @staticmethod
   def my_custom_motif(length: int, amplitude: float, **kwargs) -> np.ndarray:
       # Your implementation
       return motif
   ```

2. Update `generate_motif()` to handle the new type:
   ```python
   elif motif_type == 'my_custom':
       return MotifGenerator.my_custom_motif(length, amplitude, **kwargs)
   ```

3. Add to `AVAILABLE_MOTIF_TYPES` in the simulator

### Batch Processing

Load and process saved datasets:

```python
from multidimensionnal_motifs_simulator import MultidimensionalSimulator

simulator, signal = MultidimensionalSimulator.load_simulation('my_dataset.pkl')
metadata = simulator.get_metadata()

for motif in metadata.motifs:
    print(f"Motif {motif.motif_type} at index {motif.start_index}")
```

### Deterministic Reproduction

For reproducible datasets:

```python
simulator = MultidimensionalSimulator(
    signal_length=1000,
    n_dimensions=3,
    seed=42  # Set seed for reproducibility
)
```

---

## Troubleshooting

### Motifs Not Being Added
- **Cause**: Collisions with existing motifs
- **Solution**: Use more attempts or smaller motifs; check motif length vs. signal length

### Unrealistic Signal Appearance
- **Cause**: AR coefficients too large or wrong noise levels
- **Solution**: Use coefficients < 0.7 for stability; adjust noise levels

### Memory Issues with Large Signals
- **Cause**: High resolution × many dimensions
- **Solution**: Reduce signal length or dimensions; process in chunks

### Plotly Visualization Slow
- **Cause**: Too many data points or dimensions
- **Solution**: Reduce visualization resolution; plot fewer dimensions at a time

---

## Example Workflows

### Workflow 1: Simple Single-Dimensional Dataset

```
1. Signal length: 1000 samples
2. AR(2) process with φ₁=0.5, φ₂=0.2, noise=1.0
3. Add 5 random motifs of various types
4. Save dataset
5. Load and inspect metadata
```

### Workflow 2: Multi-Dimensional with Specific Motifs

```
1. Signal length: 2000, dimensions: 3
2. Sinusoidal base signal, different noise per dimension
3. Add 3 specific Morlet wavelets in dimension 0-1
4. Add 2 exponential decays in dimension 1-2
5. Edit amplitudes to fine-tune
6. Save dataset for algorithm testing
```

### Workflow 3: Validation Dataset

```
1. Create base signal
2. Add known number of motifs with recorded parameters
3. Save multiple variations with different noise levels
4. Use for benchmarking motif detection algorithms
```

---

## Performance Characteristics

| Operation | Time (typical) | Notes |
|-----------|---|---|
| Generate base signal (1000 samples, 5D) | ~100ms | Depends on AR order |
| Add single motif | ~10ms | Fast collision detection |
| Add 20 random motifs | ~200ms | With collision retries |
| Save to pickle | ~50ms | Fast serialization |
| Interactive plot render | ~500ms | Plotly creation |
| Edit motif parameter | ~100ms | Subtract + add motif |

---

## Future Enhancements

Potential additions for future versions:
- **Morphing motifs**: Continuously varying motif shape
- **Phase-locked motifs**: Synchronized across dimensions
- **Outlier injection**: Add sparse impulses or anomalies
- **Cross-dimensional coupling**: Motifs correlated across dimensions
- **Synthetic templates**: Generate motif libraries from real data
- **Batch API**: Generate multiple datasets programmatically

---

## Citation

If you use this tool in your research, please cite:

```
Multidimensional Motif Simulator v1.0
A tool for generating synthetic signals with embedded motifs
for motif detection algorithm validation
```

---

## License

This tool is provided for research and development purposes.

---

## Contact and Support

For issues, feature requests, or improvements, please contact the development team or submit issues through the project repository.

---

## Appendix: Mathematical Reference

### AR(p) Stationarity Condition
An AR(p) process is stationary if all roots of the characteristic equation lie outside the unit circle:
$$1 - \phi_1 z - \phi_2 z^2 - \cdots - \phi_p z^p = 0 \Rightarrow |z_i| > 1 \forall i$$

### Morlet Wavelet
$$\psi(t) = e^{-t^2/(2\sigma^2)} \cos(2\pi f t)$$

### Dampened Oscillator
$$x(t) = A e^{-\zeta t} \sin(2\pi f t)$$

### Signal Model with Motifs
$$s(t) = s_0(t) + \sum_{i=1}^{M} \sum_{d \in D_i} m_i(t - \tau_i^d)$$

where:
- $s_0(t)$: Base signal
- $M$: Number of motifs
- $D_i$: Dimensions of motif $i$
- $m_i()$: Motif waveform
- $\tau_i^d$: Start time of motif $i$ in dimension $d$

