# Multidimensional Motif Simulator - Tutorial & Examples

## Quick Tutorial: Creating Your First Dataset

This tutorial walks you through creating a complete dataset from scratch.

### Step 1: Launch the App

```bash
streamlit run streamlit_motif_simulator.py
```

The app opens at `http://localhost:8501`

### Step 2: Generate Base Signal

In the **Sidebar** under "Signal Setup":

1. Set **Signal Length** to `1000`
2. Set **Number of Dimensions** to `2`
3. Select **"AR Process"** as base signal type
4. Set **AR Order** to `2`
5. Set **φ₁** (phi_1) to `0.5`
6. Set **φ₂** (phi_2) to `0.3`
7. Set **Base Noise Variance** to `1.0`
8. Set **Dimension 0 Noise Level** to `1.0`
9. Set **Dimension 1 Noise Level** to `1.2`
10. Click the blue **"Generate Base Signal"** button

**Result**: A 1000-sample, 2-dimensional AR(2) signal appears in the visualization panel.

### Step 3: Add Your First Motif

On the **left panel** under "Add Motifs":

1. Select **Motif Type**: `sine`
2. Set **Motif Length**: `50`
3. Set **Motif Amplitude**: `2.0`
4. Set **Frequency**: `0.15`
5. Check both **Dimension 0** and **Dimension 1** checkboxes
6. Set both dimension offsets to `0`
7. Move the **Start Index** slider to `200`
8. Click the green **"Add Motif"** button

**Result**: 
- A sine wave appears in the plot at index 200 in both dimensions
- The motif is labeled "sine" in the visualization
- A row appears in the motif table

### Step 4: Add More Motifs

Repeat Step 3 with different parameters:

**Motif 2: Morlet Wavelet**
- Type: `morlet`
- Length: `60`
- Amplitude: `1.5`
- Frequency: `0.12`
- Sigma: `1.5`
- Dimensions: `0` only
- Start Index: `450`
- Offset dim 0: `0`

**Motif 3: Exponential Decay**
- Type: `exponential`
- Length: `75`
- Amplitude: `2.5`
- Decay Rate: `0.1`
- Dimensions: `1` only
- Start Index: `700`
- Offset dim 1: `0`

**Result**: Your signal now has 3 distinct motifs at different locations.

### Step 5: Add Multiple Random Motifs

Instead of adding one at a time, add multiple randomly:

1. Expand **"Add Multiple Random Motifs"** section
2. Set **Number of Motifs**: `5`
3. Select **Motif Types**: check all available types
4. Set **Min Motif Length**: `30`
5. Set **Max Motif Length**: `100`
6. Set **Min Amplitude**: `1.0`
7. Set **Max Amplitude**: `2.5`
8. Set **Max Dimension Offset**: `5`
9. Click **"Add Multiple Motifs"**

**Result**: 5 additional motifs are placed randomly without overlaps.

### Step 6: Visualize and Inspect

In the **right panel**:
- Hover over any point to see exact values
- Use the Plotly toolbar (top-right) to zoom, pan, or save
- Click motif labels for information
- Each dimension shows separately

### Step 7: Edit an Existing Motif

Under **"Edit Existing Motifs"**:

1. Select **Motif 0** (the first sine motif) from dropdown
2. Move **New Amplitude** slider to `3.0`
3. Adjust **Frequency** to `0.18`
4. Click blue **"Apply Changes"**

**Result**:
- The motif appears larger in the plot
- The frequency increases (more oscillations per unit)
- No other motifs are affected
- The signal is updated in real-time

### Step 8: Save Your Dataset

Under **"Save Dataset"**:

1. Set **Filename**: `my_first_dataset`
2. Click **"Save Dataset"**
3. View the metadata JSON displayed

**Result**: A file `my_first_dataset.pkl` is created containing:
- The complete 1000×2 signal
- All 8 motif instances with exact parameters
- Base signal configuration
- Noise levels

---

## Complete Python Example: Programmatic Usage

```python
from multidimensionnal_motifs_simulator import MultidimensionalSimulator
import numpy as np

# Create simulator
sim = MultidimensionalSimulator(
    signal_length=2000,
    n_dimensions=3,
    base_signal_type='ar',
    seed=42  # For reproducibility
)

# Configure noise
sim.set_noise_levels([1.0, 1.2, 0.9])

# Generate base AR(2) signal
signal = sim.generate_base_signal(
    ar_coeffs=[0.5, 0.2],
    noise_variance=1.0
)

# Add individual motifs
sim.add_motif(
    motif_type='morlet',
    start_index=300,
    dimensions=[0, 1],  # Spans dimensions 0 and 1
    length=60,
    amplitude=2.0,
    frequency=0.1,
    sigma=1.5
)

sim.add_motif(
    motif_type='sine',
    start_index=600,
    dimensions=[1, 2],  # Spans dimensions 1 and 2
    length=80,
    amplitude=1.5,
    frequency=0.15,
    dimension_offsets={1: 0, 2: 5}  # 5-bin offset in dimension 2
)

# Add multiple random motifs
config = {
    'types': ['sine', 'morlet', 'triangle', 'exponential', 'dampened'],
    'length_range': (40, 120),
    'amplitude_range': (1.0, 2.5),
    'dimension_range': (1, 4),
    'max_dimension_offset': 10,
    'motif_params': {
        'morlet': {'frequency': (0.08, 0.15), 'sigma': (0.8, 2.0)},
        'sine': {'frequency': (0.1, 0.2)},
        'triangle': {},
        'square': {'duty_cycle': (0.3, 0.7)},
        'exponential': {'decay_rate': (0.05, 0.2)},
        'dampened': {'frequency': (0.08, 0.2), 'damping': (0.05, 0.2)}
    }
}

added = sim.add_multiple_motifs(n_motifs=10, motif_config=config)
print(f"Added {added} motifs successfully")

# Get metadata
metadata = sim.get_metadata()
print(f"Signal shape: {signal.shape}")
print(f"Number of motifs: {len(metadata.motifs)}")
for i, motif in enumerate(metadata.motifs):
    print(f"  Motif {i}: {motif.motif_type} at index {motif.start_index}")

# Save to file
sim.save_simulation('synthetic_dataset_example.pkl')
print("Dataset saved!")

# Load and verify
loaded_sim, loaded_signal = MultidimensionalSimulator.load_simulation('synthetic_dataset_example.pkl')
print(f"Loaded signal shape: {loaded_signal.shape}")
print(f"Loaded motif count: {len(loaded_sim.motif_instances)}")

# Use the signal
print(f"Signal statistics:")
print(f"  Mean: {loaded_signal.mean():.4f}")
print(f"  Std: {loaded_signal.std():.4f}")
print(f"  Min: {loaded_signal.min():.4f}")
print(f"  Max: {loaded_signal.max():.4f}")
```

### Output:
```
Added 10 motifs successfully
Signal shape: (2000, 3)
Number of motifs: 12
  Motif 0: morlet at index 300
  Motif 1: sine at index 600
  Motif 2: sine at index 850
  ...
  Motif 11: triangle at index 1750
Dataset saved!
Loaded signal shape: (2000, 3)
Loaded motif count: 12
Signal statistics:
  Mean: 0.0234
  Std: 1.2145
  Min: -4.3421
  Max: 3.8901
```

---

## Advanced Examples

### Example 1: Create a Dataset for Algorithm Validation

```python
from multidimensionnal_motifs_simulator import MultidimensionalSimulator
import numpy as np

# Create multiple datasets with known motif locations
for seed in range(5):
    sim = MultidimensionalSimulator(
        signal_length=3000,
        n_dimensions=5,
        base_signal_type='sinusoidal',
        seed=seed
    )
    
    # Sinusoidal base with varying parameters
    sim.generate_base_signal(
        base_frequency=0.08,
        amplitude_mean=1.0,
        amplitude_variation=0.15,
        frequency_variation=0.02
    )
    
    # Add controlled motifs
    config = {
        'types': ['morlet', 'dampened', 'exponential'],
        'length_range': (50, 150),
        'amplitude_range': (2.0, 4.0),
        'dimension_range': (2, 4),
        'max_dimension_offset': 3,
        'motif_params': {
            'morlet': {'frequency': (0.1, 0.15), 'sigma': (1.0, 1.5)},
            'dampened': {'frequency': (0.1, 0.2), 'damping': (0.08, 0.12)},
            'exponential': {'decay_rate': (0.08, 0.15)}
        }
    }
    
    sim.add_multiple_motifs(n_motifs=15, motif_config=config)
    
    # Save with seed in filename
    sim.save_simulation(f'validation_dataset_seed{seed}.pkl')
    
    # Save metadata summary
    meta = sim.get_metadata()
    print(f"Seed {seed}: {len(meta.motifs)} motifs in {meta.n_dimensions}D signal")
```

### Example 2: Modify Existing Dataset

```python
from multidimensionnal_motifs_simulator import MultidimensionalSimulator, MotifGenerator

# Load dataset
sim, signal = MultidimensionalSimulator.load_simulation('my_dataset.pkl')

# Increase amplitude of motif 2
motif_idx = 2
motif = sim.motif_instances[motif_idx]

# Generate old motif and subtract
old_motif = MotifGenerator.generate_motif(
    motif.motif_type, motif.length, motif.amplitude, **motif.parameters
)
for dim in motif.dimensions:
    start = motif.start_index + motif.dimension_offsets.get(dim, 0)
    sim.signal[start:start + motif.length, dim] -= old_motif

# Generate new motif with increased amplitude and subtract
new_amplitude = motif.amplitude * 1.5
new_motif = MotifGenerator.generate_motif(
    motif.motif_type, motif.length, new_amplitude, **motif.parameters
)
for dim in motif.dimensions:
    start = motif.start_index + motif.dimension_offsets.get(dim, 0)
    sim.signal[start:start + motif.length, dim] += new_motif

# Update metadata
sim.motif_instances[motif_idx].amplitude = new_amplitude

# Save modified dataset
sim.save_simulation('my_dataset_modified.pkl')
```

### Example 3: Analyze Motif Distribution

```python
from multidimensionnal_motifs_simulator import MultidimensionalSimulator
from collections import Counter

# Load dataset
sim, signal = MultidimensionalSimulator.load_simulation('dataset.pkl')
meta = sim.get_metadata()

# Analyze motif types
types = [m.motif_type for m in meta.motifs]
type_counts = Counter(types)
print("Motif type distribution:")
for mtype, count in type_counts.most_common():
    print(f"  {mtype}: {count}")

# Analyze motif lengths
lengths = [m.length for m in meta.motifs]
print(f"\nMotif length statistics:")
print(f"  Mean: {np.mean(lengths):.1f}")
print(f"  Min: {np.min(lengths)}")
print(f"  Max: {np.max(lengths)}")

# Analyze dimension coverage
dim_usage = Counter()
for motif in meta.motifs:
    for dim in motif.dimensions:
        dim_usage[dim] += 1

print(f"\nDimension usage:")
for dim in sorted(dim_usage.keys()):
    print(f"  Dimension {dim}: {dim_usage[dim]} motifs")

# Analyze amplitude distribution
amplitudes = [m.amplitude for m in meta.motifs]
print(f"\nAmplitude statistics:")
print(f"  Mean: {np.mean(amplitudes):.3f}")
print(f"  Std: {np.std(amplitudes):.3f}")
print(f"  Range: [{np.min(amplitudes):.3f}, {np.max(amplitudes):.3f}]")

# Check for temporal coverage
motif_coverage = np.zeros(meta.signal_length)
for motif in meta.motifs:
    for dim in motif.dimensions:
        start = motif.start_index + motif.dimension_offsets.get(dim, 0)
        end = min(start + motif.length, meta.signal_length)
        motif_coverage[start:end] += 1

print(f"\nTemporal coverage:")
covered = np.sum(motif_coverage > 0)
coverage_pct = 100 * covered / meta.signal_length
print(f"  Signal length: {meta.signal_length}")
print(f"  Covered points: {covered} ({coverage_pct:.1f}%)")
print(f"  Max overlap: {np.max(motif_coverage):.0f} motifs at same point")
```

---

## Troubleshooting Examples

### Issue: Motifs Not Being Added

**Symptom**: "Added 2 out of 5 motifs" message

**Diagnosis**: Likely collision with existing motifs

**Solution 1**: Use smaller motifs
```python
config = {
    'length_range': (20, 50),  # Much smaller
    'amplitude_range': (1.0, 2.5),
    ...
}
```

**Solution 2**: Request more attempts by adding fewer at once
```python
# Instead of 20 motifs:
sim.add_multiple_motifs(n_motifs=5, motif_config=config)
sim.add_multiple_motifs(n_motifs=5, motif_config=config)
sim.add_multiple_motifs(n_motifs=5, motif_config=config)
sim.add_multiple_motifs(n_motifs=5, motif_config=config)
```

**Solution 3**: Increase signal length
```python
sim = MultidimensionalSimulator(
    signal_length=5000,  # Much longer
    n_dimensions=3
)
```

### Issue: Unrealistic Base Signal

**Symptom**: AR signal explodes or doesn't look stationary

**Diagnosis**: AR coefficients violate stability condition

**Solution**:
```python
# Use smaller coefficients for stability
sim.generate_base_signal(
    ar_coeffs=[0.4, 0.25],  # Smaller values
    noise_variance=1.0
)

# Or reduce order
sim.generate_base_signal(
    ar_coeffs=[0.6],  # Only AR(1)
    noise_variance=1.0
)

# Or pure noise
sim.generate_base_signal(
    ar_coeffs=[],  # AR(0) = pure noise
    noise_variance=1.0
)
```

### Issue: Plotly Visualization Slow

**Symptom**: Plot takes several seconds to render

**Diagnosis**: Too many data points or dimensions

**Solution 1**: Reduce signal length
```python
sim = MultidimensionalSimulator(signal_length=1000)  # Instead of 10000
```

**Solution 2**: Reduce dimensions
```python
sim = MultidimensionalSimulator(n_dimensions=2)  # Instead of 10
```

**Solution 3**: Only visualize certain dimensions
```python
# In Streamlit, create filtered plot (advanced)
```

---

## Tips & Tricks

### Tip 1: Create Consistent Datasets
```python
# Use the same seed for reproducibility
sim1 = MultidimensionalSimulator(..., seed=42)
sim2 = MultidimensionalSimulator(..., seed=42)
# Both will generate identical results
```

### Tip 2: Create Variations with Different Noise
```python
base_config = {...}

for noise_level in [0.5, 1.0, 2.0, 4.0]:
    sim = MultidimensionalSimulator(...)
    sim.set_noise_levels([noise_level] * n_dimensions)
    # ... create motifs ...
    sim.save_simulation(f'dataset_noise_{noise_level}.pkl')
```

### Tip 3: Create Benchmark Suite
```python
# Different signal types
for signal_type in ['ar', 'sinusoidal']:
    # Different complexities
    for n_motifs in [5, 10, 20]:
        # Different dimensions
        for n_dims in [2, 5, 10]:
            sim = MultidimensionalSimulator(
                signal_length=2000,
                n_dimensions=n_dims,
                base_signal_type=signal_type
            )
            # ... create and save ...
```

### Tip 4: Save Without UI
```python
# Batch create datasets programmatically
for i in range(100):
    sim = MultidimensionalSimulator(...)
    sim.generate_base_signal(...)
    sim.add_multiple_motifs(...)
    sim.save_simulation(f'batch_dataset_{i:03d}.pkl')
print("Batch complete: 100 datasets created")
```

---

## Next Steps

1. **Start with the tutorial** - Follow the Quick Tutorial above
2. **Experiment interactively** - Use the Streamlit app to learn
3. **Create your first dataset** - Generate a simple 2D dataset
4. **Read the full guide** - `MULTIDIMENSIONAL_MOTIF_SIMULATOR_GUIDE.md`
5. **Try programmatic creation** - Use Python examples
6. **Create benchmark datasets** - For algorithm validation

---

**Happy dataset creation!** 🎉

For questions or issues, refer to:
- `README_QUICKSTART.md` - Quick reference
- `MULTIDIMENSIONAL_MOTIF_SIMULATOR_GUIDE.md` - Full documentation
- `IMPLEMENTATION_COMPLETE.md` - Implementation details
