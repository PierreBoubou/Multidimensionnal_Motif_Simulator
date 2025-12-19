# Multidimensional Motif Simulator - Quick Start

## What is This?

A comprehensive Streamlit application for creating synthetic multidimensional time-series datasets with embedded motifs. Perfect for testing and validating motif detection algorithms.

## Installation

Ensure you have the required packages:
```bash
pip install streamlit plotly pandas numpy scipy
```

## Launch the App

```bash
streamlit run streamlit_motif_simulator.py
```

The app will open in your default browser at `http://localhost:8501`

## What Can You Do?

### 1. **Generate Base Signals**
- AR(p) processes with configurable coefficients and noise
- Sinusoidal signals with varying amplitude and frequency
- Multidimensional support (1 to 10+ dimensions)

### 2. **Add Motifs**
Six different motif types available:
- **Morlet wavelet** - Localized oscillations
- **Sine period** - Single sine wave
- **Triangle** - Sharp triangular shape
- **Square** - Step-like pattern
- **Exponential decay** - Rise and decay pattern
- **Dampened oscillator** - Decaying sinusoid

### 3. **Customize Placement**
- Add single motifs at specific locations
- Add multiple random motifs automatically
- Control which dimensions each motif spans
- Set time offsets between dimensions
- Automatic collision detection prevents overlaps

### 4. **Interactive Visualization**
- Plotly charts for all dimensions
- Motif labels and markers
- Zoom, pan, and hover capabilities
- Real-time updates as you modify

### 5. **Edit Existing Motifs**
- Adjust amplitude and parameters
- See changes immediately
- Update signal without regeneration

### 6. **Save Datasets**
- Save signal and complete metadata as pickle file
- Preserves all motif information and parameters
- Easy loading for further processing

## Features

✅ **Full multidimensional support** - Signals spanning multiple dimensions  
✅ **6 motif types** - Diverse waveform options  
✅ **Dimension-specific control** - Each motif can use different dimensions  
✅ **Parameter ranges** - Sample motif parameters from distributions  
✅ **Time offsets** - Add delays between dimensions  
✅ **Collision detection** - Prevent motif overlap  
✅ **Real-time editing** - Modify motifs without regenerating  
✅ **Metadata preservation** - Save everything needed to reproduce  
✅ **Interactive visualization** - Beautiful Plotly plots  
✅ **Batch operations** - Add multiple motifs at once  

## Key Concepts

### Base Signal
The underlying signal before any motifs are added. Can be:
- **AR(p)**: Autoregressive process (stationary time series)
- **Sinusoidal**: Oscillating signal with varying amplitude/frequency

### Motif
A local pattern embedded in the signal at a specific location. Each motif has:
- Type (which waveform)
- Start index (where it begins)
- Dimensions (which dimensions it occupies)
- Length and amplitude
- Parameters (frequency, damping, etc.)
- Dimension offsets (different start times per dimension)

### Metadata
Complete information about the simulation saved with the dataset:
- Signal parameters
- Base signal type and configuration
- Noise levels per dimension
- All motif instances with exact parameters

## Example Workflow

1. **Generate base signal**
   - Length: 1000 samples
   - Dimensions: 3
   - Type: AR(2) process
   - Noise: 1.0 (all dimensions)

2. **Add motifs**
   - 5 random motifs from all types
   - Lengths: 50-150 samples
   - Amplitudes: 0.5-2.0

3. **Fine-tune**
   - Edit one motif's amplitude
   - Adjust a Morlet wavelet's frequency

4. **Save**
   - Save as `my_dataset.pkl`
   - Contains signal + complete metadata

5. **Use for testing**
   - Load saved dataset
   - Run motif detection algorithm
   - Compare detected vs. ground truth

## Files

- `multidimensionnal_motifs_simulator.py` - Core simulation module (classes and functions)
- `streamlit_motif_simulator.py` - Streamlit web application
- `MULTIDIMENSIONAL_MOTIF_SIMULATOR_GUIDE.md` - Detailed documentation
- `README_QUICKSTART.md` - This file

## Documentation

For detailed information about all features, parameters, and advanced usage, see:
- `MULTIDIMENSIONAL_MOTIF_SIMULATOR_GUIDE.md`

This comprehensive guide includes:
- Architecture overview
- Detailed feature descriptions
- Mathematical formulas
- Advanced usage examples
- Troubleshooting guide
- Performance characteristics

## Troubleshooting

**Motifs not being added?**
- Possible collision with existing motifs
- Try smaller motif lengths or fewer motifs

**Signal looks unrealistic?**
- Check AR coefficients (should be < 0.7)
- Adjust noise levels
- Try different parameter ranges

**App running slowly?**
- Reduce signal length or number of dimensions
- Reduce visualization complexity

## Tips

1. Start with simple signals (AR(0) = pure noise) to understand the UI
2. Use smaller signals (500-1000 samples) while experimenting
3. Add 3-5 motifs initially; adjust ranges once comfortable
4. Use the same dimension offsets across motifs for consistency
5. Save intermediate versions before making major changes

## Next Steps

1. Run the app: `streamlit run streamlit_motif_simulator.py`
2. Follow the sidebar prompts
3. Experiment with different parameter combinations
4. Save datasets for your algorithm testing
5. Check the comprehensive guide for advanced features

Enjoy creating synthetic datasets! 🚀
