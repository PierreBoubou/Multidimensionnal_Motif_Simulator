"""
Streamlit App for Multidimensional Motif Simulator

This app allows users to interactively create synthetic multidimensional signals with embedded motifs
for testing motif detection algorithms.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from multidimensionnal_motifs_simulator import (
    MultidimensionalSimulator, MotifGenerator, MotifInstance
)
import os
import pickle
from typing import Tuple

st.set_page_config(page_title="Multidimensional Motif Simulator", layout="wide")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'simulator' not in st.session_state:
    st.session_state.simulator = None
if 'signal_generated' not in st.session_state:
    st.session_state.signal_generated = False
if 'edited_motifs' not in st.session_state:
    st.session_state.edited_motifs = {}
if 'signal_copy' not in st.session_state:
    st.session_state.signal_copy = None
if 'collision_mask' not in st.session_state:
    st.session_state.collision_mask = None
if 'base_signal_no_motifs' not in st.session_state:
    st.session_state.base_signal_no_motifs = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def update_collision_mask():
    """Update the collision mask based on current motifs."""
    if st.session_state.simulator is None or st.session_state.simulator.signal is None:
        return
    
    signal_length = st.session_state.simulator.signal_length
    n_dims = st.session_state.simulator.n_dimensions
    
    # Initialize mask: True means available for placement
    mask = np.ones((signal_length, n_dims), dtype=bool)
    
    # Mark occupied regions
    for motif in st.session_state.simulator.motif_instances:
        for dim in motif.dimensions:
            start = motif.start_index + motif.dimension_offsets.get(dim, 0)
            end = min(start + motif.length, signal_length)
            if start >= 0 and start < signal_length:
                mask[start:end, dim] = False
    
    st.session_state.collision_mask = mask


def find_available_slots(motif_length: int, dimensions: list, max_attempts: int = 100) -> list:
    """Find available slots for placing motifs using mask array."""
    if st.session_state.collision_mask is None:
        update_collision_mask()
    
    mask = st.session_state.collision_mask
    signal_length = st.session_state.simulator.signal_length
    available_slots = []
    
    for attempt in range(max_attempts):
        # Find positions where all required dimensions are available
        available_positions = np.ones(signal_length - motif_length, dtype=bool)
        for dim in dimensions:
            for pos in range(signal_length - motif_length):
                if not np.all(mask[pos:pos + motif_length, dim]):
                    available_positions[pos] = False
        
        # Get all available positions
        positions = np.where(available_positions)[0]
        if len(positions) > 0:
            available_slots.append(positions)
    
    return available_slots if available_slots else []


def generate_motif_preview(motif_type: str, length: int, amplitude: float, **params) -> np.ndarray:
    """Generate a motif for preview."""
    return MotifGenerator.generate_motif(motif_type, length, amplitude, **params)


def create_interactive_plot():
    """Create an interactive Plotly plot of the signal with motif annotations."""
    if st.session_state.simulator is None or st.session_state.simulator.signal is None:
        return None
    
    signal = st.session_state.simulator.signal
    motifs = st.session_state.simulator.motif_instances
    base_signal = st.session_state.base_signal_no_motifs if st.session_state.base_signal_no_motifs is not None else None
    
    n_dims = signal.shape[1]
    
    # Create subplots
    fig = make_subplots(
        rows=n_dims, cols=1,
        subplot_titles=[f"Dimension {i}" for i in range(n_dims)],
        shared_xaxes=True,
        vertical_spacing=0.08
    )
    
    # Add base signal as semi-transparent shadow
    if base_signal is not None:
        for dim in range(n_dims):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(base_signal.shape[0]),
                    y=base_signal[:, dim],
                    mode='lines',
                    name=f'Baseline {dim}',
                    opacity=0.3,
                    hovertemplate=f'<b>Baseline Dim {dim}</b><br>Index: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>',
                    line=dict(color=f'hsl({dim * 360 / n_dims}, 70%, 50%)', width=2)
                ),
                row=dim + 1, col=1
            )
    
    # Add signal traces (with motifs)
    for dim in range(n_dims):
        fig.add_trace(
            go.Scatter(
                x=np.arange(signal.shape[0]),
                y=signal[:, dim],
                mode='lines',
                name=f'Signal Dim {dim}',
                hovertemplate=f'<b>Dimension {dim}</b><br>Index: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>',
                line=dict(color=f'hsl({dim * 360 / n_dims}, 70%, 50%)', width=2)
            ),
            row=dim + 1, col=1
        )
    
    # Add motif markers
    colors_motifs = {}
    motif_types = list(set([m.motif_type for m in motifs]))
    for i, mtype in enumerate(motif_types):
        colors_motifs[mtype] = f'hsl({i * 360 / len(motif_types)}, 100%, 50%)' if motif_types else 'gray'
    
    for motif_idx, motif in enumerate(motifs):
        for dim in motif.dimensions:
            start_idx = motif.start_index + motif.dimension_offsets.get(dim, 0)
            
            # Add vertical line at motif start
            fig.add_vline(
                x=start_idx,
                line_dash="dash",
                line_color=colors_motifs.get(motif.motif_type, 'gray'),
                line_width=2,
                opacity=0.5,
                row=dim + 1, col=1
            )
            
            # Add annotation
            fig.add_annotation(
                x=start_idx,
                y=np.max(st.session_state.simulator.signal[:, dim]),
                text=f"{motif.motif_type}<br>M{motif_idx}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=colors_motifs.get(motif.motif_type, 'gray'),
                ax=0,
                ay=-30,
                bgcolor=colors_motifs.get(motif.motif_type, 'gray'),
                opacity=0.7,
                font=dict(color="white", size=10),
                row=dim + 1, col=1
            )
    
    fig.update_layout(
        height=300 * n_dims,
        title="Multidimensional Signal with Motifs",
        hovermode='x unified',
        showlegend=True
    )
    
    fig.update_yaxes(title_text="Amplitude")
    fig.update_xaxes(title_text="Time Index", row=n_dims, col=1)
    
    return fig


def create_motif_preview_plot(motif_type: str, length: int, amplitude: float, **params):
    """Create a simple plot of the motif preview."""
    motif = generate_motif_preview(motif_type, length, amplitude, **params)
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(length),
            y=motif,
            mode='lines',
            name=f'{motif_type.capitalize()} Motif',
            line=dict(width=3, color='#FF6B6B')
        )
    )
    fig.update_layout(
        title=f"{motif_type.upper()} Motif Preview",
        xaxis_title="Sample",
        yaxis_title="Amplitude",
        height=300,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def create_batch_preview_plot(motif_type: str, length_min: int, length_max: int, 
                              amp_min: float, amp_max: float, param_ranges: dict):
    """
    Create a preview showing min, typical, and max motifs from a batch.
    This helps visualize the range of variations in the batch.
    """
    # Calculate three variants
    typical_length = (length_min + length_max) // 2 if length_min <= length_max else length_min
    
    # Typical parameters (midpoint of ranges)
    typical_params = {}
    for param_name, (p_min, p_max) in param_ranges.items():
        typical_params[param_name] = (p_min + p_max) / 2 if p_min <= p_max else p_min
    
    # Min variant: min length, min amplitude, min parameters
    min_params = {}
    for param_name, (p_min, p_max) in param_ranges.items():
        min_params[param_name] = p_min
    
    # Max variant: max length, max amplitude, max parameters
    max_params = {}
    for param_name, (p_min, p_max) in param_ranges.items():
        max_params[param_name] = p_max
    
    # Generate the three motifs
    try:
        min_motif = generate_motif_preview(motif_type, length_min, amp_min, **min_params)
        typical_motif = generate_motif_preview(motif_type, typical_length, (amp_min + amp_max) / 2, **typical_params)
        max_motif = generate_motif_preview(motif_type, length_max, amp_max, **max_params)
    except Exception as e:
        st.error(f"Error generating preview: {str(e)[:100]}")
        return None
    
    # Create combined plot
    fig = go.Figure()
    
    # Add all three traces
    fig.add_trace(go.Scatter(
        x=np.arange(len(min_motif)),
        y=min_motif,
        mode='lines',
        name='Min (Min params)',
        line=dict(color='#d62728', width=2, dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=np.arange(len(typical_motif)),
        y=typical_motif,
        mode='lines',
        name='Typical (Mid params)',
        line=dict(color='#1f77b4', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=np.arange(len(max_motif)),
        y=max_motif,
        mode='lines',
        name='Max (Max params)',
        line=dict(color='#2ca02c', width=2, dash='dot')
    ))
    
    fig.update_layout(
        title=f"{motif_type.upper()} Batch Preview (Min, Typical, Max)",
        xaxis_title="Sample",
        yaxis_title="Amplitude",
        height=350,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def reconstruct_signal_with_motifs():
    """Reconstruct signal by adding all motifs to the base signal."""
    if st.session_state.simulator is None or st.session_state.base_signal_no_motifs is None:
        return
    
    # Start with base signal (no motifs)
    st.session_state.simulator.signal = st.session_state.base_signal_no_motifs.copy()
    
    # Re-add all motifs
    for motif in st.session_state.simulator.motif_instances:
        motif_signal = MotifGenerator.generate_motif(
            motif.motif_type, motif.length, motif.amplitude, **motif.parameters
        )
        for dim in motif.dimensions:
            start = motif.start_index + motif.dimension_offsets.get(dim, 0)
            end = min(start + motif.length, st.session_state.simulator.signal_length)
            st.session_state.simulator.signal[start:end, dim] += motif_signal[:end-start]


def get_motif_parameter_description(motif_type: str, param_name: str) -> str:
    """Get description for motif parameters."""
    descriptions = {
        'morlet': {
            'frequency': 'Center frequency of the wavelet (0-0.5). Lower = slower oscillation.',
            'sigma': 'Bandwidth parameter (0.1-5.0). Higher = wider frequency spread.'
        },
        'sine': {
            'frequency': 'Frequency of oscillation (0-0.5). Higher = more cycles.'
        },
        'square': {
            'duty_cycle': 'Fraction of cycle that is high (0-1). 0.5 = equal high/low.'
        },
        'exponential': {
            'rise_tau': 'Rise time constant. Controls how fast signal rises to peak.',
            'decay_tau': 'Decay time constant. Controls how fast signal decays from peak.'
        },
        'dampened': {
            'frequency': 'Center frequency of oscillation (0-0.5).',
            'damping': 'Damping coefficient (0-0.5). Higher = faster dampening.'
        }
    }
    
    return descriptions.get(motif_type, {}).get(param_name, param_name)


def parse_range_input(text_min: str, text_max: str, default_min: float, default_max: float) -> Tuple[float, float]:
    """Parse text inputs for min/max values with fallback to defaults."""
    try:
        min_val = float(text_min) if text_min.strip() else default_min
    except (ValueError, AttributeError):
        min_val = default_min
    
    try:
        max_val = float(text_max) if text_max.strip() else default_max
    except (ValueError, AttributeError):
        max_val = default_max
    
    # Ensure min <= max
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    
    return min_val, max_val


def create_param_toggle_input(param_name: str, min_default: float, max_default: float, key_prefix: str) -> Tuple[float, float]:
    """
    Create a parameter input with toggle between fixed and range modes.
    
    Returns:
    --------
    Tuple[float, float] : (min_value, max_value)
    """
    col1, col2 = st.columns([0.3, 1])
    
    with col1:
        is_range = st.checkbox("Range", value=False, key=f"{key_prefix}_toggle")
    
    with col2:
        if is_range:
            # Show min and max side-by-side
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                min_val = parse_range_input(
                    st.text_input("Min", str(min_default), key=f"{key_prefix}_min"),
                    "",
                    min_default,
                    min_default
                )[0]
            with subcol2:
                max_val = parse_range_input(
                    "",
                    st.text_input("Max", str(max_default), key=f"{key_prefix}_max"),
                    max_default,
                    max_default
                )[1]
        else:
            # Show single fixed value
            fixed_val = parse_range_input(
                st.text_input("Value", str(min_default), key=f"{key_prefix}_fixed"),
                "",
                min_default,
                min_default
            )[0]
            min_val, max_val = fixed_val, fixed_val
    
    return min_val, max_val


def serialize_simulator(sim):
    """Convert simulator to a serializable dict."""
    data = {
        'signal': sim.signal,
        'signal_length': sim.signal_length,
        'n_dimensions': sim.n_dimensions,
        'base_signal_type': sim.base_signal_type,
        'base_signal_params': sim.base_signal_params,
        'noise_levels': sim.noise_levels,
        'motif_families': sim.motif_families,
        'motifs': [
            {
                'motif_type': m.motif_type,
                'start_index': m.start_index,
                'dimensions': m.dimensions,
                'length': m.length,
                'amplitude': float(m.amplitude),
                'parameters': {k: float(v) if isinstance(v, np.ndarray) else v 
                              for k, v in m.parameters.items()},
                'dimension_offsets': {int(k): int(v) for k, v in m.dimension_offsets.items()},
                'family_label': getattr(m, 'family_label', None)
            }
            for m in sim.motif_instances
        ]
    }
    return data



# ============================================================================
# MAIN APP LAYOUT
# ============================================================================

st.title("🧬 Multidimensional Motif Simulator")

st.write("""
This application allows you to create synthetic multidimensional signals with embedded motifs
for testing and validating motif detection algorithms.
""")

# Add tabs for main operations
main_tabs = st.tabs(["Create Dataset", "Load Dataset"])

with main_tabs[0]:  # CREATE DATASET TAB
    st.divider()
    
    # ========================================================================
    # SIDEBAR: SIGNAL SETUP
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Signal Setup")
        
        # Basic parameters
        signal_length = st.number_input("Signal Length", value=10000, min_value=100, max_value=500000, step=100)
        n_dimensions = st.number_input("Number of Dimensions", value=2, min_value=1, max_value=10)
        
        st.subheader("Base Signal Type")
        base_signal_type = st.radio("Select Base Signal", ["AR Process", "Sinusoidal"])
        
        if base_signal_type == "AR Process":
            ar_order = st.slider("AR Order (p)", 0, 10, 2)
            if ar_order > 0:
                ar_coeffs = []
                st.write("AR Coefficients:")
                cols = st.columns(ar_order)
                for i in range(ar_order):
                    coeff = cols[i % 3].number_input(f"φ_{i+1}", value=0.5, min_value=-0.9, max_value=0.9, step=0.1, key=f"ar_coeff_{i}")
                    ar_coeffs.append(coeff)
            else:
                ar_coeffs = []
            
            base_noise_var = st.slider("Base Noise Variance", 0.1, 5.0, 1.0, step=0.1)
        else:
            base_frequency_text = st.text_input("Base Frequency", "0.1", key="base_freq_input")
            amplitude_mean_text = st.text_input("Amplitude Mean", "1.0", key="amplitude_mean_input")
            amplitude_var_text = st.text_input("Amplitude Variation (SD)", "0.1", key="amplitude_var_input")
            frequency_var_text = st.text_input("Frequency Variation (SD)", "0.01", key="frequency_var_input")
            
            # Parse values with validation
            try:
                base_frequency = float(base_frequency_text)
                base_frequency = max(0.01, min(0.45, base_frequency))
            except ValueError:
                base_frequency = 0.1
            
            try:
                amplitude_mean = float(amplitude_mean_text)
                amplitude_mean = max(0.1, amplitude_mean)
            except ValueError:
                amplitude_mean = 1.0
            
            try:
                amplitude_var = float(amplitude_var_text)
                amplitude_var = max(0.0, amplitude_var)
            except ValueError:
                amplitude_var = 0.1
            
            try:
                frequency_var = float(frequency_var_text)
                frequency_var = max(0.0, frequency_var)
            except ValueError:
                frequency_var = 0.01
        
        st.subheader("Dimension-specific Noise")
        noise_levels = []
        for dim in range(n_dimensions):
            nl = st.slider(f"Dimension {dim} Noise Level", 0.5, 2.0, 1.0, step=0.1, key=f"noise_{dim}")
            noise_levels.append(nl)
        
        # Generate base signal button
        if st.button("🎲 Generate Base Signal", type="primary"):
            st.session_state.simulator = MultidimensionalSimulator(
                signal_length=signal_length,
                n_dimensions=n_dimensions,
                base_signal_type="ar" if base_signal_type == "AR Process" else "sinusoidal"
            )
            st.session_state.simulator.set_noise_levels(noise_levels)
            
            if base_signal_type == "AR Process":
                st.session_state.simulator.generate_base_signal(
                    ar_coeffs=ar_coeffs,
                    noise_variance=base_noise_var
                )
            else:
                st.session_state.simulator.generate_base_signal(
                    base_frequency=base_frequency,
                    amplitude_mean=amplitude_mean,
                    amplitude_variation=amplitude_var,
                    frequency_variation=frequency_var
                )
            
            # Store base signal (before any motifs) and the scaled copy
            st.session_state.base_signal_no_motifs = st.session_state.simulator.signal.copy()
            st.session_state.signal_copy = st.session_state.simulator.signal.copy()
            st.session_state.signal_generated = True
            update_collision_mask()
            st.success("✅ Base signal generated!")
            st.rerun()
        
        # Base signal amplitude adjustment
        if st.session_state.signal_generated:
            st.divider()
            st.subheader("Adjust Base Signal")
            base_amplitude_text = st.text_input("Base Signal Amplitude Scale", "1.0", key="base_amp_scale")
            if st.button("Apply Amplitude Scale", key="apply_base_scale"):
                try:
                    base_amplitude_scale = float(base_amplitude_text)
                    if base_amplitude_scale <= 0:
                        st.error("Amplitude scale must be positive")
                    else:
                        # Scale the base signal and reconstruct with motifs
                        st.session_state.base_signal_no_motifs = st.session_state.signal_copy.copy() * base_amplitude_scale
                        reconstruct_signal_with_motifs()
                        st.success(f"✅ Base signal amplitude scaled by {base_amplitude_scale}x! Motifs preserved.")
                        st.rerun()
                except ValueError:
                    st.error("Please enter a valid number")


    # ========================================================================
    # MAIN CONTENT: ADD MOTIFS
    # ========================================================================
    
    if st.session_state.signal_generated:
        
        # Create two columns: left for controls, right for visualization
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.header("➕ Add Motifs")
            
            # Tabs for single vs multiple motifs
            motif_tabs = st.tabs(["Single Motif", "Multiple Motifs"])
            
            with motif_tabs[0]:  # SINGLE MOTIF
                st.subheader("Add Single Motif")
                
                motif_type = st.selectbox(
                    "Motif Type",
                    MultidimensionalSimulator.AVAILABLE_MOTIF_TYPES,
                    key="motif_type_single"
                )
                
                # Motif-specific parameters (NO RANGES for single motif)
                st.write("**Motif Parameters:**")
                motif_params = {}
                
                if motif_type == "morlet":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f'**Frequency** - {get_motif_parameter_description("morlet", "frequency")}')
                        freq_text = st.text_input("Frequency", "0.05", key="morlet_freq_single")
                        try:
                            freq_val = float(freq_text)
                            freq_val = max(0.01, min(0.45, freq_val))
                            motif_params["frequency"] = freq_val
                        except:
                            motif_params["frequency"] = 0.05
                    
                    with col2:
                        st.markdown(f'**Sigma** - {get_motif_parameter_description("morlet", "sigma")}')
                        sigma_text = st.text_input("Sigma", "10.0", key="morlet_sigma_single")
                        try:
                            sigma_val = float(sigma_text)
                            sigma_val = max(0.5, sigma_val)
                            motif_params["sigma"] = sigma_val
                        except:
                            motif_params["sigma"] = 10.0
                
                elif motif_type == "sine":
                    st.markdown(f'**Frequency** - {get_motif_parameter_description("sine", "frequency")}')
                    freq_text = st.text_input("Frequency", "1.0", key="sine_freq_single")
                    try:
                        freq_val = float(freq_text)
                        freq_val = max(0.01, min(10.0, freq_val))
                        motif_params["frequency"] = freq_val
                    except:
                        motif_params["frequency"] = 1.0
                
                elif motif_type == "square":
                    st.markdown(f'**Duty Cycle** - {get_motif_parameter_description("square", "duty_cycle")}')
                    duty_text = st.text_input("Duty Cycle (0-1)", "0.5", key="square_duty_single")
                    try:
                        duty_val = float(duty_text)
                        duty_val = max(0.01, min(0.99, duty_val))
                        motif_params["duty_cycle"] = duty_val
                    except:
                        motif_params["duty_cycle"] = 0.5
                
                elif motif_type == "exponential":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f'**Rise Tau** - {get_motif_parameter_description("exponential", "rise_tau")}')
                        rise_text = st.text_input("Rise Tau", "10.0", key="exp_rise_single")
                        try:
                            rise_val = float(rise_text)
                            rise_val = max(0.5, rise_val)
                            motif_params["rise_tau"] = rise_val
                        except:
                            motif_params["rise_tau"] = 10.0
                    
                    with col2:
                        st.markdown(f'**Decay Tau** - {get_motif_parameter_description("exponential", "decay_tau")}')
                        decay_text = st.text_input("Decay Tau", "20.0", key="exp_decay_single")
                        try:
                            decay_val = float(decay_text)
                            decay_val = max(0.5, decay_val)
                            motif_params["decay_tau"] = decay_val
                        except:
                            motif_params["decay_tau"] = 20.0
                
                elif motif_type == "dampened":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f'**Frequency** - {get_motif_parameter_description("dampened", "frequency")}')
                        freq_text = st.text_input("Frequency", "0.05", key="damp_freq_single")
                        try:
                            freq_val = float(freq_text)
                            freq_val = max(0.01, min(0.45, freq_val))
                            motif_params["frequency"] = freq_val
                        except:
                            motif_params["frequency"] = 0.05
                    
                    with col2:
                        st.markdown(f'**Damping** - {get_motif_parameter_description("dampened", "damping")}')
                        damp_text = st.text_input("Damping (0-1)", "0.05", key="damp_damp_single")
                        try:
                            damp_val = float(damp_text)
                            damp_val = max(0.01, min(0.99, damp_val))
                            motif_params["damping"] = damp_val
                        except:
                            motif_params["damping"] = 0.05
                
                # Length and Amplitude
                st.divider()
                st.write("**Motif Length and Amplitude:**")
                col1, col2 = st.columns(2)
                with col1:
                    motif_length_text = st.text_input("Length", "50", key="motif_length_text")
                    try:
                        motif_length = int(motif_length_text)
                        motif_length = max(5, min(signal_length // 2, motif_length))
                    except:
                        motif_length = 50
                
                with col2:
                    motif_amplitude_text = st.text_input("Amplitude", "1.0", key="motif_amplitude_text")
                    try:
                        motif_amplitude = float(motif_amplitude_text)
                        motif_amplitude = max(0.1, motif_amplitude)
                    except:
                        motif_amplitude = 1.0
                
                # Preview
                st.divider()
                st.write("**Motif Preview:**")
                try:
                    preview_fig = create_motif_preview_plot(motif_type, motif_length, motif_amplitude, **motif_params)
                    st.plotly_chart(preview_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate preview: {str(e)[:100]}")
                
                # Dimension selection
                st.divider()
                st.write("**Dimensions:**")
                dimensions = []
                for dim in range(n_dimensions):
                    if st.checkbox(f"Include Dimension {dim}", value=True, key=f"dim_single_{dim}"):
                        dimensions.append(dim)
                
                if not dimensions:
                    st.warning("Select at least one dimension")
                
                # Dimension offsets
                st.write("**Dimension Time Offsets (bins):**")
                dimension_offsets = {}
                for dim in dimensions:
                    offset = st.number_input(
                        f"Dim {dim} Offset",
                        value=0,
                        min_value=-motif_length,
                        max_value=motif_length,
                        step=1,
                        key=f"offset_{dim}_single"
                    )
                    dimension_offsets[dim] = offset
                
                # Start index slider
                max_start = max(1, signal_length - motif_length - 1)
                start_index = st.slider("Start Index", 0, max_start, max_start // 2, key="start_index_single")
                
                # Add motif button
                if st.button("✅ Add Motif", type="secondary", key="add_single_motif"):
                    if not dimensions:
                        st.error("Select at least one dimension")
                    else:
                        success = st.session_state.simulator.add_motif(
                            motif_type=motif_type,
                            start_index=start_index,
                            dimensions=dimensions,
                            length=motif_length,
                            amplitude=motif_amplitude,
                            dimension_offsets=dimension_offsets,
                            **motif_params
                        )
                        if success:
                            update_collision_mask()
                            st.success("✅ Motif added successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Motif placement failed (collision detected)")
            
            with motif_tabs[1]:  # MULTIPLE MOTIFS
                st.subheader("Add Multiple Random Motifs")
                
                # Step 1: Select motif type
                selected_motif_type = st.selectbox(
                    "Motif Type to Add",
                    MultidimensionalSimulator.AVAILABLE_MOTIF_TYPES,
                    key="motif_type_multiple"
                )
                
                st.divider()
                
                # Step 2: Motif-specific parameters with ranges
                st.write(f"**{selected_motif_type.upper()} Parameters:**")
                motif_param_ranges = {}
                
                if selected_motif_type == "morlet":
                    st.markdown(f'**Frequency** - {get_motif_parameter_description("morlet", "frequency")}')
                    col1, col2 = st.columns(2)
                    with col1:
                        freq_text = st.text_input("Min/Fixed Frequency", "0.05", key="param_morlet_freq_min")
                        try:
                            freq_min = float(freq_text)
                            # freq_min = max(0.01, min(0.45, freq_min))
                        except:
                            freq_min = 0.05
                    
                    with col2:
                        freq_max_text = st.text_input("Max Frequency", "0.05", key="param_morlet_freq_max")
                        try:
                            freq_max = float(freq_max_text)
                            # freq_max = max(0.01, min(0.45, freq_max))
                        except:
                            freq_max = freq_min
                        freq_max = max(freq_min, freq_max)
                    
                    motif_param_ranges["frequency"] = (freq_min, freq_max)
                    
                    st.markdown(f'**Sigma** - {get_motif_parameter_description("morlet", "sigma")}')
                    col1, col2 = st.columns(2)
                    with col1:
                        sigma_text = st.text_input("Min/Fixed Sigma", "10.0", key="param_morlet_sigma_min")
                        try:
                            sigma_min = float(sigma_text)
                            # sigma_min = max(0.5, sigma_min)
                        except:
                            sigma_min = 10.0
                    
                    with col2:
                        sigma_max_text = st.text_input("Max Sigma", "10.0", key="param_morlet_sigma_max")
                        try:
                            sigma_max = float(sigma_max_text)
                            # sigma_max = max(0.5, sigma_max)
                        except:
                            sigma_max = sigma_min
                        sigma_max = max(sigma_min, sigma_max)
                    
                    motif_param_ranges["sigma"] = (sigma_min, sigma_max)
                
                elif selected_motif_type == "sine":
                    st.markdown(f'**Frequency** - {get_motif_parameter_description("sine", "frequency")}')
                    col1, col2 = st.columns(2)
                    with col1:
                        freq_text = st.text_input("Min/Fixed Frequency", "1.0", key="param_sine_freq_min")
                        try:
                            freq_min = float(freq_text)
                            # freq_min = max(0.01, min(10.0, freq_min))
                        except:
                            freq_min = 1.0
                    
                    with col2:
                        freq_max_text = st.text_input("Max Frequency", "1.0", key="param_sine_freq_max")
                        try:
                            freq_max = float(freq_max_text)
                            # freq_max = max(0.01, min(10.0, freq_max))
                        except:
                            freq_max = freq_min
                        freq_max = max(freq_min, freq_max)
                    
                    motif_param_ranges["frequency"] = (freq_min, freq_max)
                
                elif selected_motif_type == "square":
                    st.markdown(f'**Duty Cycle** - {get_motif_parameter_description("square", "duty_cycle")}')
                    col1, col2 = st.columns(2)
                    with col1:
                        duty_text = st.text_input("Min/Fixed Duty Cycle (0-1)", "0.5", key="param_square_duty_min")
                        try:
                            duty_min = float(duty_text)
                            # duty_min = max(0.01, min(0.99, duty_min))
                        except:
                            duty_min = 0.5
                    
                    with col2:
                        duty_max_text = st.text_input("Max Duty Cycle (0-1)", "0.5", key="param_square_duty_max")
                        try:
                            duty_max = float(duty_max_text)
                            # duty_max = max(0.01, min(0.99, duty_max))
                        except:
                            duty_max = duty_min
                        duty_max = max(duty_min, duty_max)
                    
                    motif_param_ranges["duty_cycle"] = (duty_min, duty_max)
                
                elif selected_motif_type == "exponential":
                    st.markdown(f'**Rise Tau** - {get_motif_parameter_description("exponential", "rise_tau")}')
                    col1, col2 = st.columns(2)
                    with col1:
                        rise_text = st.text_input("Min/Fixed Rise Tau", "10.0", key="param_exp_rise_min")
                        try:
                            rise_min = float(rise_text)
                            # rise_min = max(0.5, rise_min)
                        except:
                            rise_min = 10.0
                    
                    with col2:
                        rise_max_text = st.text_input("Max Rise Tau", "10.0", key="param_exp_rise_max")
                        try:
                            rise_max = float(rise_max_text)
                            # rise_max = max(0.5, rise_max)
                        except:
                            rise_max = rise_min
                        rise_max = max(rise_min, rise_max)
                    
                    motif_param_ranges["rise_tau"] = (rise_min, rise_max)
                    
                    st.markdown(f'**Decay Tau** - {get_motif_parameter_description("exponential", "decay_tau")}')
                    col1, col2 = st.columns(2)
                    with col1:
                        decay_text = st.text_input("Min/Fixed Decay Tau", "20.0", key="param_exp_decay_min")
                        try:
                            decay_min = float(decay_text)
                            # decay_min = max(0.5, decay_min)
                        except:
                            decay_min = 20.0
                    
                    with col2:
                        decay_max_text = st.text_input("Max Decay Tau", "20.0", key="param_exp_decay_max")
                        try:
                            decay_max = float(decay_max_text)
                            # decay_max = max(0.5, decay_max)
                        except:
                            decay_max = decay_min
                        decay_max = max(decay_min, decay_max)
                    
                    motif_param_ranges["decay_tau"] = (decay_min, decay_max)
                
                elif selected_motif_type == "dampened":
                    st.markdown(f'**Frequency** - {get_motif_parameter_description("dampened", "frequency")}')
                    col1, col2 = st.columns(2)
                    with col1:
                        freq_text = st.text_input("Min/Fixed Frequency", "0.05", key="param_damp_freq_min")
                        try:
                            freq_min = float(freq_text)
                            # freq_min = max(0.01, min(0.45, freq_min))
                        except:
                            freq_min = 0.05
                    
                    with col2:
                        freq_max_text = st.text_input("Max Frequency", "0.05", key="param_damp_freq_max")
                        try:
                            freq_max = float(freq_max_text)
                            # freq_max = max(0.01, min(0.45, freq_max))
                        except:
                            freq_max = freq_min
                        freq_max = max(freq_min, freq_max)
                    
                    motif_param_ranges["frequency"] = (freq_min, freq_max)
                    
                    st.markdown(f'**Damping** - {get_motif_parameter_description("dampened", "damping")}')
                    col1, col2 = st.columns(2)
                    with col1:
                        damp_text = st.text_input("Min/Fixed Damping", "0.05", key="param_damp_damp_min")
                        try:
                            damp_min = float(damp_text)
                            # damp_min = max(0.01, damp_min)
                        except:
                            damp_min = 0.05
                    
                    with col2:
                        damp_max_text = st.text_input("Max Damping", "0.05", key="param_damp_damp_max")
                        try:
                            damp_max = float(damp_max_text)
                            # damp_max = max(0.01, damp_max)
                        except:
                            damp_max = damp_min
                        damp_max = max(damp_min, damp_max)
                    
                    motif_param_ranges["damping"] = (damp_min, damp_max)
                
                st.divider()
                
                # Step 3: Length and Amplitude ranges
                st.write("**Length and Amplitude Ranges:**")
                col1, col2 = st.columns(2)
                with col1:
                    length_min_text = st.text_input("Min/Fixed Length", "100", key="length_multi_min")
                    try:
                        length_min = int(length_min_text)
                        length_min = max(5, min(signal_length // 2, length_min))
                    except:
                        length_min = 100
                
                with col2:
                    length_max_text = st.text_input("Max Length", "100", key="length_multi_max")
                    try:
                        length_max = int(length_max_text)
                        length_max = max(5, min(signal_length // 2, length_max))
                    except:
                        length_max = length_min
                    length_max = max(length_min, length_max)
                
                col1, col2 = st.columns(2)
                with col1:
                    amp_min_text = st.text_input("Min/Fixed Amplitude", "0.5", key="amplitude_multi_min")
                    try:
                        amp_min = float(amp_min_text)
                        # amp_min = max(0.1, amp_min)
                    except:
                        amp_min = 0.5
                
                with col2:
                    amp_max_text = st.text_input("Max Amplitude", "0.5", key="amplitude_multi_max")
                    try:
                        amp_max = float(amp_max_text)
                        # amp_max = max(0.1, amp_max)
                    except:
                        amp_max = amp_min
                    amp_max = max(amp_min, amp_max)
                
                st.divider()
                
                # Step 4: Preview FIRST (moved before batch configuration)
                st.write("**Batch Preview (Min, Typical, Max variations):**")
                batch_preview_fig = create_batch_preview_plot(
                    selected_motif_type, 
                    length_min, length_max,
                    amp_min, amp_max,
                    motif_param_ranges
                )
                if batch_preview_fig:
                    st.plotly_chart(batch_preview_fig, use_container_width=True)
                
                st.divider()
                
                # Step 5: Number of motifs and dimensions (moved after preview)
                st.write("**Batch Configuration:**")
                col1, col2 = st.columns(2)
                with col1:
                    n_motifs_to_add_text = st.text_input("Number of Motifs", "10", key="num_motifs")
                    try:
                        n_motifs_to_add = float(n_motifs_to_add_text)
                        # amp_max = max(0.1, amp_max)
                    except:
                        n_motifs_to_add = 10
                    # n_motifs_to_add = st.slider("Number of Motifs", 1, 20, 5, key="num_motifs_slider")
                
                with col2:
                    max_dim_offset = st.slider("Max Dimension Offset", 0, 50, 5, key="max_dim_offset_multi")
                
                st.write("**Dimensions for All Motifs:**")
                selected_dimensions = []
                for dim in range(n_dimensions):
                    if st.checkbox(f"Include Dimension {dim}", value=True, key=f"dim_multi_{dim}"):
                        selected_dimensions.append(dim)
                
                if not selected_dimensions:
                    st.warning("Select at least one dimension")
                
                # Add motifs button
                if st.button("🎲 Add Multiple Motifs", type="secondary", key="add_multiple_motifs"):
                    if not selected_dimensions:
                        st.error("Select at least one dimension")
                    else:
                        config = {
                            'types': [selected_motif_type],
                            'length_range': (length_min, length_max),
                            'amplitude_range': (amp_min, amp_max),
                            'dimension_range': (len(selected_dimensions), len(selected_dimensions) + 1),
                            'max_dimension_offset': max_dim_offset,
                            'motif_params': {selected_motif_type: motif_param_ranges},
                            'forced_dimensions': selected_dimensions
                        }
                        
                        added = st.session_state.simulator.add_multiple_motifs_improved(
                            n_motifs_to_add, config
                        )
                        update_collision_mask()
                        st.info(f"✅ Added {added} out of {n_motifs_to_add} motifs")
                        st.rerun()
        
        with col_right:
            st.header("📊 Visualization")
            
            if st.session_state.simulator.signal is not None:
                fig = create_interactive_plot()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        # ====================================================================
        # MOTIF EDITING
        # ====================================================================
        
        st.divider()
        st.header("✏️ Edit Existing Motifs")
        
        if st.session_state.simulator.motif_instances:
            # Create a dataframe of motifs with family labels
            motif_data = []
            for idx, motif in enumerate(st.session_state.simulator.motif_instances):
                # Use getattr for backward compatibility with old saved motifs
                family = getattr(motif, 'family_label', None) or 'single'
                motif_data.append({
                    'ID': idx,
                    'Type': motif.motif_type,
                    'Family': family,
                    'Start Index': motif.start_index,
                    'Length': motif.length,
                    'Amplitude': f"{motif.amplitude:.3f}",
                    'Dimensions': str(motif.dimensions),
                })
            
            df_motifs = pd.DataFrame(motif_data)
            st.dataframe(df_motifs, use_container_width=True)
            
            # Select motif to edit
            motif_to_edit = st.selectbox(
                "Select motif to edit",
                range(len(st.session_state.simulator.motif_instances)),
                format_func=lambda x: f"Motif {x} ({st.session_state.simulator.motif_instances[x].motif_type}) [Family: {getattr(st.session_state.simulator.motif_instances[x], 'family_label', None) or 'single'}]"
            )
            
            motif = st.session_state.simulator.motif_instances[motif_to_edit]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Edit Motif {motif_to_edit}")
                
                # Check if this motif is part of a family
                family_label = getattr(motif, 'family_label', None)
                
                if family_label:
                    st.warning(f"This motif is part of family: **{family_label}**")
                    edit_mode = st.radio("Edit mode", ["Single Motif", "Scale All in Family"], key=f"edit_mode_{motif_to_edit}")
                else:
                    edit_mode = "Single Motif"
                
                if edit_mode == "Scale All in Family":
                    st.markdown("**Scale all motifs in this family by a multiplication factor**")
                    scale_factor = st.slider(
                        "Scale Factor (1.0 = no change)",
                        0.1, 3.0, 1.0, step=0.1,
                        key=f"scale_factor_{motif_to_edit}"
                    )
                    
                    if st.button("🔄 Scale All Motifs in Family", type="secondary", key=f"scale_family_{motif_to_edit}"):
                        # Find all motifs in this family
                        family_motifs = []
                        for idx, m in enumerate(st.session_state.simulator.motif_instances):
                            if getattr(m, 'family_label', None) == family_label:
                                family_motifs.append((idx, m))
                        
                        if family_motifs:
                            # Apply scale to all motifs
                            for idx, m in family_motifs:
                                old_motif = MotifGenerator.generate_motif(
                                    m.motif_type, m.length, m.amplitude, **m.parameters
                                )
                                
                                new_amplitude = m.amplitude * scale_factor
                                
                                for dim in m.dimensions:
                                    start = m.start_index + m.dimension_offsets.get(dim, 0)
                                    st.session_state.simulator.signal[start:start + m.length, dim] -= old_motif
                                
                                new_motif = MotifGenerator.generate_motif(
                                    m.motif_type, m.length, new_amplitude, **m.parameters
                                )
                                
                                for dim in m.dimensions:
                                    start = m.start_index + m.dimension_offsets.get(dim, 0)
                                    st.session_state.simulator.signal[start:start + m.length, dim] += new_motif
                                
                                st.session_state.simulator.motif_instances[idx].amplitude = new_amplitude
                            
                            # Update family metadata
                            if family_label in st.session_state.simulator.motif_families:
                                family_info = st.session_state.simulator.motif_families[family_label]
                                old_amp = family_info.get('amplitude', (1.0, 1.0))
                                family_info['amplitude'] = (old_amp[0] * scale_factor, old_amp[1] * scale_factor)
                            
                            st.success(f"✅ Scaled all {len(family_motifs)} motifs by {scale_factor}x")
                            st.rerun()
                    
                    st.divider()
                
                new_amplitude = st.slider(
                    "New Amplitude (Single Motif)",
                    0.1, 5.0, motif.amplitude, step=0.1,
                    key=f"edit_amp_{motif_to_edit}"
                )
                
                st.write("**Edit Parameters:**")
                new_params = motif.parameters.copy()
                
                for param_name, param_val in motif.parameters.items():
                    if param_name == 'frequency':
                        new_params[param_name] = st.slider(
                            f"{param_name}", 0.01, 0.5, param_val, step=0.02,
                            key=f"edit_param_{motif_to_edit}_{param_name}"
                        )
                    elif param_name == 'sigma':
                        new_params[param_name] = st.slider(
                            f"{param_name}", 0.3, 5.0, param_val, step=0.2,
                            key=f"edit_param_{motif_to_edit}_{param_name}"
                        )
                    elif param_name == 'damping':
                        new_params[param_name] = st.slider(
                            f"{param_name}", 0.01, 0.5, param_val, step=0.02,
                            key=f"edit_param_{motif_to_edit}_{param_name}"
                        )
                    elif param_name == 'decay_rate':
                        new_params[param_name] = st.slider(
                            f"{param_name}", 0.01, 0.5, param_val, step=0.02,
                            key=f"edit_param_{motif_to_edit}_{param_name}"
                        )
                    elif param_name == 'duty_cycle':
                        new_params[param_name] = st.slider(
                            f"{param_name}", 0.1, 0.9, param_val, step=0.1,
                            key=f"edit_param_{motif_to_edit}_{param_name}"
                        )
                
                if st.button("💾 Apply Changes", type="secondary", key=f"apply_{motif_to_edit}"):
                    # Subtract old motif
                    old_motif = MotifGenerator.generate_motif(
                        motif.motif_type, motif.length, motif.amplitude, **motif.parameters
                    )
                    for dim in motif.dimensions:
                        start = motif.start_index + motif.dimension_offsets.get(dim, 0)
                        st.session_state.simulator.signal[start:start + motif.length, dim] -= old_motif
                    
                    # Add new motif
                    new_motif = MotifGenerator.generate_motif(
                        motif.motif_type, motif.length, new_amplitude, **new_params
                    )
                    for dim in motif.dimensions:
                        start = motif.start_index + motif.dimension_offsets.get(dim, 0)
                        st.session_state.simulator.signal[start:start + motif.length, dim] += new_motif
                    
                    # Update motif instance
                    st.session_state.simulator.motif_instances[motif_to_edit].amplitude = new_amplitude
                    st.session_state.simulator.motif_instances[motif_to_edit].parameters = new_params
                    
                    st.success("✅ Motif updated!")
                    st.rerun()
            
            with col2:
                st.subheader("Motif Details")
                st.write(f"**Type:** {motif.motif_type}")
                st.write(f"**Start Index:** {motif.start_index}")
                st.write(f"**Length:** {motif.length}")
                st.write(f"**Current Amplitude:** {motif.amplitude:.3f}")
                st.write(f"**Dimensions:** {motif.dimensions}")
                st.write(f"**Dimension Offsets:** {motif.dimension_offsets}")
                st.write(f"**Parameters:** {motif.parameters}")
        
        else:
            st.info("No motifs added yet. Add motifs using the controls on the left.")
        
        # ====================================================================
        # SAVE DATASET
        # ====================================================================
        
        st.divider()
        st.header("💾 Save Dataset")
        
        save_filename = st.text_input("Filename (without extension)", "motif_dataset.pkl")
        
        if st.button("💾 Save Dataset", type="primary"):
            filepath = save_filename if save_filename.endswith('.pkl') else f"{save_filename}.pkl"
            try:
                # Use serializable format
                data = serialize_simulator(st.session_state.simulator)
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
                st.success(f"✅ Dataset saved to {filepath}")
            except Exception as e:
                st.error(f"❌ Error saving dataset: {str(e)}")


    else:
        st.info("👈 Generate a base signal first using the controls on the left sidebar")


with main_tabs[1]:  # LOAD DATASET TAB
    st.header("📂 Load Existing Dataset")
    
    uploaded_file = st.file_uploader("Choose a PKL file", type="pkl", key="upload_dataset")
    
    if uploaded_file is not None:
        try:
            data = pickle.load(uploaded_file)
            
            # Display metadata
            st.subheader("Dataset Information")
            st.write(f"**Signal Shape:** {data['signal'].shape}")
            st.write(f"**Dimensions:** {data['n_dimensions']}")
            st.write(f"**Number of Motifs:** {len(data['motifs'])}")
            st.write(f"**Base Signal Type:** {data['base_signal_type']}")
            
            # Load button
            if st.button("📥 Load This Dataset", type="primary", key="load_dataset_btn"):
                # Reconstruct simulator
                sim = MultidimensionalSimulator(
                    signal_length=data['signal_length'],
                    n_dimensions=data['n_dimensions'],
                    base_signal_type=data['base_signal_type']
                )
                sim.signal = data['signal'].copy()
                sim.base_signal_params = data['base_signal_params']
                sim.noise_levels = data['noise_levels']
                sim.motif_families = data.get('motif_families', {})
                
                # Reconstruct motif instances
                for m_data in data['motifs']:
                    motif = MotifInstance(
                        motif_type=m_data['motif_type'],
                        start_index=m_data['start_index'],
                        dimensions=m_data['dimensions'],
                        length=m_data['length'],
                        amplitude=m_data['amplitude'],
                        parameters=m_data['parameters'],
                        dimension_offsets=m_data['dimension_offsets'],
                        family_label=m_data.get('family_label', None)
                    )
                    sim.motif_instances.append(motif)
                
                st.session_state.simulator = sim
                st.session_state.signal_generated = True
                st.session_state.signal_copy = sim.signal.copy()
                update_collision_mask()
                st.success("✅ Dataset loaded successfully!")
                st.rerun()
            
            # Display signal preview
            st.subheader("Signal Preview")
            fig = go.Figure()
            
            for dim in range(min(data['n_dimensions'], 3)):
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(data['signal'].shape[0]),
                        y=data['signal'][:, dim],
                        mode='lines',
                        name=f'Dim {dim}',
                        line=dict(width=1)
                    )
                )
            
            fig.update_layout(
                title="Signal Preview (first 3 dimensions)",
                xaxis_title="Time Index",
                yaxis_title="Amplitude",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display motif information
            if data['motifs']:
                st.subheader("Motifs Information")
                motif_data = []
                for idx, m in enumerate(data['motifs']):
                    motif_data.append({
                        'ID': idx,
                        'Type': m['motif_type'],
                        'Start': m['start_index'],
                        'Length': m['length'],
                        'Amplitude': f"{m['amplitude']:.3f}",
                        'Dimensions': str(m['dimensions'])
                    })
                
                df_motifs = pd.DataFrame(motif_data)
                st.dataframe(df_motifs, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption(
    "Multidimensional Motif Simulator - "
    "Create synthetic datasets with embedded motifs for algorithm testing"
)

