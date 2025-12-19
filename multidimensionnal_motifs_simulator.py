"""
Multidimensional Motifs Simulator Module

This module provides comprehensive functionality for generating synthetic multidimensional signals
with embedded motifs for testing motif detection algorithms.

Key capabilities:
- Generate base signals (AR(p) process or sinusoidal with varying amplitude/frequency)
- Create various motif shapes (Morlet wavelet, sine, triangle, square, exponential, dampened oscillator)
- Add motifs to signals with collision detection
- Support for multidimensional signals with dimension-specific motif subspaces
- Serialize and deserialize simulations with complete metadata preservation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import pickle
from scipy import signal as scipy_signal
from scipy.special import gamma
import warnings

warnings.filterwarnings('ignore')


@dataclass
class MotifInstance:
    """Represents a single motif instance in the signal."""
    motif_type: str
    start_index: int
    dimensions: List[int]  # Which dimensions this motif spans
    length: int
    amplitude: float
    parameters: Dict[str, float]  # Motif-specific parameters
    dimension_offsets: Dict[int, int]  # Offset for each dimension (in bins)
    family_label: Optional[str] = None  # Label for motifs from same batch (e.g., 'family_0')


@dataclass
class SimulationMetadata:
    """Stores all metadata about a simulation."""
    signal_length: int
    n_dimensions: int
    base_signal_type: str  # 'ar' or 'sinusoidal'
    base_signal_params: Dict[str, Any]
    noise_levels: List[float]  # One per dimension
    motifs: List[MotifInstance]
    motif_families: Optional[Dict[str, Dict[str, Any]]] = None  # Family label -> parameters used
    
    def to_dict(self):
        return asdict(self)


class BaseSignalGenerator:
    """Generates base signals for the simulation."""
    
    @staticmethod
    def ar_process(length: int, ar_coeffs: Optional[List[float]] = None, 
                   noise_variance: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate an AR(p) process with Gaussian noise.
        
        Parameters:
        -----------
        length : int
            Length of the signal
        ar_coeffs : List[float], optional
            AR coefficients. If None or empty, generates pure noise (AR(0))
        noise_variance : float
            Variance of Gaussian noise
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray : Generated AR(p) signal
        """
        if seed is not None:
            np.random.seed(seed)
            
        if ar_coeffs is None or len(ar_coeffs) == 0:
            # Pure noise case
            return np.random.normal(0, np.sqrt(noise_variance), length)
        
        # Ensure AR process is stable (all roots outside unit circle)
        ar_coeffs = np.array(ar_coeffs)
        roots = np.roots(np.concatenate([[1], -ar_coeffs]))
        if np.any(np.abs(roots) <= 1.0):
            warnings.warn("AR coefficients may lead to non-stationary process. Adjusting...")
            ar_coeffs = ar_coeffs * 0.9
        
        # Generate AR process
        noise = np.random.normal(0, np.sqrt(noise_variance), length)
        signal = np.zeros(length)
        p = len(ar_coeffs)
        
        for i in range(p, length):
            signal[i] = np.sum(ar_coeffs * signal[i-p:i][::-1]) + noise[i]
        
        return signal
    
    @staticmethod
    def sinusoidal_signal(length: int, base_frequency: float = 0.1, 
                         amplitude_mean: float = 1.0, amplitude_variation: float = 0.1,
                         frequency_variation: float = 0.01, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a sinusoidal signal with varying amplitude and frequency.
        Mimics EMD-like behavior.
        
        Parameters:
        -----------
        length : int
            Length of the signal
        base_frequency : float
            Base frequency of the sinusoid (as fraction of Nyquist)
        amplitude_mean : float
            Mean amplitude
        amplitude_variation : float
            Standard deviation of amplitude variations
        frequency_variation : float
            Standard deviation of frequency variations
        seed : int, optional
            Random seed
            
        Returns:
        --------
        np.ndarray : Generated sinusoidal signal
        """
        if seed is not None:
            np.random.seed(seed)
        
        t = np.arange(length)
        
        # Create slowly-varying amplitude
        amplitude_envelope = amplitude_mean + np.cumsum(np.random.normal(0, amplitude_variation, length)) / 100
        amplitude_envelope = np.clip(amplitude_envelope, 0.1, None)
        
        # Create slowly-varying frequency
        frequency_variations = np.cumsum(np.random.normal(0, frequency_variation, length)) / 1000
        frequency = base_frequency + frequency_variations
        frequency = np.clip(frequency, 0.01, 0.45)
        
        # Generate phase
        phase = 2 * np.pi * np.cumsum(frequency)
        
        signal = amplitude_envelope * np.sin(phase)
        
        return signal


class MotifGenerator:
    """Generates various motif shapes."""
    
    @staticmethod
    def morlet_wavelet(length: int, frequency: float = None, sigma: float = None,
                       amplitude: float = 1.0) -> np.ndarray:
        """
        Generate a Morlet wavelet.
        
        Parameters:
        -----------
        length : int
            Length of the motif
        frequency : float, optional
            Normalized frequency of the wavelet. If None, defaults to 5/length (fills window with main oscillations)
        sigma : float, optional
            Time standard deviation. If None, defaults to length/5 (main oscillations fill window)
        amplitude : float
            Amplitude scaling
            
        Returns:
        --------
        np.ndarray : Morlet wavelet motif
        """
        # Default: main oscillations fill the window without large flat edges
        if frequency is None:
            frequency = 5.0 / length
        if sigma is None:
            sigma = length / 5.0
        
        t = np.arange(length) - length / 2
        real_part = np.exp(-t**2 / (2 * sigma**2)) * np.cos(2 * np.pi * frequency * t)
        motif = amplitude * real_part
        return motif / np.max(np.abs(motif)) * amplitude  # Normalize to amplitude
    
    @staticmethod
    def sine_period(length: int, frequency: float = None, amplitude: float = 1.0) -> np.ndarray:
        """
        Generate a single sine period.
        
        Parameters:
        -----------
        length : int
            Length of the motif
        frequency : float, optional
            Normalized frequency. If None, defaults to 1.0 (complete period fills window)
        amplitude : float
            Amplitude of the sine
            
        Returns:
        --------
        np.ndarray : Sine period motif
        """
        # Default: complete sine period fits in the window
        if frequency is None:
            frequency = 1.0
        
        t = np.arange(length) / length
        motif = amplitude * np.sin(2 * np.pi * frequency * t)
        return motif
    
    @staticmethod
    def triangle(length: int, amplitude: float = 1.0) -> np.ndarray:
        """
        Generate a triangle wave.
        
        Parameters:
        -----------
        length : int
            Length of the motif
        amplitude : float
            Amplitude of the triangle
            
        Returns:
        --------
        np.ndarray : Triangle wave motif
        """
        t = np.arange(length) / length
        # Rising from 0 to length/2, falling from length/2 to length
        motif = np.where(t < 0.5, 
                        amplitude * 2 * t,
                        amplitude * 2 * (1 - t))
        return motif
    
    @staticmethod
    def square(length: int, amplitude: float = 1.0, duty_cycle: float = 0.5) -> np.ndarray:
        """
        Generate a square wave.
        
        Parameters:
        -----------
        length : int
            Length of the motif
        amplitude : float
            Amplitude of the square
        duty_cycle : float
            Fraction of period where signal is high (0-1)
            
        Returns:
        --------
        np.ndarray : Square wave motif
        """
        t = np.arange(length) / length
        motif = amplitude * np.where(t < duty_cycle, 1.0, -1.0)
        return motif
    
    @staticmethod
    def exponential_pulse(length: int, amplitude: float = 1.0, 
                         rise_tau: float = None, decay_tau: float = None) -> np.ndarray:
        """
        Generate exponential pulse using difference of exponentials formula.
        This creates a smooth rise-and-decay shape similar to alpha functions used in neural models.
        Formula: amplitude * (exp(-decay*t) - exp(-rise*t))
        
        Parameters:
        -----------
        length : int
            Length of the motif
        amplitude : float
            Peak amplitude (normalized so peak reaches this value)
        rise_tau : float, optional
            Rise time constant (inverse rate). If None, defaults to length/8
        decay_tau : float, optional
            Decay time constant (inverse rate). If None, defaults to length/4
            
        Returns:
        --------
        np.ndarray : Exponential pulse motif
        """
        if rise_tau is None:
            rise_tau = length / 8.0
        if decay_tau is None:
            decay_tau = length / 4.0
        
        t = np.arange(length, dtype=float)
        
        # Difference of exponentials formula: exp(-t/decay) - exp(-t/rise)
        # This naturally rises (because rise is smaller initially dominates negatively)
        # and decays (because decay dominates at end)
        raw_motif = np.exp(-t / decay_tau) - np.exp(-t / rise_tau)
        
        # Find peak for normalization
        peak_idx = np.argmax(np.abs(raw_motif))
        peak_val = raw_motif[peak_idx]
        
        # Normalize to reach desired amplitude
        if peak_val != 0:
            motif = amplitude * raw_motif / peak_val
        else:
            motif = raw_motif
        
        return motif
    
    @staticmethod
    def dampened_oscillator(length: int, frequency: float = 0.1, 
                           damping: float = 0.1, amplitude: float = 1.0) -> np.ndarray:
        """
        Generate a dampened oscillator (decaying sinusoid).
        
        Parameters:
        -----------
        length : int
            Length of the motif
        frequency : float
            Normalized frequency
        damping : float
            Damping coefficient (0-1), higher = faster decay
        amplitude : float
            Initial amplitude
            
        Returns:
        --------
        np.ndarray : Dampened oscillator motif
        """
        t = np.arange(length)
        envelope = amplitude * np.exp(-damping * t / length)
        oscillation = np.sin(2 * np.pi * frequency * t)
        motif = envelope * oscillation
        return motif
    
    @staticmethod
    def generate_motif(motif_type: str, length: int, amplitude: float = 1.0,
                      **kwargs) -> np.ndarray:
        """
        Generate a motif of specified type.
        
        Parameters:
        -----------
        motif_type : str
            Type of motif: 'morlet', 'sine', 'triangle', 'square', 'exponential', 'dampened'
        length : int
            Length of the motif
        amplitude : float
            Amplitude of the motif
        **kwargs : dict
            Additional parameters specific to motif type
            
        Returns:
        --------
        np.ndarray : Generated motif
        """
        if motif_type == 'morlet':
            frequency = kwargs.get('frequency', None)
            sigma = kwargs.get('sigma', None)
            return MotifGenerator.morlet_wavelet(length, frequency, sigma, amplitude)
        
        elif motif_type == 'sine':
            frequency = kwargs.get('frequency', None)
            return MotifGenerator.sine_period(length, frequency, amplitude)
        
        elif motif_type == 'triangle':
            return MotifGenerator.triangle(length, amplitude)
        
        elif motif_type == 'square':
            duty_cycle = kwargs.get('duty_cycle', 0.5)
            return MotifGenerator.square(length, amplitude, duty_cycle)
        
        elif motif_type == 'exponential':
            rise_tau = kwargs.get('rise_tau', None)
            decay_tau = kwargs.get('decay_tau', None)
            return MotifGenerator.exponential_pulse(length, amplitude, rise_tau, decay_tau)
        
        elif motif_type == 'dampened':
            frequency = kwargs.get('frequency', 0.1)
            damping = kwargs.get('damping', 0.1)
            return MotifGenerator.dampened_oscillator(length, frequency, damping, amplitude)
        
        else:
            raise ValueError(f"Unknown motif type: {motif_type}")


class MultidimensionalSimulator:
    """Main class for creating multidimensional signals with motifs."""
    
    AVAILABLE_MOTIF_TYPES = ['morlet', 'sine', 'triangle', 'square', 'exponential', 'dampened']
    
    def __init__(self, signal_length: int, n_dimensions: int = 1, 
                 base_signal_type: str = 'ar', seed: Optional[int] = None):
        """
        Initialize the simulator.
        
        Parameters:
        -----------
        signal_length : int
            Length of the signal
        n_dimensions : int
            Number of dimensions
        base_signal_type : str
            'ar' for AR process, 'sinusoidal' for sinusoidal signal
        seed : int, optional
            Random seed
        """
        self.signal_length = signal_length
        self.n_dimensions = n_dimensions
        self.base_signal_type = base_signal_type
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        self.signal = None
        self.motif_instances = []
        self.motif_families = {}  # Family label -> parameters dict
        self.base_signal_params = {}
        self.noise_levels = [1.0] * n_dimensions
        
    def generate_base_signal(self, **params) -> np.ndarray:
        """
        Generate the base signal (without motifs).
        
        Parameters depend on signal type:
        For AR process:
            - ar_coeffs: list of AR coefficients
            - noise_variance: variance of noise
        For sinusoidal:
            - base_frequency: base frequency
            - amplitude_mean: mean amplitude
            - amplitude_variation: amplitude variation SD
            - frequency_variation: frequency variation SD
            
        Returns:
        --------
        np.ndarray : Base signal of shape (signal_length, n_dimensions)
        """
        self.base_signal_params = params
        signal = np.zeros((self.signal_length, self.n_dimensions))
        
        if self.base_signal_type == 'ar':
            ar_coeffs = params.get('ar_coeffs', [])
            noise_variance = params.get('noise_variance', 1.0)
            
            for dim in range(self.n_dimensions):
                noise_var = noise_variance * self.noise_levels[dim]
                signal[:, dim] = BaseSignalGenerator.ar_process(
                    self.signal_length, ar_coeffs, noise_var
                )
        
        elif self.base_signal_type == 'sinusoidal':
            base_frequency = params.get('base_frequency', 0.1)
            amplitude_mean = params.get('amplitude_mean', 1.0)
            amplitude_variation = params.get('amplitude_variation', 0.1)
            frequency_variation = params.get('frequency_variation', 0.01)
            
            for dim in range(self.n_dimensions):
                signal[:, dim] = BaseSignalGenerator.sinusoidal_signal(
                    self.signal_length, base_frequency, amplitude_mean,
                    amplitude_variation, frequency_variation
                ) * self.noise_levels[dim]
        
        self.signal = signal
        return signal
    
    def set_noise_levels(self, noise_levels: List[float]):
        """Set noise levels for each dimension."""
        if len(noise_levels) != self.n_dimensions:
            raise ValueError(f"Expected {self.n_dimensions} noise levels")
        self.noise_levels = noise_levels
    
    def add_motif(self, motif_type: str, start_index: int, dimensions: List[int],
                  length: int, amplitude: float, dimension_offsets: Optional[Dict[int, int]] = None,
                  family_label: Optional[str] = None, **motif_params) -> bool:
        """
        Add a single motif to the signal.
        
        Parameters:
        -----------
        motif_type : str
            Type of motif
        start_index : int
            Starting index in the signal
        dimensions : List[int]
            Dimensions where the motif is placed
        length : int
            Length of the motif
        amplitude : float
            Amplitude of the motif
        dimension_offsets : Dict[int, int], optional
            Offset for each dimension (in bins)
        family_label : str, optional
            Label for grouping motifs from same batch (e.g., 'family_0')
        **motif_params : dict
            Additional parameters for the motif generator
            
        Returns:
        --------
        bool : True if successful, False if collision detected
        """
        if self.signal is None:
            raise ValueError("Generate base signal first with generate_base_signal()")
        
        if dimension_offsets is None:
            dimension_offsets = {dim: 0 for dim in dimensions}
        
        # Check for collisions
        if not self._check_collision_free(start_index, length, dimensions, dimension_offsets):
            return False
        
        # Generate motif
        motif = MotifGenerator.generate_motif(motif_type, length, amplitude, **motif_params)
        
        # Add motif to each specified dimension
        for dim in dimensions:
            actual_start = start_index + dimension_offsets.get(dim, 0)
            actual_end = actual_start + length
            
            if actual_start >= 0 and actual_end <= self.signal_length:
                self.signal[actual_start:actual_end, dim] += motif
        
        # Record motif instance
        instance = MotifInstance(
            motif_type=motif_type,
            start_index=start_index,
            dimensions=dimensions,
            length=length,
            amplitude=amplitude,
            parameters=motif_params,
            dimension_offsets=dimension_offsets,
            family_label=family_label
        )
        self.motif_instances.append(instance)
        
        return True
    
    def add_multiple_motifs(self, n_motifs: int, motif_config: Dict[str, Any],
                           allow_overlaps: bool = False) -> int:
        """
        Add multiple motifs randomly to the signal.
        
        Parameters:
        -----------
        n_motifs : int
            Number of motifs to add
        motif_config : Dict
            Configuration for motif generation:
            {
                'types': List[str] or 'random',  # motif types to sample from
                'length_range': Tuple[int, int],
                'amplitude_range': Tuple[float, float],
                'dimension_range': Tuple[int, int],  # min/max dimensions per motif
                'motif_params': Dict[str, Tuple],  # parameter ranges for each motif type
            }
        allow_overlaps : bool
            If False, don't place motifs where others exist
            
        Returns:
        --------
        int : Number of motifs successfully added
        """
        if self.signal is None:
            raise ValueError("Generate base signal first")
        
        added = 0
        attempts = 0
        max_attempts = n_motifs * 10
        
        while added < n_motifs and attempts < max_attempts:
            attempts += 1
            
            # Sample motif parameters
            motif_type = np.random.choice(motif_config['types'])
            length = np.random.randint(*motif_config['length_range'])
            amplitude = np.random.uniform(*motif_config['amplitude_range'])
            
            # Random dimensions
            n_dims_motif = np.random.randint(*motif_config.get('dimension_range', (1, self.n_dimensions + 1)))
            n_dims_motif = min(n_dims_motif, self.n_dimensions)
            dimensions = list(np.random.choice(self.n_dimensions, n_dims_motif, replace=False))
            
            # Random start index
            start_index = np.random.randint(0, self.signal_length - length)
            
            # Random dimension offsets
            max_offset = motif_config.get('max_dimension_offset', 0)
            dimension_offsets = {dim: np.random.randint(-max_offset, max_offset + 1) 
                               for dim in dimensions}
            
            # Prepare motif params
            params = {}
            if motif_type in motif_config.get('motif_params', {}):
                param_ranges = motif_config['motif_params'][motif_type]
                for param_name, (min_val, max_val) in param_ranges.items():
                    params[param_name] = np.random.uniform(min_val, max_val)
            
            # Try to add motif
            if self.add_motif(motif_type, start_index, dimensions, length,
                            amplitude, dimension_offsets, **params):
                added += 1
        
        return added
    
    def add_multiple_motifs_improved(self, n_motifs: int, motif_config: Dict[str, Any]) -> int:
        """
        Add multiple motifs with specific motif type and dimension constraints.
        All motifs added in this batch will have a family_label and their parameters will be saved.
        
        Parameters:
        -----------
        n_motifs : int
            Number of motifs to add
        motif_config : Dict
            Configuration including:
            - 'types': List with single motif type
            - 'length_range': Tuple[int, int] or single value
            - 'amplitude_range': Tuple[float, float] or single value
            - 'dimension_range': Tuple for dimension count
            - 'max_dimension_offset': int
            - 'motif_params': parameter ranges for the motif type (can be Tuple or single value)
            - 'forced_dimensions': List of dimensions to use (optional)
            
        Returns:
        --------
        int : Number of motifs successfully added
        """
        if self.signal is None:
            raise ValueError("Generate base signal first")
        
        # Create family label for this batch
        family_id = len([k for k in self.motif_families.keys() if k.startswith('family_')])
        family_label = f'family_{family_id}'
        
        # Store the parameters used for this family
        family_params = {
            'motif_type': motif_config['types'][0],
            'length_range': motif_config['length_range'],
            'amplitude_range': motif_config['amplitude_range'],
            'motif_params': motif_config.get('motif_params', {}),
            'dimensions': motif_config.get('forced_dimensions', 'variable'),
            'max_dimension_offset': motif_config.get('max_dimension_offset', 0)
        }
        self.motif_families[family_label] = family_params
        
        added = 0
        attempts = 0
        max_attempts = n_motifs * 10
        
        motif_type = motif_config['types'][0]  # Single type for this method
        forced_dimensions = motif_config.get('forced_dimensions', None)
        
        while added < n_motifs and attempts < max_attempts:
            attempts += 1
            
            # ===== SIMPLE AND ROBUST PARAMETER SAMPLING =====
            # For LENGTH: check if min == max, then use fixed; otherwise sample
            length_range = motif_config['length_range']
            length_min = length_range[0] if isinstance(length_range, tuple) else length_range
            length_max = length_range[1] if isinstance(length_range, tuple) and len(length_range) > 1 else length_range
            
            if length_min == length_max:
                length = int(length_min)
            else:
                length = np.random.randint(int(length_min), int(length_max) + 1)
            
            # For AMPLITUDE: check if min == max, then use fixed; otherwise sample
            amp_range = motif_config['amplitude_range']
            amp_min = amp_range[0] if isinstance(amp_range, tuple) else amp_range
            amp_max = amp_range[1] if isinstance(amp_range, tuple) and len(amp_range) > 1 else amp_range
            
            if amp_min == amp_max:
                amplitude = float(amp_min)
            else:
                amplitude = np.random.uniform(float(amp_min), float(amp_max))
            
            # Determine dimensions
            if forced_dimensions:
                dimensions = forced_dimensions.copy()
            else:
                n_dims_motif = np.random.randint(*motif_config.get('dimension_range', (1, self.n_dimensions + 1)))
                n_dims_motif = min(n_dims_motif, self.n_dimensions)
                dimensions = list(np.random.choice(self.n_dimensions, n_dims_motif, replace=False))
            
            # Random start index
            if length < self.signal_length:
                start_index = np.random.randint(0, max(1, self.signal_length - length))
            else:
                continue
            
            # Random dimension offsets
            max_offset = motif_config.get('max_dimension_offset', 0)
            dimension_offsets = {dim: np.random.randint(-max_offset, max_offset + 1) 
                               for dim in dimensions}
            
            # ===== SIMPLE AND ROBUST MOTIF PARAMETER SAMPLING =====
            # For each parameter, check if min == max, then use fixed; otherwise sample
            params = {}
            if motif_type in motif_config.get('motif_params', {}):
                param_ranges = motif_config['motif_params'][motif_type]
                for param_name, param_range in param_ranges.items():
                    # Handle both tuple ranges and single values
                    param_min = param_range[0] if isinstance(param_range, tuple) else param_range
                    param_max = param_range[1] if isinstance(param_range, tuple) and len(param_range) > 1 else param_range
                    
                    if param_min == param_max:
                        # Fixed parameter
                        params[param_name] = float(param_min)
                    else:
                        # Range: sample uniformly
                        params[param_name] = np.random.uniform(float(param_min), float(param_max))
            
            # Try to add motif with family label
            if self.add_motif(motif_type, start_index, dimensions, length,
                            amplitude, dimension_offsets, family_label=family_label, **params):
                added += 1
        
        return added

    
    def _check_collision_free(self, start_index: int, length: int,
                            dimensions: List[int], dimension_offsets: Dict[int, int]) -> bool:
        """Check if a motif placement collides with existing motifs."""
        for existing in self.motif_instances:
            # Check if dimensions overlap
            if not set(dimensions).isdisjoint(existing.dimensions):
                # Check time overlap
                for dim in dimensions:
                    if dim in existing.dimensions:
                        existing_start = existing.start_index + existing.dimension_offsets.get(dim, 0)
                        existing_end = existing_start + existing.length
                        
                        new_start = start_index + dimension_offsets.get(dim, 0)
                        new_end = new_start + length
                        
                        # Check for overlap
                        if not (new_end <= existing_start or new_start >= existing_end):
                            return False
        
        return True
    
    def get_metadata(self) -> SimulationMetadata:
        """Get metadata about the current simulation."""
        return SimulationMetadata(
            signal_length=self.signal_length,
            n_dimensions=self.n_dimensions,
            base_signal_type=self.base_signal_type,
            base_signal_params=self.base_signal_params,
            noise_levels=self.noise_levels,
            motifs=self.motif_instances,
            motif_families=self.motif_families if self.motif_families else None
        )
    
    def save_simulation(self, filepath: str):
        """Save the signal and metadata to a pickle file."""
        data = {
            'signal': self.signal,
            'metadata': self.get_metadata().to_dict()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load_simulation(cls, filepath: str) -> Tuple['MultidimensionalSimulator', np.ndarray]:
        """Load a simulation from a pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        meta = data['metadata']
        signal = data['signal']
        
        simulator = cls(meta['signal_length'], meta['n_dimensions'], 
                       meta['base_signal_type'])
        simulator.base_signal_params = meta['base_signal_params']
        simulator.noise_levels = meta['noise_levels']
        simulator.signal = signal
        
        # Load motif instances with backward compatibility for family_label
        simulator.motif_instances = []
        for motif_dict in meta['motifs']:
            # Ensure family_label exists (backward compatibility with old saves)
            if 'family_label' not in motif_dict:
                motif_dict['family_label'] = None
            simulator.motif_instances.append(MotifInstance(**motif_dict))
        
        simulator.motif_families = meta.get('motif_families', {}) or {}
        
        return simulator, signal
