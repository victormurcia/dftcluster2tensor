import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional, Any
from scipy.interpolate import interp1d
from scipy.special import erf
import warnings
import periodictable

class HenkeDataProcessor:
    """
    Class to process Henke .nff files and calculate bare atom and compound absorption
    """
    
    def __init__(self, data_directory: str):
        """
        Initialize with directory containing .nff files
        
        Args:
            data_directory: Path to directory containing element .nff files
        """
        self.data_directory = Path(data_directory)
        self.element_data = {}
        self.unified_energy = None
        
    def _get_atomic_weight(self, element_symbol: str) -> float:
        """
        Get atomic weight for an element using periodictable module
        
        Args:
            element_symbol: Element symbol (e.g., 'C', 'O', 'H')
            
        Returns:
            Atomic weight in g/mol
        """
        try:
            element = getattr(periodictable, element_symbol.lower())
            return float(element.mass)
        except AttributeError:
            warnings.warn(f"Element {element_symbol} not found in periodictable, using 1.0")
            return 1.0
    
    def _get_atomic_number(self, element_symbol: str) -> int:
        """
        Get atomic number for an element using periodictable module
        
        Args:
            element_symbol: Element symbol (e.g., 'C', 'O', 'H')
            
        Returns:
            Atomic number
        """
        try:
            element = getattr(periodictable, element_symbol.lower())
            return int(element.number)
        except AttributeError:
            warnings.warn(f"Element {element_symbol} not found in periodictable, using 1")
            return 1
    
    def load_nff_file(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load a single .nff file
        
        Args:
            filename: Name of .nff file (e.g., 'c.nff')
            
        Returns:
            Tuple of (energy, f1, f2) arrays
        """
        filepath = self.data_directory / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")
            
        # Read the file, skipping lines that start with # or are empty
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            energy = float(parts[0])
                            f1 = float(parts[1]) if parts[1] != '-9999.' else np.nan
                            f2 = float(parts[2]) if parts[2] != '-9999.' else np.nan
                            data.append([energy, f1, f2])
                        except ValueError:
                            continue
        
        if not data:
            raise ValueError(f"No valid data found in {filename}")
            
        data = np.array(data)
        return data[:, 0], data[:, 1], data[:, 2]
    
    def load_all_elements(self) -> None:
        """Load all available .nff files in the directory"""
        nff_files = list(self.data_directory.glob("*.nff"))
        
        if not nff_files:
            raise FileNotFoundError(f"No .nff files found in {self.data_directory}")
        
        print(f"Loading {len(nff_files)} element files...")
        
        # First pass: load all data and collect all energies
        all_energies = set()
        
        for nff_file in nff_files:
            element_symbol = nff_file.stem.capitalize()  # e.g., 'c.nff' -> 'C'
            
            try:
                energy, f1, f2 = self.load_nff_file(nff_file.name)
                
                # Get atomic weight using periodictable
                atomic_weight = self._get_atomic_weight(element_symbol)
                
                # Store raw data
                self.element_data[element_symbol] = {
                    'energy_raw': energy,
                    'f1_raw': f1,
                    'f2_raw': f2,
                    'atomic_weight': atomic_weight
                }
                
                # Collect all unique energies
                all_energies.update(energy)
                
                print(f"Loaded {element_symbol}: {len(energy)} energy points, MW = {atomic_weight:.3f} g/mol")
                
            except Exception as e:
                print(f"Warning: Could not load {nff_file}: {e}")
                continue
        
        # Create unified energy grid
        self.unified_energy = np.array(sorted(all_energies))
        print(f"Created unified energy grid with {len(self.unified_energy)} points")
        
        # Second pass: interpolate all data to unified grid
        self._interpolate_to_unified_grid()
    
    def _interpolate_to_unified_grid(self) -> None:
        """Interpolate all element data to the unified energy grid"""
        print("Interpolating all elements to unified energy grid...")
        
        for element, data in self.element_data.items():
            # Remove NaN values for interpolation
            valid_mask = ~(np.isnan(data['f1_raw']) | np.isnan(data['f2_raw']))
            
            if np.sum(valid_mask) < 2:
                print(f"Warning: Not enough valid data for {element}")
                continue
                
            energy_valid = data['energy_raw'][valid_mask]
            f1_valid = data['f1_raw'][valid_mask]
            f2_valid = data['f2_raw'][valid_mask]
            
            # Create interpolation functions
            try:
                f1_interp = interp1d(energy_valid, f1_valid, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
                f2_interp = interp1d(energy_valid, f2_valid, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
                
                # Interpolate to unified grid
                data['f1'] = f1_interp(self.unified_energy)
                data['f2'] = f2_interp(self.unified_energy)
                
            except Exception as e:
                print(f"Warning: Could not interpolate {element}: {e}")
                continue

    def calculate_compound_absorption_from_dict(self, composition_dict: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate mass absorption coefficient for a compound using composition dictionary
        
        Args:
            composition_dict: Dictionary mapping element symbols to counts
                             e.g., {'C': 32, 'H': 16, 'Cu': 1, 'N': 8} for CuPc
            
        Returns:
            Tuple of (energy, mass_absorption_coefficient) in [eV, cm²/g]
        """
        if not composition_dict:
            raise ValueError("Composition dictionary cannot be empty")
        
        print(f"Calculating absorption for composition: {composition_dict}")
        
        # Calculate weighted sum of f2 values and total molecular weight
        f2_sum = np.zeros_like(self.unified_energy)
        molecular_weight = 0.0
        z_star = 0.0  # Relativistic correction to Z
        
        for element, count in composition_dict.items():
            if element not in self.element_data:
                raise ValueError(f"Element {element} not found in loaded data")
                
            data = self.element_data[element]
            
            if 'f2' not in data:
                raise ValueError(f"Interpolated data not available for {element}")
                
            f2_sum += count * data['f2']
            molecular_weight += count * data['atomic_weight']
            
            # Calculate relativistic correction using periodictable
            z = self._get_atomic_number(element)
            z_star += count * (z - (z/82.5)**2.37)
        
        # Physical constants
        Na = 6.0221415e23  # Avogadro's number [atoms/mol]
        re = 2.81794e-13   # Classical electron radius [cm]
        
        # Calculate wavelength [cm]
        lambda_cm = 1.23984e-4 / self.unified_energy
        
        # Calculate compound mass absorption coefficient [cm²/g]
        mu_compound = 2 * re * lambda_cm * Na * f2_sum / molecular_weight
        
        print(f"Compound: MW = {molecular_weight:.2f} g/mol, Z* = {z_star:.1f}")
        
        return self.unified_energy.copy(), mu_compound

class AbsorptionEdgeBuilder:
    """
    A comprehensive pipeline for building theoretical absorption edges from DFT calculations.
    """
    
    def __init__(self):
        self.preedge_coeffs = None
        self.postedge_coeffs = None
        self.individual_edges = {}
        self.total_edge = None
        
    @staticmethod
    def Gstep(x: np.ndarray, x0: float, width: float) -> np.ndarray:
        """Gaussian step function for smooth transitions."""
        c = 2 * np.sqrt(2)
        return 0.5 + 0.5 * erf((x - x0) / (width / c))
    
    @staticmethod
    def polynomial_fit(energy: np.ndarray, mu: np.ndarray, degree: int = 2) -> np.ndarray:
        """Fit polynomial to data."""
        return np.polyfit(energy, mu, degree)
    
    @staticmethod
    def evaluate_polynomial(energy: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate polynomial at given energy points."""
        return np.polyval(coeffs, energy)
    
    def fit_baseline_polynomials(self, mu_energy: np.ndarray, mu: np.ndarray,
                                preedge_range: list, postedge_range: list,
                                poly_degree: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit polynomial baselines to preedge and postedge regions.
        
        Returns:
            tuple: (preedge_coeffs, postedge_coeffs)
        """
        # Create masks for fitting regions
        preedge_mask = (mu_energy >= preedge_range[0]) & (mu_energy <= preedge_range[1])
        postedge_mask = (mu_energy >= postedge_range[0]) & (mu_energy <= postedge_range[1])
        
        # Fit polynomials
        self.preedge_coeffs = self.polynomial_fit(mu_energy[preedge_mask], 
                                                mu[preedge_mask], poly_degree)
        self.postedge_coeffs = self.polynomial_fit(mu_energy[postedge_mask], 
                                                 mu[postedge_mask], poly_degree)
        
        return self.preedge_coeffs, self.postedge_coeffs
    
    def build_absorption_edge(self, df: pd.DataFrame, energy_array: np.ndarray, 
                            mu_energy: np.ndarray, mu: np.ndarray,
                            preedge_range: list, postedge_range: list,
                            stepWid1: float, stepWid2: float, stepDecay: float,
                            poly_degree: int = 2, return_individual: bool = False,
                            debug: bool = False) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Build theoretical absorption edge from DFT ionization potentials.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with atomic data including corrected ionization potentials
        energy_array : np.ndarray
            Energy grid for calculation
        mu_energy : np.ndarray
            Experimental energy points
        mu : np.ndarray
            Experimental absorption coefficient
        preedge_range : list
            [min, max] energy range for preedge fitting
        postedge_range : list
            [min, max] energy range for postedge fitting
        stepWid1, stepWid2 : float
            Step width parameters (min and max)
        stepDecay : float
            Exponential decay parameter for step heights
        poly_degree : int
            Degree of polynomial for baseline fitting
        return_individual : bool
            Whether to return individual atomic contributions
        debug : bool
            Whether to print debug information
        
        Returns:
        --------
        tuple: (total_edge, individual_edges_dict or None)
        """
        
        # Fit baseline polynomials
        self.fit_baseline_polynomials(mu_energy, mu, preedge_range, 
                                    postedge_range, poly_degree)
        
        # Calculate corrected ionization potentials
        IPcorr = df['TP_Ionization potential (eV)'] + df['Energy_Correction (eV)']
        
        # Find min and max corrected IPs for step width calculation
        V_minloc = IPcorr.idxmin()
        V_maxloc = IPcorr.idxmax()
        IP_min = IPcorr[V_minloc]
        IP_max = IPcorr[V_maxloc]
        
        if debug:
            print(f"IP range: {IP_min:.2f} - {IP_max:.2f} eV")
            print(f"Number of atoms: {len(df)}")
        
        # Calculate baseline polynomials across full energy range
        preedge_baseline = self.evaluate_polynomial(energy_array, self.preedge_coeffs)
        postedge_baseline = self.evaluate_polynomial(energy_array, self.postedge_coeffs)
        
        # CORRECTED APPROACH: Start with preedge baseline, 
        # then transition to postedge baseline through atomic steps
        self.total_edge = preedge_baseline.copy()
        self.individual_edges = {}
        
        # Sort atoms by ionization potential for consistent processing
        df_sorted = df.sort_values('TP_Ionization potential (eV)')
        IPcorr_sorted = IPcorr[df_sorted.index]
        
        # Build step edge for each atom
        for idx, (orig_idx, row) in enumerate(df_sorted.iterrows()):
            E0 = IPcorr_sorted.loc[orig_idx]
            
            # Calculate step width for this atom (linear interpolation)
            if IP_max != IP_min:
                width_factor = (E0 - IP_min) / (IP_max - IP_min)
                stepWid = stepWid1 + (stepWid2 - stepWid1) * width_factor
            else:
                stepWid = stepWid1
            
            # Calculate step height: difference between post and pre edge baselines at E0
            preedge_at_E0 = self.evaluate_polynomial(E0, self.preedge_coeffs)
            postedge_at_E0 = self.evaluate_polynomial(E0, self.postedge_coeffs)
            
            # The step height should be the LOCAL contribution of this atom
            # We need to calculate how much THIS atom contributes to the total step
            total_step_height = postedge_at_E0 - preedge_at_E0
            
            # Distribute the total step height among all atoms
            # Simple approach: equal distribution (can be refined with oscillator strengths)
            atom_step_height = total_step_height / len(df)
            
            # Apply decay to step height for energies above E0
            decay_factor = np.where(energy_array > E0, 
                                   np.exp(-stepDecay * (energy_array - E0)), 
                                   1.0)
            effective_step_height = atom_step_height * decay_factor
            
            # Create step function
            step_func = self.Gstep(energy_array, E0, stepWid)
            
            # Calculate this atom's contribution
            atom_contribution = effective_step_height * step_func
            
            # Store individual edge (preedge + this atom's step)
            self.individual_edges[row['Atom']] = preedge_baseline + atom_contribution
            
            # Add this atom's contribution to total
            self.total_edge += atom_contribution
            
            if debug:
                print(f"Atom {row['Atom']}: E0={E0:.2f} eV, width={stepWid:.2f} eV, "
                      f"height={atom_step_height:.4f}")
        
        # Verify edge matching at key points
        if debug:
            self._debug_edge_matching(energy_array, preedge_range, postedge_range)
        
        if return_individual:
            return self.total_edge, self.individual_edges
        else:
            return self.total_edge, None
    
    def _debug_edge_matching(self, energy_array: np.ndarray, 
                           preedge_range: list, postedge_range: list):
        """Debug function to check edge matching."""
        # Check preedge matching
        preedge_energy = preedge_range[1]
        preedge_idx = np.argmin(np.abs(energy_array - preedge_energy))
        preedge_expected = self.evaluate_polynomial(preedge_energy, self.preedge_coeffs)
        preedge_actual = self.total_edge[preedge_idx]
        
        # Check postedge matching
        postedge_energy = postedge_range[0]
        postedge_idx = np.argmin(np.abs(energy_array - postedge_energy))
        postedge_expected = self.evaluate_polynomial(postedge_energy, self.postedge_coeffs)
        postedge_actual = self.total_edge[postedge_idx]
        
        print(f"\nEdge matching check:")
        print(f"Preedge at {preedge_energy} eV: expected={preedge_expected:.4f}, "
              f"actual={preedge_actual:.4f}, diff={preedge_actual-preedge_expected:.4f}")
        print(f"Postedge at {postedge_energy} eV: expected={postedge_expected:.4f}, "
              f"actual={postedge_actual:.4f}, diff={postedge_actual-postedge_expected:.4f}")
    
    def scale_to_experimental(self, energy_array: np.ndarray, mu: np.ndarray, 
                            mu_energy: np.ndarray, preedge_energy: float, 
                            postedge_energy: float) -> Tuple[np.ndarray, float, float]:
        """
        Scale the theoretical edge to match experimental data at two reference points.
        
        Returns:
            tuple: (scaled_edge, scale_factor, offset)
        """
        if self.total_edge is None:
            raise ValueError("Must build edge first using build_absorption_edge()")
        
        # Find closest indices in energy arrays
        preedge_idx_total = np.argmin(np.abs(energy_array - preedge_energy))
        postedge_idx_total = np.argmin(np.abs(energy_array - postedge_energy))
        
        preedge_idx_mu = np.argmin(np.abs(mu_energy - preedge_energy))
        postedge_idx_mu = np.argmin(np.abs(mu_energy - postedge_energy))
        
        # Get intensities at these points
        total_preedge = self.total_edge[preedge_idx_total]
        total_postedge = self.total_edge[postedge_idx_total]
        
        mu_preedge = mu[preedge_idx_mu]
        mu_postedge = mu[postedge_idx_mu]
        
        # Calculate scaling factors: scaled = a * total + b
        a = (mu_postedge - mu_preedge) / (total_postedge - total_preedge)
        b = mu_preedge - a * total_preedge
        
        # Apply scaling
        scaled_edge = a * self.total_edge + b
        
        return scaled_edge, a, b
    
    def plot_comparison(self, energy_array: np.ndarray, mu_energy: np.ndarray, 
                       mu: np.ndarray, scaled_edge: Optional[np.ndarray] = None,
                       title: str = "Absorption Edge Comparison", xrange = (270,360)):
        """
        Plot comparison between experimental and theoretical edges.
        """
        plt.figure(figsize=(12, 8))
        
        # Plot experimental data
        plt.plot(mu_energy, mu, 'k-', linewidth=1.5, label='Experimental', alpha=0.8)
        
        # Plot theoretical edge
        edge_to_plot = scaled_edge if scaled_edge is not None else self.total_edge
        plt.plot(energy_array, edge_to_plot, 'b-', linewidth=2, 
                label='Theoretical Step Edge')
        
        # Plot baselines for reference
        if self.preedge_coeffs is not None and self.postedge_coeffs is not None:
            preedge_baseline = self.evaluate_polynomial(energy_array, self.preedge_coeffs)
            postedge_baseline = self.evaluate_polynomial(energy_array, self.postedge_coeffs)
            
            plt.plot(energy_array, preedge_baseline, '--', color='gray', alpha=0.6,
                    label='Preedge baseline')
            plt.plot(energy_array, postedge_baseline, '--', color='orange', alpha=0.6,
                    label='Postedge baseline')
        
        plt.xlabel('Energy (eV)')
        plt.ylabel('Absorption')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(xrange)
        plt.show()

def create_dft_step_edge_pipeline(df_path, atom_dict, energy_min=240, energy_max=360, 
                            n_points=2000, preedge_range=[240, 284.1], 
                            postedge_range=[284.3, 360], stepWid1=0.6, stepWid2=12.0, 
                            stepDecay=0.0075, poly_degree=2, scale_to_experimental=True, 
                            plot_results=True, debug=False):
    """Complete pipeline for creating step edge from DFT calculations."""
    
    # Initialize processor with your .nff files directory
    henke_processor = HenkeDataProcessor("Element Library")

    # Load all available element data
    henke_processor.load_all_elements()

    # Calculate bare atom absorption for carbon
    mu_energy, mu_mol = henke_processor.calculate_compound_absorption_from_dict(atom_dict)

    # Load data
    df_step = pd.read_csv(df_path)
    energy_array = np.linspace(energy_min, energy_max, n_points)
    
    # Initialize the edge builder
    dft_step_builder = AbsorptionEdgeBuilder()
    
    # Build the absorption edge
    total_edge, individual_edges = dft_step_builder.build_absorption_edge(
        df_step, energy_array, mu_energy, mu_mol,
        preedge_range, postedge_range,
        stepWid1, stepWid2, stepDecay,
        poly_degree=poly_degree,
        return_individual=True,
        debug=debug
    )
    
    # Scale to experimental data if requested
    scaled_edge = None
    scale_params = None
    if scale_to_experimental:
        preedge_ref = (preedge_range[0] + preedge_range[1]) / 2
        postedge_ref = (postedge_range[0] + postedge_range[1]) / 2
        
        scaled_edge, scale_factor, offset = dft_step_builder.scale_to_experimental(
            energy_array, mu_mol, mu_energy, preedge_ref, postedge_ref
        )
        scale_params = {'scale_factor': scale_factor, 'offset': offset}
        
        if debug:
            print(f"\nScaling parameters: a={scale_factor:.4f}, b={offset:.4f}")
    
    # Create plots if requested
    if plot_results:
        dft_step_builder.plot_comparison(energy_array, mu_energy, mu_mol, scaled_edge,
                              "Absorption Edge: Experimental vs Theoretical")
    
    # Prepare results dictionary
    results = {
        'energy_array': energy_array,
        'total_edge': total_edge,
        'scaled_edge': scaled_edge,
        'individual_edges': individual_edges,
        'preedge_coeffs': dft_step_builder.preedge_coeffs,
        'postedge_coeffs': dft_step_builder.postedge_coeffs,
        'scale_params': scale_params,
        'parameters': {
            'preedge_range': preedge_range,
            'postedge_range': postedge_range,
            'stepWid1': stepWid1,
            'stepWid2': stepWid2,
            'stepDecay': stepDecay,
            'poly_degree': poly_degree
        }
    }