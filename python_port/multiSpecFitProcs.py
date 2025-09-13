import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class MultiSpectrumGaussianFitter:
    """
    Simultaneous fitting of multiple spectra using Gaussian peaks with staged optimization.
    
    Stages:
    1. Optimize only amplitudes (OS)
    2. Optimize amplitudes and positions (E)
    3. Optimize amplitudes, positions, and widths
    """
    
    def __init__(self, parameters_file: str, experimental_file: str):
        """
        Initialize the fitter with Gaussian parameters and experimental data.
        
        Args:
            parameters_file: CSV file with Gaussian parameters
            experimental_file: CSV file with experimental spectra
        """
        self.load_parameters(parameters_file)
        self.load_experimental_data(experimental_file)
        self.setup_initial_parameters()
        
    def load_parameters(self, filename: str):
        """Load Gaussian parameters from CSV file."""
        self.params_df = pd.read_csv(filename)
        self.n_gaussians = len(self.params_df)
        
        # Extract relevant parameters
        self.initial_positions = self.params_df['E'].values
        self.initial_widths = self.params_df['width'].values
        self.initial_amplitudes = self.params_df['OS'].values
        
        print(f"Loaded {self.n_gaussians} Gaussian peaks")
        print(f"Energy range: {self.initial_positions.min():.2f} - {self.initial_positions.max():.2f} eV")
        
    def load_experimental_data(self, filename: str):
        """Load experimental spectra data."""
        self.exp_df = pd.read_csv(filename)
        
        # Identify energy and intensity columns
        self.energy_columns = [col for col in self.exp_df.columns if col.startswith('E_')]
        self.intensity_columns = [col for col in self.exp_df.columns if col.startswith('CuPc_CuI_')]
        
        self.angles = [col.split('_')[1] for col in self.energy_columns]
        self.n_spectra = len(self.angles)
        
        # Store experimental data for each spectrum
        self.exp_energies = {}
        self.exp_intensities = {}
        
        for i, angle in enumerate(self.angles):
            energy_col = self.energy_columns[i]
            intensity_col = self.intensity_columns[i]
            
            # Remove NaN values
            mask = ~(pd.isna(self.exp_df[energy_col]) | pd.isna(self.exp_df[intensity_col]))
            self.exp_energies[angle] = self.exp_df.loc[mask, energy_col].values
            self.exp_intensities[angle] = self.exp_df.loc[mask, intensity_col].values
            
        print(f"Loaded {self.n_spectra} experimental spectra at angles: {self.angles}")
        
    def setup_initial_parameters(self):
        """Set up initial parameter arrays for optimization."""
        # Parameter organization: [amplitudes, positions, widths] for each spectrum
        # Each spectrum has its own amplitude scaling, but shares positions and widths
        
        # For each spectrum, we have separate amplitudes
        self.n_params_per_spectrum = self.n_gaussians  # amplitudes only
        self.n_shared_params = 2 * self.n_gaussians  # positions + widths
        
        # Total parameters: amplitudes for each spectrum + shared positions + shared widths
        self.total_params = self.n_spectra * self.n_gaussians + self.n_shared_params
        
    def gaussian(self, x: np.ndarray, position: float, width: float, amplitude: float) -> np.ndarray:
        """
        Calculate Gaussian peak values.
        
        Args:
            x: Energy array
            position: Peak position (eV)
            width: Peak width (FWHM)
            amplitude: Peak amplitude (oscillator strength)
            
        Returns:
            Gaussian values
        """
        sigma = width / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
        return amplitude * np.exp(-0.5 * ((x - position) / sigma) ** 2)
        
    def calculate_spectrum(self, energies: np.ndarray, amplitudes: np.ndarray, 
                          positions: np.ndarray, widths: np.ndarray) -> np.ndarray:
        """
        Calculate complete spectrum from all Gaussian peaks.
        
        Args:
            energies: Energy array
            amplitudes: Amplitude array for all peaks
            positions: Position array for all peaks  
            widths: Width array for all peaks
            
        Returns:
            Calculated spectrum intensity
        """
        spectrum = np.zeros_like(energies)
        for i in range(self.n_gaussians):
            spectrum += self.gaussian(energies, positions[i], widths[i], amplitudes[i])
        return spectrum
        
    def unpack_parameters(self, params: np.ndarray, stage: str) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Unpack parameter array based on optimization stage.
        
        Args:
            params: Flattened parameter array
            stage: Optimization stage ('amplitudes', 'positions', 'widths')
            
        Returns:
            amplitudes_dict: Dictionary of amplitudes for each spectrum
            positions: Shared positions array
            widths: Shared widths array
        """
        amplitudes_dict = {}
        
        if stage == 'amplitudes':
            # Only amplitudes are being optimized
            for i, angle in enumerate(self.angles):
                start_idx = i * self.n_gaussians
                end_idx = (i + 1) * self.n_gaussians
                amplitudes_dict[angle] = params[start_idx:end_idx]
            positions = self.initial_positions.copy()
            widths = self.initial_widths.copy()
            
        elif stage == 'positions':
            # Amplitudes and positions are being optimized
            for i, angle in enumerate(self.angles):
                start_idx = i * self.n_gaussians
                end_idx = (i + 1) * self.n_gaussians
                amplitudes_dict[angle] = params[start_idx:end_idx]
            
            pos_start = self.n_spectra * self.n_gaussians
            pos_end = pos_start + self.n_gaussians
            positions = params[pos_start:pos_end]
            widths = self.initial_widths.copy()
            
        elif stage == 'widths':
            # All parameters are being optimized
            for i, angle in enumerate(self.angles):
                start_idx = i * self.n_gaussians
                end_idx = (i + 1) * self.n_gaussians
                amplitudes_dict[angle] = params[start_idx:end_idx]
                
            pos_start = self.n_spectra * self.n_gaussians
            pos_end = pos_start + self.n_gaussians
            positions = params[pos_start:pos_end]
            
            width_start = pos_end
            width_end = width_start + self.n_gaussians
            widths = params[width_start:width_end]
            
        return amplitudes_dict, positions, widths
        
    def objective_function(self, params: np.ndarray, stage: str) -> float:
        """
        Objective function for least squares fitting.
        
        Args:
            params: Parameter array
            stage: Optimization stage
            
        Returns:
            Sum of squared residuals
        """
        amplitudes_dict, positions, widths = self.unpack_parameters(params, stage)
        
        total_residual = 0.0
        
        for angle in self.angles:
            # Calculate theoretical spectrum
            calc_spectrum = self.calculate_spectrum(
                self.exp_energies[angle], 
                amplitudes_dict[angle], 
                positions, 
                widths
            )
            
            # Calculate residuals
            residuals = self.exp_intensities[angle] - calc_spectrum
            total_residual += np.sum(residuals ** 2)
            
        return total_residual
        
    def setup_bounds_and_constraints(self, stage: str) -> Bounds:
        """
        Set up parameter bounds based on optimization stage.
        
        Args:
            stage: Optimization stage
            
        Returns:
            Bounds object for scipy optimization
        """
        lower_bounds = []
        upper_bounds = []
        
        # Amplitude bounds (must be positive)
        for i in range(self.n_spectra):
            lower_bounds.extend([0.0] * self.n_gaussians)
            upper_bounds.extend([np.inf] * self.n_gaussians)
            
        if stage in ['positions', 'widths']:
            # Position bounds (initial ± 2*width)
            for i in range(self.n_gaussians):
                lower_bound = self.initial_positions[i] - 2 * self.initial_widths[i]
                upper_bound = self.initial_positions[i] + 2 * self.initial_widths[i]
                lower_bounds.append(lower_bound)
                upper_bounds.append(upper_bound)
                
        if stage == 'widths':
            # Width bounds (positive, not greater than 2*initial)
            for i in range(self.n_gaussians):
                lower_bounds.append(1e-6)  # Small positive value
                upper_bounds.append(2 * self.initial_widths[i])
                
        return Bounds(lower_bounds, upper_bounds)
        
    def get_initial_parameters(self, stage: str) -> np.ndarray:
        """
        Get initial parameter array for optimization stage.
        
        Args:
            stage: Optimization stage
            
        Returns:
            Initial parameter array
        """
        params = []
        
        # Add amplitudes for each spectrum (start with initial values)
        for i in range(self.n_spectra):
            params.extend(self.initial_amplitudes)
            
        if stage in ['positions', 'widths']:
            # Add positions
            params.extend(self.initial_positions)
            
        if stage == 'widths':
            # Add widths
            params.extend(self.initial_widths)
            
        return np.array(params)
        
    def fit_stage(self, stage: str, initial_params: np.ndarray = None) -> Dict:
        """
        Perform fitting for a specific stage.
        
        Args:
            stage: Optimization stage
            initial_params: Initial parameters (if None, use default)
            
        Returns:
            Dictionary with optimization results
        """
        print(f"\n=== Stage: {stage.upper()} ===")
        
        if initial_params is None:
            initial_params = self.get_initial_parameters(stage)
            
        bounds = self.setup_bounds_and_constraints(stage)
        
        print(f"Optimizing {len(initial_params)} parameters...")
        print(f"Initial objective: {self.objective_function(initial_params, stage):.6f}")
        
        # Perform optimization
        result = minimize(
            fun=self.objective_function,
            x0=initial_params,
            args=(stage,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        print(f"Final objective: {result.fun:.6f}")
        print(f"Optimization success: {result.success}")
        print(f"Number of iterations: {result.nit}")
        
        return {
            'result': result,
            'stage': stage,
            'final_params': result.x,
            'objective_value': result.fun,
            'success': result.success
        }
        
    def fit_all_stages(self) -> Dict:
        """
        Perform complete staged fitting: amplitudes -> positions -> widths.
        
        Returns:
            Dictionary with all fitting results
        """
        print("Starting staged optimization...")
        
        results = {}
        
        # Stage 1: Optimize amplitudes only
        stage1_result = self.fit_stage('amplitudes')
        results['stage1'] = stage1_result
        
        # Stage 2: Optimize amplitudes and positions
        # Use stage 1 results as starting point
        stage2_initial = self.get_initial_parameters('positions')
        # Update amplitudes from stage 1
        n_amp_params = self.n_spectra * self.n_gaussians
        stage2_initial[:n_amp_params] = stage1_result['final_params'][:n_amp_params]
        
        stage2_result = self.fit_stage('positions', stage2_initial)
        results['stage2'] = stage2_result
        
        # Stage 3: Optimize all parameters
        # Use stage 2 results as starting point
        stage3_initial = self.get_initial_parameters('widths')
        stage3_initial[:len(stage2_result['final_params'])] = stage2_result['final_params']
        
        stage3_result = self.fit_stage('widths', stage3_initial)
        results['stage3'] = stage3_result
        
        # Store final parameters
        self.final_parameters = stage3_result['final_params']
        self.final_amplitudes, self.final_positions, self.final_widths = \
            self.unpack_parameters(self.final_parameters, 'widths')
            
        return results
        
    def calculate_fit_statistics(self) -> Dict:
        """
        Calculate fitting statistics for final parameters.
        
        Returns:
            Dictionary with fit statistics
        """
        if not hasattr(self, 'final_parameters'):
            raise ValueError("Must run fitting first!")
            
        stats = {}
        total_points = 0
        total_ss_res = 0
        total_ss_tot = 0
        
        for angle in self.angles:
            # Calculate fitted spectrum
            fitted = self.calculate_spectrum(
                self.exp_energies[angle],
                self.final_amplitudes[angle],
                self.final_positions,
                self.final_widths
            )
            
            # Calculate statistics
            residuals = self.exp_intensities[angle] - fitted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((self.exp_intensities[angle] - np.mean(self.exp_intensities[angle])) ** 2)
            
            n_points = len(self.exp_intensities[angle])
            rmse = np.sqrt(ss_res / n_points)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            stats[f'angle_{angle}'] = {
                'n_points': n_points,
                'rmse': rmse,
                'r_squared': r_squared,
                'ss_res': ss_res,
                'ss_tot': ss_tot
            }
            
            total_points += n_points
            total_ss_res += ss_res
            total_ss_tot += ss_tot
            
        # Overall statistics
        stats['overall'] = {
            'total_points': total_points,
            'total_rmse': np.sqrt(total_ss_res / total_points),
            'total_r_squared': 1 - (total_ss_res / total_ss_tot) if total_ss_tot > 0 else 0,
            'n_parameters': len(self.final_parameters)
        }
        
        return stats
        
    def plot_results(self, save_filename: str = None):
        """
        Plot experimental vs fitted spectra for all angles.
        
        Args:
            save_filename: Optional filename to save plot
        """
        if not hasattr(self, 'final_parameters'):
            raise ValueError("Must run fitting first!")
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, angle in enumerate(self.angles):
            ax = axes[i]
            
            # Plot experimental data
            ax.plot(self.exp_energies[angle], self.exp_intensities[angle], 
                   'bo', markersize=3, alpha=0.6, label='Experimental')
            
            # Plot fitted spectrum
            fitted = self.calculate_spectrum(
                self.exp_energies[angle],
                self.final_amplitudes[angle],
                self.final_positions,
                self.final_widths
            )
            ax.plot(self.exp_energies[angle], fitted, 'r-', linewidth=2, label='Fitted')
            
            # Plot individual Gaussians (optional - can be commented out for clarity)
            """
            for j in range(self.n_gaussians):
                gaussian_contrib = self.gaussian(
                    self.exp_energies[angle],
                    self.final_positions[j],
                    self.final_widths[j],
                    self.final_amplitudes[angle][j]
                )
                if np.max(gaussian_contrib) > 0.01 * np.max(fitted):  # Only show significant peaks
                    ax.plot(self.exp_energies[angle], gaussian_contrib, 'g--', alpha=0.5, linewidth=1)
            """
            
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel('Intensity')
            ax.set_title(f'Angle: {angle}°')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_filename:
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {save_filename}")
            
        plt.show()
        
    def export_results(self, filename: str = 'fitting_results.csv'):
        """
        Export final fitted parameters to CSV file.
        
        Args:
            filename: Output filename
        """
        if not hasattr(self, 'final_parameters'):
            raise ValueError("Must run fitting first!")
            
        # Create results dataframe
        results_data = []
        
        for i in range(self.n_gaussians):
            row = {
                'peak_id': i,
                'initial_position': self.initial_positions[i],
                'final_position': self.final_positions[i],
                'position_change': self.final_positions[i] - self.initial_positions[i],
                'initial_width': self.initial_widths[i],
                'final_width': self.final_widths[i],
                'width_change': self.final_widths[i] - self.initial_widths[i],
                'initial_amplitude': self.initial_amplitudes[i]
            }
            
            # Add final amplitudes for each angle
            for angle in self.angles:
                row[f'final_amplitude_{angle}'] = self.final_amplitudes[angle][i]
                row[f'amplitude_change_{angle}'] = (self.final_amplitudes[angle][i] - 
                                                   self.initial_amplitudes[i])
                
            results_data.append(row)
            
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")
        
    def print_summary(self):
        """Print summary of fitting results."""
        if not hasattr(self, 'final_parameters'):
            print("No fitting results available. Run fit_all_stages() first.")
            return
            
        print("\n" + "="*50)
        print("FITTING SUMMARY")
        print("="*50)
        
        # Parameter changes
        pos_changes = self.final_positions - self.initial_positions
        width_changes = self.final_widths - self.initial_widths
        
        print(f"\nPosition changes:")
        print(f"  Mean: {np.mean(pos_changes):.4f} eV")
        print(f"  Std:  {np.std(pos_changes):.4f} eV")
        print(f"  Range: {np.min(pos_changes):.4f} to {np.max(pos_changes):.4f} eV")
        
        print(f"\nWidth changes:")
        print(f"  Mean: {np.mean(width_changes):.4f} eV")
        print(f"  Std:  {np.std(width_changes):.4f} eV")
        print(f"  Range: {np.min(width_changes):.4f} to {np.max(width_changes):.4f} eV")
        
        # Amplitude statistics for each angle
        print(f"\nAmplitude statistics by angle:")
        for angle in self.angles:
            amp_changes = self.final_amplitudes[angle] - self.initial_amplitudes
            print(f"  {angle}°:")
            print(f"    Mean change: {np.mean(amp_changes):.4f}")
            print(f"    Total amplitude: {np.sum(self.final_amplitudes[angle]):.4f}")
            
        # Fit statistics
        stats = self.calculate_fit_statistics()
        print(f"\nOverall fit quality:")
        print(f"  R²: {stats['overall']['total_r_squared']:.4f}")
        print(f"  RMSE: {stats['overall']['total_rmse']:.6f}")
        print(f"  Total data points: {stats['overall']['total_points']}")
        print(f"  Parameters fitted: {stats['overall']['n_parameters']}")