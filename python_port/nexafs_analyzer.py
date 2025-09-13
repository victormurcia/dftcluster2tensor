import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from numba import jit, prange
import warnings
import time
from pathlib import Path
import copy
from overlapProcs import calculate_overlap_matrix_hybrid, calculate_overlap_matrix_numba, gaussian_numba
from clusteringProcs import fast_sequential_clustering, fast_skip_tolerant_clustering
from stepEdgeProcs import HenkeDataProcessor, AbsorptionEdgeBuilder
warnings.filterwarnings('ignore')

class NEXAFSAnalyzer:
    """
    Complete NEXAFS analysis pipeline class with iterative processing capability.
    
    This class encapsulates the entire analysis workflow including data preprocessing,
    overlap calculation, clustering, peak combination, and iterative analysis until convergence.
    """
    
    def __init__(self, data_path=None, df=None):
        """
        Initialize the analyzer with data.
        
        Parameters:
        -----------
        data_path : str or Path, optional
            Path to CSV file containing NEXAFS data
        df : pandas.DataFrame, optional
            DataFrame with NEXAFS data (alternative to data_path)
        """
        if data_path is not None:
            self.df_raw = pd.read_csv(data_path)
        elif df is not None:
            self.df_raw = df.copy()
        else:
            raise ValueError("Either data_path or df must be provided")
        
        self.df_processed = None

    def filter_by_energy(self, energy_threshold=320):
        """Remove transitions above the energy threshold."""
        print(f"Original data: {len(self.df_raw)} transitions")
        df_filtered = self.df_raw[self.df_raw['E'] <= energy_threshold].copy()
        print(f"After energy filter (E <= {energy_threshold}): {len(df_filtered)} transitions")
        return df_filtered

    def create_normalized_os(self, df):
        """Create normalized_OS column based on maximum OS value."""
        df = df.copy()
        max_os = df['OS'].max()
        df['normalized_OS'] = df['OS'] / max_os
        return df

    def filter_by_os_threshold(self, df, os_threshold_percent=None):
        """Filter transitions based on normalized OS threshold."""
        if os_threshold_percent is None:
            return df
        
        df_filtered = df.copy()
        os_threshold = os_threshold_percent / 100.0
        df_filtered = df_filtered[df_filtered['normalized_OS'] >= os_threshold].copy()
        print(f"After {os_threshold_percent}% OS threshold: {len(df_filtered)} transitions")
        return df_filtered
    
    def calculate_overlap_matrix(self, method = 'numba'):
        """Calculate overlap matrix for all peak pairs in df_processed."""
        if not hasattr(self, 'df_processed') or self.df_processed is None:
            raise ValueError("No processed data available. Run preprocessing first.")
                
        energies = self.df_processed['E'].values
        widths = self.df_processed['width'].values  
        amplitudes = self.df_processed['OS'].values
        
        print(f"Calculating overlap matrix for {len(energies)} peaks...")
        
        if method == 'numba':
            overlap_matrix = calculate_overlap_matrix_numba(energies, widths, amplitudes)
        elif method == 'hybrid':
            overlap_matrix = calculate_overlap_matrix_hybrid(energies, widths, amplitudes)
        
        self.overlap_matrix = overlap_matrix
        print(f"Overlap matrix calculated: {overlap_matrix.shape}")
        
        return overlap_matrix
    
    def create_nexafs_spectrum(self, df, energy_range=None, n_points=2000):
        """
        Create a NEXAFS spectrum by summing Gaussian peaks from dataframe.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing peaks with columns 'E' (energy), 'OS' (amplitude), 'width' (sigma)
        energy_range : tuple, optional
            (min_energy, max_energy) range for spectrum. If None, auto-calculated from data
        n_points : int
            Number of points in the output spectrum
            
        Returns:
        --------
        energy_axis : numpy.ndarray
            Energy values for the spectrum
        intensity : numpy.ndarray
            Total intensity at each energy point
        """
        
        if energy_range is None:
            # Auto-calculate range with padding
            min_energy = df['E'].min() - 3 * df['width'].max()
            max_energy = df['E'].max() + 3 * df['width'].max()
        else:
            min_energy, max_energy = energy_range
        
        # Create energy axis
        energy_axis = np.linspace(min_energy, max_energy, n_points)
        
        # Initialize intensity array
        intensity = np.zeros(n_points)
        
        # Sum all Gaussian peaks
        for _, peak in df.iterrows():
            peak_intensity = gaussian_numba(energy_axis, peak['OS'], peak['E'], peak['width'])
            intensity += peak_intensity
        
        return energy_axis, intensity
    
    def cluster_peaks(self, overlap_threshold=50, method='sequential', n_skipped=1):
        """
        Cluster peaks based on overlap matrix using specified clustering method.
        
        Parameters:
        -----------
        overlap_threshold : float
            Minimum overlap percentage to group peaks into same cluster
        method : str
            Clustering method: 'sequential' or 'skip_tolerant'
        n_skipped : int
            For skip_tolerant method: number of peaks to skip before terminating cluster
            
        Returns:
        --------
        clusters : list of lists
            Each inner list contains indices of peaks belonging to that cluster
        cluster_info : pandas.DataFrame
            DataFrame with cluster assignment for each peak
        """
        if not hasattr(self, 'overlap_matrix') or self.overlap_matrix is None:
            raise ValueError("No overlap matrix available. Run calculate_overlap_matrix() first.")
        
        if not hasattr(self, 'df_processed') or self.df_processed is None:
            raise ValueError("No processed data available. Run preprocessing first.")
        
        # Sort peaks by energy (already should be sorted, but ensure it)
        df_sorted = self.df_processed.sort_values('E').reset_index(drop=True)
        sorted_indices = np.arange(len(df_sorted))
        
        # Convert overlap matrix to array for numba functions
        overlap_matrix_array = self.overlap_matrix.astype(np.float64)
        
        print(f"Clustering {len(df_sorted)} peaks with {method} method...")
        print(f"Overlap threshold: {overlap_threshold}%")
        
        if method == 'sequential':
            clusters = fast_sequential_clustering(overlap_matrix_array, sorted_indices, overlap_threshold)
        elif method == 'skip_tolerant':
            print(f"Skip tolerance: {n_skipped} peaks")
            clusters = fast_skip_tolerant_clustering(overlap_matrix_array, sorted_indices, 
                                                    overlap_threshold, n_skipped)
        else:
            raise ValueError("Method must be 'sequential' or 'skip_tolerant'")
        
        # Create cluster info DataFrame
        cluster_assignments = np.zeros(len(df_sorted), dtype=int)
        for cluster_id, peak_indices in enumerate(clusters):
            for peak_idx in peak_indices:
                cluster_assignments[peak_idx] = cluster_id
        
        # Add cluster information to dataframe copy
        cluster_info = df_sorted.copy()
        cluster_info['cluster_id'] = cluster_assignments
        cluster_info['cluster_size'] = cluster_info['cluster_id'].map(
            {i: len(clusters[i]) for i in range(len(clusters))}
        )
        
        # Store results
        self.clusters = clusters
        self.cluster_info = cluster_info
        
        # Print summary
        print(f"Created {len(clusters)} clusters")
        cluster_sizes = [len(cluster) for cluster in clusters]
        print(f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={np.mean(cluster_sizes):.1f}")
        
        return clusters, cluster_info
    
    def fit_cluster_gaussians(self, energy_range=None, n_points=2000):
        """
        Create Gaussian representations of clusters by summing peaks and fitting.
        
        Parameters:
        -----------
        energy_range : tuple, optional
            (min_energy, max_energy) range for fitting. If None, auto-calculated
        n_points : int
            Number of points for numerical fitting
            
        Returns:
        --------
        cluster_gaussians : pandas.DataFrame
            DataFrame with fitted Gaussian parameters for each cluster
        """
        if not hasattr(self, 'cluster_info') or self.cluster_info is None:
            raise ValueError("No cluster info available. Run cluster_peaks() first.")
        
        # Set energy range for fitting
        if energy_range is None:
            min_energy = self.cluster_info['E'].min() - 3 * self.cluster_info['width'].max()
            max_energy = self.cluster_info['E'].max() + 3 * self.cluster_info['width'].max()
        else:
            min_energy, max_energy = energy_range
        
        energy_axis = np.linspace(min_energy, max_energy, n_points)
        
        # Define Gaussian function for fitting (scipy format)
        def gaussian_for_fitting(x, amplitude, center, sigma):
            return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
        
        def calculate_reduced_chi_squared(observed, expected, n_params=3):
            """Calculate reduced chi-squared with realistic variance model"""
            threshold = np.max(observed) * 1e-4
            mask = observed > threshold
            
            if np.sum(mask) <= n_params:
                return np.inf
            
            observed_masked = observed[mask]
            expected_masked = expected[mask]
            
            # Use a constant relative uncertainty (e.g., 5% of maximum signal)
            max_signal = np.max(observed_masked)
            variance = (0.05 * max_signal) ** 2  # 5% uncertainty
            
            chi_squared = np.sum(((observed_masked - expected_masked) ** 2) / variance)
            degrees_of_freedom = len(observed_masked) - n_params
            
            return chi_squared / degrees_of_freedom if degrees_of_freedom > 0 else np.inf
        
        cluster_gaussians = []
        n_clusters = self.cluster_info['cluster_id'].max() + 1
        
        print(f"Fitting Gaussians for {n_clusters} clusters...")
        
        for cluster_id in range(n_clusters):
            # Get peaks in this cluster
            cluster_peaks = self.cluster_info[self.cluster_info['cluster_id'] == cluster_id]
            
            # Sum all Gaussians in this cluster
            cluster_intensity = np.zeros(n_points)
            total_os = 0
            
            for _, peak in cluster_peaks.iterrows():
                peak_intensity = gaussian_numba(energy_axis, peak['OS'], peak['E'], peak['width'])
                cluster_intensity += peak_intensity
                total_os += peak['OS']
            
            # Find reasonable initial guesses for fitting
            max_idx = np.argmax(cluster_intensity)
            center_guess = energy_axis[max_idx]
            amplitude_guess = cluster_intensity[max_idx]
            
            # Estimate sigma from peak width at half maximum
            half_max = amplitude_guess / 2
            left_idx = np.where(cluster_intensity[:max_idx] <= half_max)[0]
            right_idx = np.where(cluster_intensity[max_idx:] <= half_max)[0]
            
            if len(left_idx) > 0 and len(right_idx) > 0:
                fwhm = energy_axis[max_idx + right_idx[0]] - energy_axis[left_idx[-1]]
                sigma_guess = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
            else:
                # Fallback: use average width of peaks in cluster
                sigma_guess = cluster_peaks['width'].mean()
            
            try:
                # Fit Gaussian to summed cluster
                popt, pcov = curve_fit(
                    gaussian_for_fitting, 
                    energy_axis, 
                    cluster_intensity,
                    p0=[amplitude_guess, center_guess, sigma_guess],
                    bounds=([0, min_energy, 0.001], [np.inf, max_energy, 10.0]),
                    maxfev=5000
                )
                
                fitted_amplitude, fitted_center, fitted_sigma = popt
                
                # Generate fitted curve for chi-squared calculation
                fitted_curve = gaussian_for_fitting(energy_axis, fitted_amplitude, fitted_center, fitted_sigma)
                
                # Calculate reduced chi-squared
                reduced_chi_squared = calculate_reduced_chi_squared(cluster_intensity, fitted_curve, n_params=3)
                
                # Calculate parameter uncertainties from covariance matrix
                param_errors = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan, np.nan]
                
                # Calculate normalized amplitude (area under curve)
                # For normalized Gaussian: area = amplitude * sigma * sqrt(2*pi)
                normalized_amplitude = fitted_amplitude * fitted_sigma * np.sqrt(2 * np.pi)
                
                cluster_gaussians.append({
                    'cluster_id': cluster_id,
                    'E': fitted_center,
                    'width': fitted_sigma,
                    'OS': normalized_amplitude,
                    'height_at_center': fitted_amplitude,
                    'n_peaks': len(cluster_peaks),
                    'total_original_OS': total_os,
                    'reduced_chi_squared': reduced_chi_squared,
                    'center_error': param_errors[1],
                    'sigma_error': param_errors[2],
                    'amplitude_error': param_errors[0],
                    'fit_success': True
                })
                
            except Exception as e:
                print(f"Warning: Failed to fit cluster {cluster_id}: {e}")
                # Use weighted average as fallback
                weighted_center = np.average(cluster_peaks['E'], weights=cluster_peaks['OS'])
                weighted_sigma = np.average(cluster_peaks['width'], weights=cluster_peaks['OS'])
                
                cluster_gaussians.append({
                    'cluster_id': cluster_id,
                    'E': weighted_center,
                    'width': weighted_sigma,
                    'OS': total_os,
                    'height_at_center': total_os / (weighted_sigma * np.sqrt(2 * np.pi)),
                    'n_peaks': len(cluster_peaks),
                    'total_original_OS': total_os,
                    'reduced_chi_squared': np.inf,
                    'center_error': np.nan,
                    'sigma_error': np.nan,
                    'amplitude_error': np.nan,
                    'fit_success': False
                })
        
        cluster_gaussians_df = pd.DataFrame(cluster_gaussians)
        
        # Store results
        self.cluster_gaussians = cluster_gaussians_df
        
        print(f"Successfully fitted {cluster_gaussians_df['fit_success'].sum()} out of {len(cluster_gaussians_df)} clusters")
        
        # Print fit quality summary
        successful_fits = cluster_gaussians_df[cluster_gaussians_df['fit_success']]
        if len(successful_fits) > 0:
            print(f"Reduced χ² statistics:")
            print(f"  Mean: {successful_fits['reduced_chi_squared'].mean():.3f}")
            print(f"  Median: {successful_fits['reduced_chi_squared'].median():.3f}")
            print(f"  Min: {successful_fits['reduced_chi_squared'].min():.3f}")
            print(f"  Max: {successful_fits['reduced_chi_squared'].max():.3f}")
            
            # Flag potentially poor fits
            poor_fits = successful_fits[successful_fits['reduced_chi_squared'] > 2.0]
            if len(poor_fits) > 0:
                print(f"Warning: {len(poor_fits)} clusters have reduced χ² > 2.0 (poor fits)")
        
        return cluster_gaussians_df
    
    def iterative_analysis(self, overlap_threshold=50, method='sequential', n_skipped=1, 
                      max_iterations=10, convergence_check=True):
        """
        Perform iterative analysis until no overlaps exceed threshold or max iterations reached.
        
        Parameters:
        -----------
        overlap_threshold : float
            Minimum overlap percentage to group peaks into same cluster
        method : str
            Clustering method: 'sequential' or 'skip_tolerant'
        n_skipped : int
            For skip_tolerant method: number of peaks to skip before terminating cluster
        max_iterations : int
            Maximum number of iterations to perform
        convergence_check : bool
            Whether to check for convergence (no overlaps above threshold)
            
        Returns:
        --------
        iteration_results : dict
            Dictionary containing results from each iteration
        final_peaks : pandas.DataFrame
            Final set of peaks after all iterations
        """
        if not hasattr(self, 'df_processed') or self.df_processed is None:
            raise ValueError("No processed data available. Run preprocessing first.")
        
        # Initialize storage for iteration results
        iteration_results = {
            'overlap_matrices': [],
            'cluster_infos': [],
            'cluster_gaussians': [],
            'iteration_summaries': []
        }
        
        # Start with processed data
        current_peaks = self.df_processed.copy()
        
        print(f"Starting iterative analysis with {len(current_peaks)} initial peaks")
        print(f"Overlap threshold: {overlap_threshold}%")
        print(f"Method: {method}")
        print("=" * 60)
        
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}:")
            print(f"Input peaks: {len(current_peaks)}")
            
            # Update df_processed for this iteration
            self.df_processed = current_peaks.copy()
            
            # Calculate overlap matrix
            overlap_matrix = self.calculate_overlap_matrix()
            
            # Check convergence: are there any off-diagonal elements >= threshold?
            if convergence_check:
                # Create mask for off-diagonal elements
                n = overlap_matrix.shape[0]
                mask = ~np.eye(n, dtype=bool)
                off_diagonal_overlaps = overlap_matrix[mask]
                max_overlap = np.max(off_diagonal_overlaps) if len(off_diagonal_overlaps) > 0 else 0
                
                print(f"Maximum off-diagonal overlap: {max_overlap:.1f}%")
                
                if max_overlap < overlap_threshold:
                    print(f"Convergence reached! No overlaps >= {overlap_threshold}%")
                    
                    # Store final results
                    iteration_results['overlap_matrices'].append(overlap_matrix.copy())
                    iteration_results['cluster_infos'].append(None)  # No clustering needed
                    iteration_results['cluster_gaussians'].append(None)  # No clustering needed
                    iteration_results['iteration_summaries'].append({
                        'iteration': iteration + 1,
                        'input_peaks': len(current_peaks),
                        'output_peaks': len(current_peaks),
                        'max_overlap': max_overlap,
                        'converged': True
                    })
                    
                    print(f"Final analysis complete after {iteration + 1} iterations")
                    break
            
            # Perform clustering
            clusters, cluster_info = self.cluster_peaks(
                overlap_threshold=overlap_threshold, 
                method=method, 
                n_skipped=n_skipped
            )
            
            # Fit cluster Gaussians
            cluster_gaussians = self.fit_cluster_gaussians()
            
            # Store iteration results
            iteration_results['overlap_matrices'].append(overlap_matrix.copy())
            iteration_results['cluster_infos'].append(cluster_info.copy())
            iteration_results['cluster_gaussians'].append(cluster_gaussians.copy())
            
            # Create summary for this iteration
            summary = {
                'iteration': iteration + 1,
                'input_peaks': len(current_peaks),
                'output_peaks': len(cluster_gaussians),
                'n_clusters': len(clusters),
                'max_overlap': max_overlap if convergence_check else np.max(overlap_matrix[~np.eye(overlap_matrix.shape[0], dtype=bool)]),
                'converged': False
            }
            iteration_results['iteration_summaries'].append(summary)
            
            print(f"Output peaks: {len(cluster_gaussians)} (reduced from {len(current_peaks)})")
            print(f"Reduction: {len(current_peaks) - len(cluster_gaussians)} peaks")
            
            # Prepare for next iteration: use fitted cluster Gaussians as new peaks
            # Keep only the columns needed for the next iteration
            current_peaks = cluster_gaussians[['E', 'width', 'OS']].copy()
            current_peaks = current_peaks.sort_values('E').reset_index(drop=True)
            
            # Check if we've converged naturally (no reduction in peaks)
            if len(current_peaks) == summary['input_peaks']:
                print("No further peak reduction possible - analysis complete")
                summary['converged'] = True
                break
        
        else:
            print(f"\nMaximum iterations ({max_iterations}) reached")
        
        # Final summary
        print("\n" + "=" * 60)
        print("ITERATION SUMMARY:")
        for summary in iteration_results['iteration_summaries']:
            status = "CONVERGED" if summary['converged'] else "CONTINUED"
            print(f"Iteration {summary['iteration']}: {summary['input_peaks']} → {summary['output_peaks']} peaks "
                f"(max overlap: {summary['max_overlap']:.1f}%) [{status}]")
        
        # Store final results in analyzer
        self.iteration_results = iteration_results
        self.final_peaks = current_peaks
        
        print(f"\nFinal result: {len(current_peaks)} peaks")
        
        return iteration_results, current_peaks
    
    def run_complete_analysis(self, 
                          energy_threshold=320,
                          os_threshold_percent=2,
                          overlap_threshold=50,
                          method='skip_tolerant',
                          n_skipped=1,
                          max_iterations=10,
                          convergence_check=True,
                          compound_dict=None,
                          henke_data_dir="Element Library",
                          dft_step_data_path=None,
                          step_edge_params=None,
                          energy_range=(280,360),
                          n_points=2000):
        """
        Complete NEXAFS analysis pipeline combining preprocessing, clustering, 
        theoretical absorption, and optional step edge analysis.
        
        Parameters:
        -----------
        energy_threshold : float
            Maximum energy for filtering transitions
        os_threshold_percent : float  
            Minimum oscillator strength threshold percentage
        overlap_threshold : float
            Overlap threshold for clustering
        method : str
            Clustering method ('sequential' or 'skip_tolerant')
        n_skipped : int
            Number of peaks to skip in skip_tolerant method
        max_iterations : int
            Maximum iterations for iterative analysis
        convergence_check : bool
            Whether to check for convergence
        compound_dict : dict
            Dictionary for compound composition (e.g., {'C': 32, 'H': 16, 'Cu': 1, 'N': 8})
        henke_data_dir : str
            Directory containing .nff files
        dft_step_data_path : str, optional
            Path to CSV with DFT step edge data
        step_edge_params : dict, optional
            Parameters for step edge building
        energy_range : tuple, optional
            (min, max) energy range for spectra
        n_points : int
            Number of points for spectrum generation
            
        Returns:
        --------
        dict : Complete analysis results
        """
        
        results = {}
        
        # STEP 1: PREPROCESSING
        print("=" * 60)
        print("STEP 1: PREPROCESSING")
        print("=" * 60)
        
        df_after_energy = self.filter_by_energy(energy_threshold=energy_threshold)
        df_with_normalized = self.create_normalized_os(df_after_energy)
        df_filtered = self.filter_by_os_threshold(df_with_normalized, 
                                                os_threshold_percent=os_threshold_percent)
        df_filtered = df_filtered.sort_values(by='E').reset_index(drop=True)
        self.df_processed = df_filtered
        
        # Store preprocessing results
        results['preprocessing'] = {
            'df_raw': self.df_raw.copy(),
            'df_after_energy': df_after_energy.copy(),
            'df_with_normalized': df_with_normalized.copy(), 
            'df_filtered': df_filtered.copy(),
            'df_processed': self.df_processed.copy()
        }
        
        # STEP 2: ITERATIVE CLUSTERING ANALYSIS
        print("\n" + "=" * 60)
        print("STEP 2: ITERATIVE CLUSTERING ANALYSIS") 
        print("=" * 60)
        
        iteration_results, final_peaks = self.iterative_analysis(
            overlap_threshold=overlap_threshold,
            method=method,
            n_skipped=n_skipped,
            max_iterations=max_iterations,
            convergence_check=convergence_check
        )
        
        results['clustering'] = {
            'iteration_results': iteration_results,
            'final_peaks': final_peaks.copy(),
            'parameters': {
                'overlap_threshold': overlap_threshold,
                'method': method,
                'n_skipped': n_skipped,
                'max_iterations': max_iterations
            }
        }
        
        # STEP 3: GENERATE NEXAFS SPECTRA FROM EACH ITERATION
        print("\n" + "=" * 60)
        print("STEP 3: GENERATING NEXAFS SPECTRA")
        print("=" * 60)
        
        if energy_range is None:
            min_e = min(self.df_processed['E'].min(), final_peaks['E'].min()) - 10
            max_e = max(self.df_processed['E'].max(), final_peaks['E'].max()) + 10
            energy_range = (min_e, max_e)
        
        # Generate spectrum from original processed data
        energy_axis, intensity_original = self.create_nexafs_spectrum(
            self.df_processed, energy_range, n_points
        )
        
        # Generate spectrum from final peaks
        energy_axis, intensity_final = self.create_nexafs_spectrum(
            final_peaks, energy_range, n_points
        )
        
        # Generate spectra from each iteration if cluster_gaussians exist
        iteration_spectra = {}
        for i, gaussians in enumerate(iteration_results['cluster_gaussians']):
            if gaussians is not None:
                _, intensity_iter = self.create_nexafs_spectrum(
                    gaussians[['E', 'width', 'OS']], energy_range, n_points
                )
                iteration_spectra[f'iteration_{i+1}'] = intensity_iter
        
        results['spectra'] = {
            'energy_axis': energy_axis,
            'intensity_original': intensity_original,
            'intensity_final': intensity_final,
            'iteration_spectra': iteration_spectra,
            'energy_range': energy_range,
            'n_points': n_points
        }
        
        # STEP 4: THEORETICAL ABSORPTION CALCULATION
        print("\n" + "=" * 60)
        print("STEP 4: THEORETICAL ABSORPTION")
        print("=" * 60)
        
        if compound_dict is not None:
            try:
                processor = HenkeDataProcessor(henke_data_dir)
                processor.load_all_elements()
                mu_energy, mu_compound = processor.calculate_compound_absorption_from_dict(compound_dict)
                
                results['theoretical_absorption'] = {
                    'mu_energy': mu_energy,
                    'mu_compound': mu_compound,
                    'compound_dict': compound_dict,
                    'processor': processor
                }
                print(f"Calculated absorption for compound: {compound_dict}")
                
            except Exception as e:
                print(f"Warning: Could not calculate theoretical absorption: {e}")
                results['theoretical_absorption'] = None
        else:
            print("No compound dictionary provided, skipping theoretical absorption")
            results['theoretical_absorption'] = None
        
        # STEP 5: DFT STEP EDGE ANALYSIS (OPTIONAL)
        print("\n" + "=" * 60)
        print("STEP 5: DFT STEP EDGE ANALYSIS")
        print("=" * 60)
        
        if dft_step_data_path is not None and results['theoretical_absorption'] is not None:
            try:
                # Default step edge parameters - use your working values
                default_step_params = {
                    'energy_min': energy_range[0],
                    'energy_max': energy_range[1], 
                    'n_points': n_points,
                    'preedge_range': [240, 284.1],  # Use your working values
                    'postedge_range': [284.3, 360], # Use your working values  
                    'stepWid1': 0.6,
                    'stepWid2': 12.0,
                    'stepDecay': 0.0075,  # Your working value
                    'poly_degree': 2,
                    'scale_to_experimental': True
                }
                
                if step_edge_params is not None:
                    default_step_params.update(step_edge_params)
                
                # Load DFT data
                import pandas as pd
                df_step = pd.read_csv(dft_step_data_path)
                
                # Create energy array for step edge
                step_energy_array = np.linspace(default_step_params['energy_min'], 
                                            default_step_params['energy_max'], 
                                            default_step_params['n_points'])
                
                # Initialize edge builder
                edge_builder = AbsorptionEdgeBuilder()
                
                # Build step edge
                total_edge, individual_edges = edge_builder.build_absorption_edge(
                    df_step, step_energy_array, 
                    results['theoretical_absorption']['mu_energy'],
                    results['theoretical_absorption']['mu_compound'],
                    default_step_params['preedge_range'],
                    default_step_params['postedge_range'],
                    default_step_params['stepWid1'],
                    default_step_params['stepWid2'], 
                    default_step_params['stepDecay'],
                    poly_degree=default_step_params['poly_degree'],
                    return_individual=True
                )
                
                # Scale to experimental if requested
                scaled_edge = None
                if default_step_params['scale_to_experimental']:
                    preedge_ref = sum(default_step_params['preedge_range']) / 2
                    postedge_ref = sum(default_step_params['postedge_range']) / 2
                    
                    scaled_edge, scale_factor, offset = edge_builder.scale_to_experimental(
                        step_energy_array,
                        results['theoretical_absorption']['mu_compound'],
                        results['theoretical_absorption']['mu_energy'],
                        preedge_ref, postedge_ref
                    )
                
                results['step_edge'] = {
                    'df_step': df_step.copy(),
                    'step_energy_array': step_energy_array,
                    'total_edge': total_edge,
                    'scaled_edge': scaled_edge,
                    'individual_edges': individual_edges,
                    'edge_builder': edge_builder,
                    'parameters': default_step_params
                }
                
                print("DFT step edge analysis completed")
                
            except Exception as e:
                print(f"Warning: Could not complete step edge analysis: {e}")
                results['step_edge'] = None
        else:
            print("DFT step edge analysis skipped (missing data path or theoretical absorption)")
            results['step_edge'] = None
        
        # STEP 6: SUMMARY AND STORAGE
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"Original peaks: {len(self.df_raw)}")
        print(f"After preprocessing: {len(self.df_processed)}")
        print(f"Final peaks after clustering: {len(final_peaks)}")
        print(f"Reduction factor: {len(self.df_processed) / len(final_peaks):.1f}x")
        
        if results['theoretical_absorption'] is not None:
            print(f"Theoretical absorption calculated for: {compound_dict}")
        
        if results['step_edge'] is not None:
            print(f"DFT step edge analysis completed")
        
        # Store complete results in analyzer
        self.complete_analysis_results = results
        
        return results