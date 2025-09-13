import matplotlib.pyplot as plt
import numpy as np
from overlapProcs import gaussian_numba

def plot_iteration_nexafs(results, iterations='all', figsize=(12, 8), 
                         alpha=0.7, linewidth=1.5, show_legend=True):
    """
    Plot NEXAFS spectra from specified iterations.
    
    Parameters:
    -----------
    results : dict
        Results from NEXAFSAnalyzer.run_complete_analysis()
    iterations : str, int, or list
        'all', 'original', 'final', specific iteration number, or list of iterations
    """
    
    if 'spectra' not in results:
        raise ValueError("No spectra data found in results")
    
    spectra_data = results['spectra']
    energy_axis = spectra_data['energy_axis']
    
    plt.figure(figsize=figsize)
    
    # Handle different iteration specifications
    if iterations == 'all':
        # Plot original
        plt.plot(energy_axis, spectra_data['intensity_original'], 
                color='red', linewidth=linewidth+0.5, label='Original', alpha=alpha+0.1)
        
        # Plot all iterations
        colors = plt.cm.viridis(np.linspace(0, 1, len(spectra_data['iteration_spectra'])))
        for i, (iter_name, intensity) in enumerate(spectra_data['iteration_spectra'].items()):
            iter_num = int(iter_name.split('_')[1])
            plt.plot(energy_axis, intensity, color=colors[i], 
                    linewidth=linewidth, alpha=alpha, label=f'Iteration {iter_num}')
        
        # Plot final
        plt.plot(energy_axis, spectra_data['intensity_final'], 
                color='black', linewidth=linewidth+0.5, label='Final', linestyle='--')
                
    elif iterations == 'original':
        plt.plot(energy_axis, spectra_data['intensity_original'], 
                color='red', linewidth=linewidth, label='Original')
                
    elif iterations == 'final':
        plt.plot(energy_axis, spectra_data['intensity_final'], 
                color='black', linewidth=linewidth, label='Final')
                
    elif isinstance(iterations, (int, list)):
        if isinstance(iterations, int):
            iterations = [iterations]
            
        colors = plt.cm.tab10(np.linspace(0, 1, len(iterations)))
        
        for i, iter_num in enumerate(iterations):
            iter_name = f'iteration_{iter_num}'
            if iter_name in spectra_data['iteration_spectra']:
                intensity = spectra_data['iteration_spectra'][iter_name]
                plt.plot(energy_axis, intensity, color=colors[i], 
                        linewidth=linewidth, alpha=alpha, label=f'Iteration {iter_num}')
            else:
                print(f"Warning: Iteration {iter_num} not found")
    
    plt.xlabel('Energy (eV)')
    plt.ylabel('Intensity')
    plt.title('NEXAFS Spectra')
    if show_legend:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_overlap_heatmaps(results, iterations='all', figsize_per_plot=(6, 5), 
                         cmap='viridis', show_colorbar=True, annotate_values=False):
    """
    Plot overlap matrices as heatmaps for specified iterations.
    
    Parameters:
    -----------
    results : dict
        Results from NEXAFSAnalyzer.run_complete_analysis()
    iterations : str, int, or list
        'all', specific iteration number, or list of iterations
    figsize_per_plot : tuple
        Size of each individual heatmap
    cmap : str
        Colormap for heatmap
    show_colorbar : bool
        Whether to show colorbar
    annotate_values : bool
        Whether to annotate values in cells (only for small matrices)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if 'clustering' not in results:
        raise ValueError("No clustering data found in results")
    
    overlap_matrices = results['clustering']['iteration_results']['overlap_matrices']
    
    # Determine which iterations to plot
    available_iterations = list(range(1, len(overlap_matrices) + 1))
    
    if iterations == 'all':
        plot_iterations = available_iterations
    elif isinstance(iterations, int):
        plot_iterations = [iterations]
    elif isinstance(iterations, list):
        plot_iterations = [i for i in iterations if i in available_iterations]
        if not plot_iterations:
            raise ValueError("No valid iterations found")
    else:
        raise ValueError("iterations must be 'all', int, or list of ints")
    
    n_plots = len(plot_iterations)
    
    # Calculate grid layout (max 4 columns)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Calculate figure size
    total_width = n_cols * figsize_per_plot[0]
    total_height = n_rows * figsize_per_plot[1]
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(total_width, total_height))
    
    # Handle single plot case
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if hasattr(axes, '__iter__') else [axes]
    else:
        axes = axes.flatten()
    
    for i, iter_num in enumerate(plot_iterations):
        ax = axes[i]
        
        # Get overlap matrix for this iteration (0-indexed)
        overlap_matrix = overlap_matrices[iter_num - 1]
        
        # Create heatmap
        im = ax.imshow(overlap_matrix, cmap=cmap, aspect='auto', 
                      vmin=0, vmax=100, interpolation='nearest')
        
        # Add annotations if requested and matrix is small enough
        if annotate_values and overlap_matrix.shape[0] <= 20:
            for row in range(overlap_matrix.shape[0]):
                for col in range(overlap_matrix.shape[1]):
                    value = overlap_matrix[row, col]
                    text_color = 'white' if value < 50 else 'black'
                    ax.text(col, row, f'{value:.0f}', 
                           ha='center', va='center', color=text_color, fontsize=8)
        
        # Set labels and title
        ax.set_xlabel('Peak Index')
        ax.set_ylabel('Peak Index')
        ax.set_title(f'Iteration {iter_num}\n({overlap_matrix.shape[0]} peaks)')
        
        # Add colorbar if requested
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Overlap (%)')
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Overlap Matrix Evolution', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

def plot_gaussian_fits(results, iteration, peaks='all', max_peaks=30, 
                      figsize_per_plot=(4, 3), energy_padding=2):
    """
    Plot Gaussian fits for clusters from a specific iteration in a grid layout.
    
    Parameters:
    -----------
    results : dict
        Results from NEXAFSAnalyzer.run_complete_analysis()
    iteration : int
        Iteration number to plot (1-indexed)
    peaks : str, int, or list
        'all', 'first_N' (where N is max_peaks), specific cluster IDs, or list of cluster IDs
    max_peaks : int
        Maximum number of plots (capped at 30)
    figsize_per_plot : tuple
        Size of each individual plot
    energy_padding : float
        Energy padding around peaks for plot range
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if 'clustering' not in results:
        raise ValueError("No clustering data found in results")
    
    iteration_results = results['clustering']['iteration_results']
    
    # Check if iteration exists
    if iteration < 1 or iteration > len(iteration_results['cluster_gaussians']):
        raise ValueError(f"Iteration {iteration} not found")
    
    cluster_gaussians = iteration_results['cluster_gaussians'][iteration - 1]
    cluster_info = iteration_results['cluster_infos'][iteration - 1]
    
    if cluster_gaussians is None or cluster_info is None:
        raise ValueError(f"No cluster data for iteration {iteration}")
    
    # Determine which clusters to plot
    available_clusters = cluster_gaussians['cluster_id'].tolist()
    max_peaks = min(max_peaks, 30)  # Hard cap at 30
    
    if peaks == 'all':
        plot_clusters = available_clusters[:max_peaks]
    elif isinstance(peaks, str) and peaks.startswith('first_'):
        n = int(peaks.split('_')[1])
        plot_clusters = available_clusters[:min(n, max_peaks)]
    elif isinstance(peaks, int):
        plot_clusters = [peaks] if peaks in available_clusters else []
    elif isinstance(peaks, list):
        plot_clusters = [c for c in peaks if c in available_clusters][:max_peaks]
    else:
        raise ValueError("peaks must be 'all', 'first_N', int, or list of ints")
    
    if not plot_clusters:
        raise ValueError("No valid clusters found to plot")
    
    n_plots = len(plot_clusters)
    
    # Calculate grid layout (max 5 columns)
    n_cols = min(5, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Calculate figure size
    total_width = n_cols * figsize_per_plot[0]
    total_height = n_rows * figsize_per_plot[1]
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(total_width, total_height))
    
    # Handle single plot case
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if hasattr(axes, '__iter__') else [axes]
    else:
        axes = axes.flatten()
    
    for i, cluster_id in enumerate(plot_clusters):
        ax = axes[i]
        
        # Get cluster data
        cluster_row = cluster_gaussians[cluster_gaussians['cluster_id'] == cluster_id].iloc[0]
        cluster_peaks = cluster_info[cluster_info['cluster_id'] == cluster_id]
        
        # Define energy range for this cluster
        center = cluster_row['E']
        width = cluster_row['width']
        e_min = center - width * 3 - energy_padding
        e_max = center + width * 3 + energy_padding
        
        # Extend range to include all original peaks in cluster
        if len(cluster_peaks) > 0:
            peak_min = cluster_peaks['E'].min() - cluster_peaks['width'].max()
            peak_max = cluster_peaks['E'].max() + cluster_peaks['width'].max()
            e_min = min(e_min, peak_min - energy_padding)
            e_max = max(e_max, peak_max + energy_padding)
        
        energy_range = np.linspace(e_min, e_max, 2000)
        
        # Plot individual peaks in cluster (summed)
        summed_intensity = np.zeros_like(energy_range)
        for _, peak in cluster_peaks.iterrows():
            peak_intensity = gaussian_numba(energy_range, peak['OS'], peak['E'], peak['width'])
            summed_intensity += peak_intensity
        
        if len(cluster_peaks) > 0:
            ax.plot(energy_range, summed_intensity, 'b--', alpha=0.6, linewidth=1.5, 
                   label=f'Summed ({len(cluster_peaks)} peaks)')
        
        # Plot fitted Gaussian - USE OS VALUE, NOT HEIGHT
        fitted_intensity = gaussian_numba(energy_range, cluster_row['OS'], 
                                       cluster_row['E'], cluster_row['width'])
        ax.plot(energy_range, fitted_intensity, 'r-', linewidth=2, label='Fitted Gaussian')
        
        # Add vertical line at fitted center
        ax.axvline(cluster_row['E'], color='red', linestyle='--', alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Intensity')
        
        # Create title with cluster info
        fit_quality = "Good" if cluster_row['reduced_chi_squared'] < 2.0 else "Poor"
        chi2_str = f"{cluster_row['reduced_chi_squared']:.2f}" if not np.isinf(cluster_row['reduced_chi_squared']) else "∞"
        
        title = f"Cluster {cluster_id}\nχ²={chi2_str} ({fit_quality})"
        ax.set_title(title, fontsize=10)
        
        # Add legend if there's room
        if len(cluster_peaks) > 0:
            ax.legend(fontsize=8, loc='upper right')
        
        ax.grid(True, alpha=0.3)
        
        # Set y-limits to show the peaks nicely
        max_intensity = max(fitted_intensity.max(), summed_intensity.max() if len(cluster_peaks) > 0 else 0)
        ax.set_ylim(0, max_intensity * 1.1)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Gaussian Fits - Iteration {iteration}', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()
    
    print(f"Displayed {len(plot_clusters)} clusters from iteration {iteration}")
    if len(available_clusters) > len(plot_clusters):
        print(f"Note: {len(available_clusters) - len(plot_clusters)} additional clusters available")

def plot_bare_atom_absorption(results, energy_xlim=None, figsize=(12, 6), 
                             log_yscale=True, log_xscale = True, show_compound_info=True):
    """
    Plot bare atom absorption coefficient from theoretical calculations.
    
    Parameters:
    -----------
    results : dict
        Results from NEXAFSAnalyzer.run_complete_analysis()
    energy_xlim : tuple, optional
        Energy range for x-axis (min, max)
    figsize : tuple
        Figure size
    log_scale : bool
        Whether to use log scale for y-axis
    show_compound_info : bool
        Whether to show compound composition in title
    """
    
    if 'theoretical_absorption' not in results or results['theoretical_absorption'] is None:
        raise ValueError("No theoretical absorption data found in results")
    
    theo_data = results['theoretical_absorption']
    mu_energy = theo_data['mu_energy']
    mu_compound = theo_data['mu_compound']
    compound_dict = theo_data['compound_dict']
    
    plt.figure(figsize=figsize)
    plt.plot(mu_energy, mu_compound, 'b-', linewidth=2, label='Theoretical Absorption')
    
    plt.xlabel('Energy (eV)')
    plt.ylabel('Mass Absorption Coefficient (cm²/g)')
    
    if show_compound_info:
        compound_str = ', '.join([f'{elem}:{count}' for elem, count in compound_dict.items()])
        plt.title(f'Bare Atom Absorption - {compound_str}')
    else:
        plt.title('Bare Atom Absorption')
    
    if log_yscale:
        plt.yscale('log')
    
    if log_xscale:
        plt.xscale('log')

    if energy_xlim:
        plt.xlim(energy_xlim)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_absorption_vs_step_edge(results, energy_xlim=None, figsize=(12, 8), ylim =(0,1e8), show_baselines=True):
    """
    Plot theoretical absorption against DFT step edge.
    
    Parameters:
    -----------
    results : dict
        Results from NEXAFSAnalyzer.run_complete_analysis()
    energy_xlim : tuple, optional
        Energy range for x-axis (min, max)
    figsize : tuple
        Figure size
    normalize : bool
        Whether to normalize both curves to [0,1] for comparison
    show_baselines : bool
        Whether to show preedge/postedge baselines
    """
    
    if 'theoretical_absorption' not in results or results['theoretical_absorption'] is None:
        raise ValueError("No theoretical absorption data found in results")
    
    if 'step_edge' not in results or results['step_edge'] is None:
        raise ValueError("No step edge data found in results")
    
    theo_data = results['theoretical_absorption']
    step_data = results['step_edge']
    
    mu_energy = theo_data['mu_energy']
    mu_compound = theo_data['mu_compound']
    step_energy = step_data['step_energy_array']
    
    # Use scaled edge if available, otherwise total edge
    step_edge = (step_data['scaled_edge'] if step_data['scaled_edge'] is not None 
                else step_data['total_edge'])
    
    plt.figure(figsize=figsize)
    plt.plot(mu_energy, mu_compound, 'k-', linewidth=2, alpha=0.8, label='Theoretical Absorption')
    plt.plot(step_energy, step_edge, 'r-', linewidth=2, label='DFT Step Edge')
    plt.ylabel('Absorption')
    
    # Show baselines if requested and not normalized
    if show_baselines and 'edge_builder' in step_data:
        edge_builder = step_data['edge_builder']
        if hasattr(edge_builder, 'preedge_coeffs') and edge_builder.preedge_coeffs is not None:
            preedge_baseline = np.polyval(edge_builder.preedge_coeffs, step_energy)
            postedge_baseline = np.polyval(edge_builder.postedge_coeffs, step_energy)
            
            plt.plot(step_energy, preedge_baseline, '--', color='gray', alpha=0.6, 
                    label='Preedge baseline')
            plt.plot(step_energy, postedge_baseline, '--', color='orange', alpha=0.6, 
                    label='Postedge baseline')
    
    plt.xlabel('Energy (eV)')
    plt.title('Theoretical Absorption vs DFT Step Edge')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if energy_xlim:
        plt.xlim(energy_xlim)
    
    if ylim:
        plt.ylim(ylim)
    
    plt.tight_layout()
    plt.show()

def plot_step_edge_fit_quality(results, figsize=(14, 10), energy_xlim=None, 
                              residuals=True, highlight_regions=True):
    """
    Plot step edge fit quality showing baselines, fit regions, and residuals.
    
    Parameters:
    -----------
    results : dict
        Results from NEXAFSAnalyzer.run_complete_analysis()
    figsize : tuple
        Figure size
    energy_xlim : tuple, optional
        Energy range for x-axis (min, max)
    residuals : bool
        Whether to show residuals plot
    highlight_regions : bool
        Whether to highlight preedge/postedge fitting regions
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if 'theoretical_absorption' not in results or results['theoretical_absorption'] is None:
        raise ValueError("No theoretical absorption data found in results")
    
    if 'step_edge' not in results or results['step_edge'] is None:
        raise ValueError("No step edge data found in results")
    
    theo_data = results['theoretical_absorption']
    step_data = results['step_edge']
    
    mu_energy = theo_data['mu_energy']
    mu_compound = theo_data['mu_compound']
    step_energy = step_data['step_energy_array']
    
    # Use scaled edge if available, otherwise total edge
    step_edge = (step_data['scaled_edge'] if step_data['scaled_edge'] is not None 
                else step_data['total_edge'])
    
    # Get baseline coefficients
    edge_builder = step_data['edge_builder']
    preedge_coeffs = edge_builder.preedge_coeffs
    postedge_coeffs = edge_builder.postedge_coeffs
    
    # Calculate baselines
    preedge_baseline = np.polyval(preedge_coeffs, step_energy)
    postedge_baseline = np.polyval(postedge_coeffs, step_energy)
    
    # Get fit parameters
    params = step_data['parameters']
    preedge_range = params['preedge_range']
    postedge_range = params['postedge_range']
    
    # Create subplots
    n_plots = 2 if residuals else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    # Main plot
    ax1 = axes[0]
    
    # Plot experimental data
    ax1.plot(mu_energy, mu_compound, 'k-', linewidth=2, alpha=0.8, label='Theoretical Absorption')
    
    # Plot step edge
    ax1.plot(step_energy, step_edge, 'r-', linewidth=2, label='DFT Step Edge')
    
    # Plot baselines
    ax1.plot(step_energy, preedge_baseline, '--', color='blue', alpha=0.7, 
            linewidth=2, label='Preedge baseline')
    ax1.plot(step_energy, postedge_baseline, '--', color='green', alpha=0.7, 
            linewidth=2, label='Postedge baseline')
    
    # Highlight fitting regions
    if highlight_regions:
        y_min, y_max = ax1.get_ylim()
        ax1.axvspan(preedge_range[0], preedge_range[1], alpha=0.2, color='blue', 
                   label='Preedge fit region')
        ax1.axvspan(postedge_range[0], postedge_range[1], alpha=0.2, color='green', 
                   label='Postedge fit region')
    
    ax1.set_ylabel('Absorption')
    ax1.set_title('Step Edge Fit Quality Assessment')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if energy_xlim:
        ax1.set_xlim(energy_xlim)
    
    # Residuals plot
    if residuals:
        ax2 = axes[1]
        
        # Interpolate step edge to experimental energy grid for residuals
        step_edge_interp = np.interp(mu_energy, step_energy, step_edge)
        residuals_data = mu_compound - step_edge_interp
        
        ax2.plot(mu_energy, residuals_data, 'purple', linewidth=1.5, alpha=0.8)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax2.fill_between(mu_energy, residuals_data, alpha=0.3, color='purple')
        
        # Highlight fitting regions in residuals
        if highlight_regions:
            ax2.axvspan(preedge_range[0], preedge_range[1], alpha=0.2, color='blue')
            ax2.axvspan(postedge_range[0], postedge_range[1], alpha=0.2, color='green')
        
        ax2.set_ylabel('Residuals')
        ax2.set_title('Fit Residuals (Experimental - DFT)')
        ax2.grid(True, alpha=0.3)
        
        # Calculate and display residual statistics
        rmse = np.sqrt(np.mean(residuals_data**2))
        max_abs_residual = np.max(np.abs(residuals_data))
        ax2.text(0.02, 0.98, f'RMSE: {rmse:.4f}\nMax |residual|: {max_abs_residual:.4f}', 
                transform=ax2.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set x-label only on bottom plot
    axes[-1].set_xlabel('Energy (eV)')
    
    plt.tight_layout()
    plt.show()

def plot_nexafs_data(data, mu=None, mu_energy=None, dft_step=None, dft_step_en = None, xrange=None):
    """Plot NEXAFS data for all angles, optionally with bare atom absorption."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine energy range
    if xrange is None:
        all_energies = []
        for dataset in data.values():
            all_energies.extend(dataset['energy'])
        if mu_energy is not None:
            all_energies.extend(mu_energy)
        xmin, xmax = min(all_energies), max(all_energies)
    else:
        xmin, xmax = xrange
    
    # Collect intensity values within energy range for dynamic y-limits
    intensities_in_range = []
    
    for angle, dataset in data.items():
        energy = dataset['energy']
        intensity = dataset['intensity']
        label = f"{dataset['sample_name']} ({angle}°)"
        
        # Filter data within energy range
        mask = (energy >= xmin) & (energy <= xmax)
        energy_filtered = energy[mask]
        intensity_filtered = intensity[mask]
        
        if len(energy_filtered) > 0:
            intensities_in_range.extend(intensity_filtered)
            ax.plot(energy_filtered, intensity_filtered, label=label, marker='o', markersize=2)
   
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Intensity')
    ax.set_title('NEXAFS Spectra')
    
    # Plot bare atom absorption if provided
    if mu is not None and mu_energy is not None:
        mask_mu = (mu_energy >= xmin) & (mu_energy <= xmax)
        mu_energy_filtered = mu_energy[mask_mu]
        mu_filtered = mu[mask_mu]
        
        if len(mu_energy_filtered) > 0:
            intensities_in_range.extend(mu_filtered)
            ax.plot(mu_energy_filtered, mu_filtered, 'k--', linewidth=2, label='Bare atom (μ)', alpha=0.7)
    
    # Plot bare atom absorption if provided
    if dft_step is not None and dft_step_en is not None:
        mask_dft_step = (dft_step_en >= xmin) & (dft_step_en <= xmax)
        dft_step_en_filtered = dft_step_en[mask_dft_step]
        dft_step_filtered = dft_step[mask_dft_step]
        
        if len(dft_step_en_filtered) > 0:
            intensities_in_range.extend(dft_step_filtered)
            ax.plot(dft_step_en_filtered, dft_step_filtered, 'k--', linewidth=2, label='DFT Step Edge (μ)', alpha=0.7)

    # Set axis limits
    ax.set_xlim(xmin, xmax)
    
    if intensities_in_range:
        ymin, ymax = min(intensities_in_range), max(intensities_in_range)
        y_margin = (ymax - ymin) * 0.05  # 5% margin
        ax.set_ylim(ymin - y_margin, ymax + y_margin)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
   
    plt.tight_layout()
    plt.show()
   
    return fig, ax

