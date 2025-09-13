import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def load_nexafs_data(filepath):
    """Load NEXAFS experimental data from CSV file."""
    df = pd.read_csv(filepath)
    
    # Separate energy and intensity columns
    energy_cols = [col for col in df.columns if col.startswith('E_')]
    intensity_cols = [col for col in df.columns if not col.startswith('E_')]
    
    # Extract angles from column names
    angles = [int(col.split('_')[-1]) for col in energy_cols]
    
    # Create structured data
    data = {}
    for i, angle in enumerate(angles):
        energy_col = f'E_{angle}'
        intensity_col = intensity_cols[i]  # Assumes same order
        
        data[angle] = {
            'energy': df[energy_col].dropna(),
            'intensity': df[intensity_col].dropna(),
            'sample_name': intensity_col
        }
    
    return data

def normalize_to_bare_atom(data, mu, mu_energy, e1, e2):
    """
    Normalize experimental spectra to bare atom absorption using two energy points.
    
    Parameters:
    data: dict from load_nexafs_data()
    mu: bare atom absorption values
    mu_energy: energy values for bare atom absorption
    e1, e2: two energy points for scaling (eV)
    
    Returns:
    normalized_data: dict with same structure as input but with normalized intensities
    """
    # Interpolate bare atom data for consistent energy grid
    mu_interp = interp1d(mu_energy, mu, kind='linear', fill_value='extrapolate')
    
    normalized_data = {}
    
    for angle, dataset in data.items():
        energy = dataset['energy']
        intensity = dataset['intensity']
        
        # Get bare atom values at the two normalization points
        mu_e1 = mu_interp(e1)
        mu_e2 = mu_interp(e2)
        
        # Find closest energy points in experimental data
        idx1 = np.argmin(np.abs(energy - e1))
        idx2 = np.argmin(np.abs(energy - e2))
        
        exp_e1 = intensity.iloc[idx1]
        exp_e2 = intensity.iloc[idx2]
        
        # Calculate scaling parameters (linear transformation)
        # exp_norm = a * exp + b, such that exp_norm(e1) = mu(e1) and exp_norm(e2) = mu(e2)
        a = (mu_e2 - mu_e1) / (exp_e2 - exp_e1)
        b = mu_e1 - a * exp_e1
        
        # Apply normalization
        normalized_intensity = a * intensity + b
        
        normalized_data[angle] = {
            'energy': energy,
            'intensity': normalized_intensity,
            'sample_name': dataset['sample_name']
        }
    
    return normalized_data

