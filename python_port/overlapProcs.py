import numpy as np
from numba import jit, prange
from scipy.integrate import quad
import matplotlib.pyplot as plt
import cupy as cp

@jit(nopython=True, parallel=True, fastmath=True)
def gaussian_numba(x, amp, center, sigma):
    """Fast Gaussian function for Numba"""
    normalization = 1 / (sigma * np.sqrt(2 * np.pi))
    return amp * normalization * np.exp(-0.5 * ((x - center) / sigma) ** 2)

@jit(nopython=True, fastmath=True)
def gaussian_overlap_numba(x, amp1, center1, sigma1, amp2, center2, sigma2):
    g1 = gaussian_numba(x, amp1, center1, sigma1)
    g2 = gaussian_numba(x, amp2, center2, sigma2)
    return np.minimum(g1, g2)

@jit(nopython=True, fastmath=True)
def get_integration_bounds(center1, sigma1, center2, sigma2, n_sigma=5):
    min_bound = min(center1 - n_sigma * sigma1, center2 - n_sigma * sigma2)
    max_bound = max(center1 + n_sigma * sigma1, center2 + n_sigma * sigma2)
    return min_bound, max_bound

def calculate_overlap_area(amp1, center1, sigma1, amp2, center2, sigma2):
    def overlap_func(x):
        g1 = amp1 / (sigma1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - center1) / sigma1) ** 2)
        g2 = amp2 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - center2) / sigma2) ** 2)
        return min(g1, g2)
    
    min_bound, max_bound = get_integration_bounds(center1, sigma1, center2, sigma2)
    
    try:
        overlap_area, _ = quad(overlap_func, min_bound, max_bound, limit=100)
        return max(0, overlap_area)
    except:
        return 0.0

@jit(nopython=True, parallel=True, fastmath=True)
def calculate_overlap_matrix_numba(energies, widths, amplitudes):
    n = len(energies)
    overlap_matrix = np.zeros((n, n), dtype=np.float64)
   
    # Set diagonal to 100.0
    for i in range(n):
        overlap_matrix[i, i] = 100.0
   
    # Only calculate upper triangle due to symmetry
    for i in prange(n):
        for j in range(i + 1, n):
            area1 = amplitudes[i]
            area2 = amplitudes[j]
            min_area = min(area1, area2)
           
            if min_area > 0:
                center1, sigma1, amp1 = energies[i], widths[i], amplitudes[i]
                center2, sigma2, amp2 = energies[j], widths[j], amplitudes[j]
               
                separation = abs(center1 - center2)
                combined_width = (sigma1 + sigma2) * 0.5
               
                if separation > 8 * combined_width:
                    percent_overlap = 0.0
                else:
                    # Optimized bounds calculation
                    sigma_min = min(sigma1, sigma2)
                    x_min = min(center1 - 5*sigma1, center2 - 5*sigma2)
                    x_max = max(center1 + 5*sigma1, center2 + 5*sigma2)
                   
                    n_points = min(10000, max(1000, int((x_max - x_min) / (sigma_min * 0.1))))
                    dx = (x_max - x_min) / (n_points - 1)
                   
                    # Vectorized calculation
                    x_vals = x_min + np.arange(n_points) * dx
                    
                    # Pre-compute constants
                    norm1 = amp1 / (sigma1 * np.sqrt(2 * np.pi))
                    norm2 = amp2 / (sigma2 * np.sqrt(2 * np.pi))
                    inv_2sigma1_sq = -0.5 / (sigma1 * sigma1)
                    inv_2sigma2_sq = -0.5 / (sigma2 * sigma2)
                   
                    # Vectorized Gaussian calculations
                    diff1 = x_vals - center1
                    diff2 = x_vals - center2
                    g1_vals = norm1 * np.exp(inv_2sigma1_sq * diff1 * diff1)
                    g2_vals = norm2 * np.exp(inv_2sigma2_sq * diff2 * diff2)
                   
                    # Vectorized minimum and integration
                    overlap_vals = np.minimum(g1_vals, g2_vals)
                    overlap_area = np.trapz(overlap_vals, dx=dx)
                   
                    percent_overlap = max(0.0, min(100.0, (overlap_area / min_area) * 100.0))
               
                overlap_matrix[i, j] = percent_overlap
                overlap_matrix[j, i] = percent_overlap  # Fill symmetric element
            else:
                overlap_matrix[i, j] = 0.0
                overlap_matrix[j, i] = 0.0
   
    return overlap_matrix

def calculate_overlap_matrix_hybrid(energies, widths, amplitudes, use_integration_threshold=1000):
    n = len(energies)
    
    if n <= use_integration_threshold:
        print(f"Using numerical integration for {n} peaks...")
        overlap_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    overlap_matrix[i, j] = 100.0
                else:
                    area1 = amplitudes[i]
                    area2 = amplitudes[j]
                    min_area = min(area1, area2)
                    
                    if min_area > 0:
                        overlap_area = calculate_overlap_area(
                            amplitudes[i], energies[i], widths[i],
                            amplitudes[j], energies[j], widths[j]
                        )
                        percent_overlap = (overlap_area / min_area) * 100.0
                        percent_overlap = max(0.0, min(100.0, percent_overlap))
                        
                        overlap_matrix[i, j] = percent_overlap
                        overlap_matrix[j, i] = percent_overlap
                    else:
                        overlap_matrix[i, j] = 0.0
                        overlap_matrix[j, i] = 0.0
        
        return overlap_matrix
    else:
        print(f"Using optimized numba calculation for {n} peaks...")
        return calculate_overlap_matrix_numba(
            energies.astype(np.float64),
            widths.astype(np.float64),
            amplitudes.astype(np.float64)
        )
