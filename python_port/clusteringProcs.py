from numba import jit, prange

@jit(nopython=True)
def fast_sequential_clustering(overlap_matrix_array, sorted_indices, threshold):
    """Sequential clustering checking overlap with FIRST peak in each cluster."""
    clusters = [[sorted_indices[0]]]  # Start with first peak
    current_cluster_idx = 0
    
    for i in range(1, len(sorted_indices)):
        current_peak_idx = sorted_indices[i]
        first_peak_idx = clusters[current_cluster_idx][0]  # Check with FIRST peak
        
        # Check overlap with first peak in current cluster
        overlap = overlap_matrix_array[current_peak_idx, first_peak_idx]
        
        if overlap >= threshold:
            # Add to current cluster
            clusters[current_cluster_idx].append(current_peak_idx)
        else:
            # Start new cluster
            clusters.append([current_peak_idx])
            current_cluster_idx += 1
    
    return clusters

@jit(nopython=True)
def fast_skip_tolerant_clustering(overlap_matrix_array, sorted_indices, threshold, n_skipped=1):
    """
    Skip-tolerant clustering that allows up to n_skipped peaks below threshold 
    before terminating a cluster.
    """
    n_peaks = len(sorted_indices)
    clustered = [False] * n_peaks  # Track which peaks are already clustered
    clusters = []
    
    i = 0
    while i < n_peaks:
        # Find next unclustered peak
        while i < n_peaks and clustered[i]:
            i += 1
        
        if i >= n_peaks:
            break
            
        # Start new cluster with this peak
        current_cluster = [sorted_indices[i]]
        first_peak_idx = sorted_indices[i]
        clustered[i] = True
        
        # Check subsequent peaks for inclusion in this cluster
        skip_count = 0
        j = i + 1
        
        while j < n_peaks and skip_count <= n_skipped:
            if clustered[j]:
                j += 1
                continue
                
            current_peak_idx = sorted_indices[j]
            overlap = overlap_matrix_array[current_peak_idx, first_peak_idx]
            
            if overlap >= threshold:
                # Add to cluster and reset skip count
                current_cluster.append(current_peak_idx)
                clustered[j] = True
                skip_count = 0  # Reset skip count after finding a valid peak
            else:
                # Skip this peak
                skip_count += 1
                if skip_count > n_skipped:
                    # Exceeded skip limit, terminate cluster
                    break
            
            j += 1
        
        clusters.append(current_cluster)
        i += 1
    
    return clusters