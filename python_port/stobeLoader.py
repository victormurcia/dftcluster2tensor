import re
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def extract_energy_information(lines):
    """
    Extracts energy information from the given lines and returns it as a dictionary.
    """
    energies = {
        "Total energy (H)": None,
        "Nuc-nuc energy (H)": None,
        "El-nuc energy (H)": None,
        "Kinetic energy (H)": None,
        "Coulomb energy (H)": None,
        "Ex-cor energy (H)": None,
        "Orbital energy core hole (H)": None,
        "Orbital energy core hole (eV)": None,
        "Rigid spectral shift (eV)": None,
        "Ionization potential (eV)": None,
    }
    
    for line in lines:
        if "Total energy   (H)" in line:
            match = re.search(r"Total energy\s+\(H\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if match:
                energies["Total energy (H)"] = float(match.group(1))
        elif "Nuc-nuc energy (H)" in line:
            match = re.search(r"Nuc-nuc energy\s+\(H\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if match:
                energies["Nuc-nuc energy (H)"] = float(match.group(1))
        elif "El-nuc energy  (H)" in line:
            match = re.search(r"El-nuc energy\s+\(H\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if match:
                energies["El-nuc energy (H)"] = float(match.group(1))
        elif "Kinetic energy (H)" in line:
            match = re.search(r"Kinetic energy\s+\(H\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if match:
                energies["Kinetic energy (H)"] = float(match.group(1))
        elif "Coulomb energy (H)" in line:
            match = re.search(r"Coulomb energy\s+\(H\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if match:
                energies["Coulomb energy (H)"] = float(match.group(1))
        elif "Ex-cor energy  (H)" in line:
            match = re.search(r"Ex-cor energy\s+\(H\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if match:
                energies["Ex-cor energy (H)"] = float(match.group(1))
        elif "Orbital energy core hole" in line:
            match = re.search(r"Orbital energy core hole\s*=\s*([-+]?[0-9]*\.?[0-9]+)\s*H\s*\(\s*([-+]?[0-9]*\.?[0-9]+)\s*eV\s*\)", line)
            if match:
                energies["Orbital energy core hole (H)"] = float(match.group(1))
                energies["Orbital energy core hole (eV)"] = float(match.group(2))
        elif "Rigid spectral shift" in line:
            match = re.search(r"Rigid spectral shift\s*=\s*([-+]?[0-9]*\.?[0-9]+)\s*eV", line)
            if match:
                energies["Rigid spectral shift (eV)"] = float(match.group(1))
        elif "Ionization potential" in line:
            match = re.search(r"Ionization potential\s*=\s*([-+]?[0-9]*\.?[0-9]+)\s*eV", line)
            if match:
                energies["Ionization potential (eV)"] = float(match.group(1))
    
    return energies

def parse_basis_line(line):
    """
    Parses a line of text to extract the atom and basis set information.
    Only processes lines that start with "Atom " and have element identifiers (not just numbers).
    """
    line = line.strip()
    
    # Only process lines that start with "Atom " and contain a colon
    if line.startswith("Atom ") and ":" in line:
        try:
            parts = line.split(':', 1)
            if len(parts) == 2:
                # Extract atom identifier (everything after "Atom " and before ":")
                atom_part = parts[0].replace("Atom ", "").strip()
                basis_part = parts[1].strip()
                
                # Key fix: Only accept atom identifiers that contain letters (element symbols)
                # This excludes pure numbers like "1", "2", "3" from exchange/correlation section
                # and only accepts identifiers like "C1", "Cu1", "N1", etc.
                if atom_part and re.match(r'^[A-Za-z]+\d*$', atom_part):
                    return atom_part, basis_part
        except (ValueError, IndexError):
            pass
    
    return None, None

def extract_all_information(file_path, originating_atom):
    """
    Extracts orbital data, basis sets, energy information, x-ray transition data, and atomic coordinates from the given file.
    
    Parameters:
    file_path (str): The path to the file containing the calculation results.
    originating_atom (str): The atom from which the data is extracted.
    
    Returns:
    tuple: A tuple containing DataFrames: df_alpha, df_beta, df_auxiliary, df_orbital, df_model, df_energies, df_xray_transitions, df_atomic_coordinates.
    """
    data_alpha = []
    data_beta = []
    auxiliary_basis = []
    orbital_basis = []
    model_potential = []
    xray_transitions = []
    atomic_coordinates = []
    first_xray_energy = None  # Initialize variable to store first X-ray transition energy

    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    energies = extract_energy_information(lines)
    energies["Atom"] = originating_atom
    energies["Calculation Type"] = "TP" if "tp.out" in file_path else "GND" if "gnd.out" in file_path else "EXC" if "exc.out" in file_path else None

    start_index = None
    end_index = None
    xray_start = False
    atomic_start_index = None
    atomic_end_index = None
    current_section = None

    for i, line in enumerate(lines):
        if "         Spin alpha                              Spin beta" in line:
            start_index = i + 2
        elif " IV)" in line:
            end_index = i
        elif "I)  AUXILIARY BASIS SETS" in line:
            current_section = "auxiliary"
        elif "II)  ORBITAL BASIS SETS" in line:
            current_section = "orbital"
        elif "III)  MODEL POTENTIALS" in line:
            current_section = "model"
        elif current_section in ["auxiliary", "orbital", "model"]:
            # Check for section end markers or next section start
            if ("BASIS DIMENSIONS" in line or 
                "IV)" in line or 
                "WARNING!" in line or
                line.strip() == "" or
                (current_section == "auxiliary" and "II)" in line) or
                (current_section == "orbital" and "III)" in line)):
                # Reset section if we hit a boundary
                if "II)" in line:
                    current_section = "orbital"
                elif "III)" in line:
                    current_section = "model"
                elif ("BASIS DIMENSIONS" in line or "IV)" in line or "WARNING!" in line):
                    current_section = None
            else:
                # Only parse lines that look like atom definitions
                atom, basis = parse_basis_line(line)
                if atom and basis:
                    if current_section == "auxiliary":
                        auxiliary_basis.append([atom, basis])
                    elif current_section == "orbital":
                        orbital_basis.append([atom, basis])
                    elif current_section == "model":
                        model_potential.append([atom, basis])
        elif "           E (eV)   OSCL       oslx       osly       oslz         osc(r2)       <r2>" in line:
            xray_start = True
        elif xray_start and "-----" in line:
            continue
        elif xray_start and line.strip() and line.startswith(" #"):
            try:
                index = int(line[2:6].strip())
                e_ev = float(line[7:17].strip())
                oscl = float(line[18:28].strip())
                oslx = float(line[29:39].strip())
                osly = float(line[40:50].strip())
                oslz = float(line[51:61].strip())
                osc_r2 = float(line[62:72].strip())
                r2 = float(line[73:].strip())
                
                # Capture the first X-ray transition energy for LUMO_En
                if index == 1 and first_xray_energy is None:
                    first_xray_energy = e_ev
                
                xray_transitions.append({
                    "Index": index,
                    "E": e_ev,
                    "OS": oscl,
                    "osx": oslx,
                    "osy": osly,
                    "osz": oslz,
                    "osc(r2)": osc_r2,
                    "<r2>": r2
                })
            except ValueError as e:
                print(f"Error parsing line: {line}\nError: {e}")
        elif "Single image calculation (Angstrom):" in line:
            atomic_start_index = i + 3
        elif 'Smallest atom distance' in line:
            atomic_end_index = i

    if start_index is not None and end_index is not None:
        for line in lines[start_index:end_index]:
            if line.strip() == "":
                continue
            components = [x for x in line.split() if x.strip()]
            if len(components) >= 9:
                mo_index_alpha, occup_alpha, energy_alpha, sym_alpha = components[:4]
                mo_index_beta, occup_beta, energy_beta, sym_beta = components[5:9]
                mo_index_alpha = mo_index_alpha.strip(')')
                mo_index_beta = mo_index_beta.strip(')')
                data_alpha.append({"MO_Index": mo_index_alpha, "Occup.": occup_alpha, "Energy(eV)": energy_alpha, "Sym.": sym_alpha})
                data_beta.append({"MO_Index": mo_index_beta, "Occup.": occup_beta, "Energy(eV)": energy_beta, "Sym.": sym_beta})

    if atomic_start_index is not None and atomic_end_index is not None:
        atomic_coordinates_lines = lines[atomic_start_index:atomic_end_index]
        for line in atomic_coordinates_lines:
            if line.strip() and not any(col in line for col in ['Atom', 'x', 'y', 'z', 'q', 'nuc', 'mass', 'neq', 'grid', 'grp']):  # Skip empty lines and the header
                split_line = line.split()
                if len(split_line) >= 11:
                    atom_info = split_line[1]  # Use the atom type and number
                    atomic_coordinates.append([atom_info] + split_line[2:11])

    # Add the first X-ray transition energy to the energies dictionary
    energies["LUMO_En"] = first_xray_energy

    df_alpha = pd.DataFrame(data_alpha)
    df_beta = pd.DataFrame(data_beta)
    df_auxiliary = pd.DataFrame(auxiliary_basis, columns=['Atom', 'Auxiliary Basis'])
    df_orbital = pd.DataFrame(orbital_basis, columns=['Atom', 'Orbital Basis'])
    df_model = pd.DataFrame(model_potential, columns=['Atom', 'Model Potential'])
    df_energies = pd.DataFrame([energies])
    df_xray_transitions = pd.DataFrame(xray_transitions)
    df_atomic_coordinates = pd.DataFrame(atomic_coordinates, columns=['Atom', 'x', 'y', 'z', 'q', 'nuc', 'mass', 'neq', 'grid', 'grp'])

    numeric_columns = ['x', 'y', 'z', 'q', 'nuc', 'mass', 'neq', 'grid', 'grp']
    df_atomic_coordinates[numeric_columns] = df_atomic_coordinates[numeric_columns].apply(pd.to_numeric, errors='coerce')
        
    return df_alpha, df_beta, df_auxiliary, df_orbital, df_model, df_energies, df_xray_transitions, df_atomic_coordinates

def process_file(file_info):
    """
    Processes a file to extract various pieces of information and combine them into DataFrames.
    """
    try:
        entry, file_path = file_info
        originating_atom = entry
        df_alpha, df_beta, df_auxiliary, df_orbital, df_model, df_energies, df_xray_transitions, df_atomic_coordinates = extract_all_information(file_path, originating_atom)
    
        file_name = os.path.basename(file_path)
        df_auxiliary['Originating File'] = file_name
        df_orbital['Originating File'] = file_name
        df_model['Originating File'] = file_name
        df_alpha['Originating File'] = file_name
        df_beta['Originating File'] = file_name
        df_energies['Originating File'] = file_name
        df_xray_transitions['Originating File'] = file_name
        df_atomic_coordinates['Originating File'] = file_name
        df_combined = df_auxiliary.merge(df_orbital, on=['Atom', 'Originating File'], how='outer').merge(df_model, on=['Atom', 'Originating File'], how='outer')
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

    return df_combined, df_energies, df_alpha, df_beta, df_xray_transitions, df_atomic_coordinates

def extract_atom_id(filename: str) -> str:
    """Extract atom identifier from filename."""
    base = os.path.basename(filename).lower()
    match = re.match(r"([a-z]+[0-9]+)", base)
    return match.group(1).upper() if match else base.split('.')[0].upper()

def sort_dataframe_naturally(df, column):
    """Sorts a DataFrame naturally by the specified column."""
    try:
        from natsort import natsorted
        df[column] = df[column].astype(str)
        sorted_index = natsorted(df[column].tolist())
        df = df.set_index(column).loc[sorted_index].reset_index()
    except ImportError:
        print("Warning: natsort not available, using standard sort")
        df = df.sort_values(column)
    return df

def process_directory(directory, width1, width2, ewid1, ewid2, verbose=True, n_jobs=None):
    """
    Processes all relevant .out files in GND, EXC, TP, and NEXAFS folders using filenames to identify the atom.
    
    Parameters:
    -----------
    directory : str
        Path to directory containing StoBe output folders
    width1, width2 : float
        Gaussian widths for spectral broadening (sigma values)
    ewid1, ewid2 : float
        Energy range for broadening transition
    verbose : bool
        Whether to print progress information
    n_jobs : int, optional
        Number of parallel processes to use (default: use all available cores)
        
    Returns:
    --------
    tuple : (basis_sets, energy_results, orbital_alpha, orbital_beta, xray_transitions, atomic_coordinates)
    """
    import numpy as np
    
    valid_folders = {'GND', 'EXC', 'TP', 'NEXAFS'}
    valid_suffixes = {'gnd.out', 'exc.out', 'tp.out', 'nexafs.out'}
    file_paths = []

    # Discover files
    for folder in valid_folders:
        folder_path = os.path.join(directory, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(suffix) for suffix in valid_suffixes):
                atom_label = extract_atom_id(file)
                full_path = os.path.join(folder_path, file)
                file_paths.append((atom_label, full_path))

    combined_results_list = []
    energy_results_list = []
    orbital_alpha_list = []
    orbital_beta_list = []
    xray_transitions_list = []
    atomic_coordinates_list = []

    total_files = len(file_paths)
    if verbose:
        print(f'Total files to process: {total_files}')
    
    processed_files = 0

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for result in executor.map(process_file, file_paths):
            if result is not None:
                df_combined, df_energies, df_alpha, df_beta, df_xray_transitions, df_atomic_coordinates = result
                combined_results_list.append(df_combined)
                energy_results_list.append(df_energies)
                orbital_alpha_list.append(df_alpha)
                orbital_beta_list.append(df_beta)
                xray_transitions_list.append(df_xray_transitions)
                atomic_coordinates_list.append(df_atomic_coordinates)

            processed_files += 1
            if verbose and processed_files % 10 == 0:
                percentage_complete = (processed_files / total_files) * 100
                print(f'Processing: {percentage_complete:.1f}% completed ({processed_files}/{total_files})')

    # Combine all results
    combined_results = pd.concat(combined_results_list, ignore_index=True) if combined_results_list else pd.DataFrame()
    energy_results = pd.concat(energy_results_list, ignore_index=True) if energy_results_list else pd.DataFrame()
    orbital_alpha = pd.concat(orbital_alpha_list, ignore_index=True) if orbital_alpha_list else pd.DataFrame()
    orbital_beta = pd.concat(orbital_beta_list, ignore_index=True) if orbital_beta_list else pd.DataFrame()
    xray_transitions = pd.concat(xray_transitions_list, ignore_index=True) if xray_transitions_list else pd.DataFrame()
    atomic_coordinates = pd.concat(atomic_coordinates_list, ignore_index=True) if atomic_coordinates_list else pd.DataFrame()

    # Sort and clean energy results
    if not energy_results.empty:
        energy_results = sort_dataframe_naturally(energy_results, 'Atom')
        energy_results['Atom'] = energy_results['Atom'].str.upper()
        energy_results = energy_results.drop_duplicates().reset_index(drop=True)

    # Add atom identifiers to dataframes
    for df_name, df in [('orbital_alpha', orbital_alpha), ('orbital_beta', orbital_beta), ('xray_transitions', xray_transitions)]:
        if not df.empty:
            df['Atom'] = df['Originating File'].apply(extract_atom_id)
            df = df[['Atom'] + [col for col in df.columns if col != 'Atom']]
            if df_name == 'orbital_alpha':
                orbital_alpha = df
            elif df_name == 'orbital_beta':
                orbital_beta = df
            elif df_name == 'xray_transitions':
                xray_transitions = df

    # Apply spectral broadening if X-ray transitions exist
    if not xray_transitions.empty and 'E' in xray_transitions.columns:
        def broad(E):
            if E < ewid1:
                return width1
            elif E > ewid2:
                return width2
            else:
                return width1 + (width2 - width1) * (E - ewid1) / (ewid2 - ewid1)

        xray_transitions['width'] = xray_transitions['E'].apply(broad)

        # Calculate normalized oscillator strength components and angles
        mag = np.sqrt(xray_transitions['osx']**2 + xray_transitions['osy']**2 + xray_transitions['osz']**2)
        # Avoid division by zero
        mag = np.where(mag == 0, 1, mag)
        
        xray_transitions['normalized_osx'] = xray_transitions['osx'] / mag
        xray_transitions['normalized_osy'] = xray_transitions['osy'] / mag
        xray_transitions['normalized_osz'] = xray_transitions['osz'] / mag
        xray_transitions['normalized_magnitude'] = np.sqrt(
            xray_transitions['normalized_osx']**2 +
            xray_transitions['normalized_osy']**2 +
            xray_transitions['normalized_osz']**2
        )
        
        # Calculate angles with respect to z-axis
        dot = xray_transitions['normalized_osz']
        theta = np.arccos(np.clip(dot, -1, 1))  # Clip to avoid numerical errors
        theta_deg = np.degrees(theta)
        theta_deg = np.where(theta_deg > 90, 180 - theta_deg, theta_deg)
        xray_transitions['theta'] = theta_deg

    # Restructure energy results and apply corrections
    if not energy_results.empty:
        restructured_energy_results = restructure_energies_dataframe(energy_results)
        
        if verbose:
            print("Available energy columns:", restructured_energy_results.columns.tolist())

        # Apply energy corrections to xray_transitions if both dataframes exist and have data
        if not xray_transitions.empty and not restructured_energy_results.empty:
            if verbose:
                print("Applying Energy Corrections")
            xray_transitions = apply_energy_corrections(xray_transitions, restructured_energy_results, verbose=verbose)
            
            # Recalculate broadening with corrected energies
            if 'E' in xray_transitions.columns:
                xray_transitions['width'] = xray_transitions['E'].apply(broad)
    else:
        restructured_energy_results = pd.DataFrame()

    if verbose:
        print(f'Processing completed. Found {len(restructured_energy_results)} unique atoms with {len(xray_transitions)} total transitions.')

    return combined_results, restructured_energy_results, orbital_alpha, orbital_beta, xray_transitions, atomic_coordinates

def restructure_energies_dataframe(df_energies):
    """
    Restructures the energies dataframe from having 3 rows per atom (one for each calculation type)
    to having 1 row per atom with prefixed columns for each calculation type.
    
    Parameters:
    -----------
    df_energies : pd.DataFrame
        Original dataframe with columns including 'Atom', 'Calculation Type', 
        and various energy columns
    
    Returns:
    --------
    pd.DataFrame : Restructured dataframe with one row per atom and prefixed columns
    """
    if df_energies.empty:
        return df_energies
    
    # Get the list of columns to restructure (exclude 'Atom' and 'Calculation Type')
    columns_to_restructure = [col for col in df_energies.columns 
                            if col not in ['Atom', 'Calculation Type']]
    
    # Create an empty list to store the restructured data
    restructured_data = []
    
    # Get unique atoms
    unique_atoms = df_energies['Atom'].unique()
    
    for atom in unique_atoms:
        atom_data = df_energies[df_energies['Atom'] == atom]
        
        # Initialize the row dictionary with the atom name
        new_row = {'Atom': atom}
        
        # For each calculation type, add prefixed columns
        for _, row in atom_data.iterrows():
            calc_type = row['Calculation Type']
            if calc_type:  # Only process if calculation type is not None
                for col in columns_to_restructure:
                    prefixed_col = f"{calc_type}_{col}"
                    new_row[prefixed_col] = row[col]
        
        restructured_data.append(new_row)
    
    # Create the new dataframe
    df_restructured = pd.DataFrame(restructured_data)
    
    # Clean up by removing redundant columns - keep only TP versions of these specific columns
    columns_to_remove = []
    cleanup_columns = ['Ionization potential (eV)', 'Orbital energy core hole (H)', 
                      'Orbital energy core hole (eV)', 'LUMO_En', 'Rigid spectral shift (eV)']
    
    for col in cleanup_columns:
        # Remove GND and EXC versions, keep only TP versions
        gnd_col = f'GND_{col}'
        exc_col = f'EXC_{col}'
        
        if gnd_col in df_restructured.columns:
            columns_to_remove.append(gnd_col)
        if exc_col in df_restructured.columns:
            columns_to_remove.append(exc_col)
    
    # Drop the redundant columns
    df_restructured = df_restructured.drop(columns=columns_to_remove, errors='ignore')
    df_restructured_final = calculate_energy_correction_restructured(df_restructured)
    
    return df_restructured_final

def calculate_energy_correction_restructured(df_restructured):
    """
    Calculate Energy_Correction for the restructured dataframe.
    Formula: EXC_Total energy - GND_Total energy - TP_LUMO_En
    Converts Hartree columns to eV and performs calculation.
    
    Parameters:
    -----------
    df_restructured : pd.DataFrame
        Restructured dataframe with prefixed columns
    
    Returns:
    --------
    pd.DataFrame : Dataframe with added 'Energy_Correction (eV)' column
    """
    df_result = df_restructured.copy()
    
    # Define required columns
    gnd_col = 'GND_Total energy (H)'
    exc_col = 'EXC_Total energy (H)'
    lumo_col = 'TP_LUMO_En'
    
    # Check if required columns exist
    required_columns = [gnd_col, exc_col, lumo_col]
    missing_columns = [col for col in required_columns if col not in df_result.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns for energy correction calculation: {missing_columns}")
        df_result['Energy_Correction (eV)'] = None
        return df_result
    
    # Conversion factor: 1 Hartree = 27.2114 eV
    hartree_to_ev = 27.2114
    
    # Calculate energy correction
    def calculate_correction(row):
        exc_energy_h = row.get(exc_col)
        gnd_energy_h = row.get(gnd_col)
        lumo_en = row.get(lumo_col)
        
        # Check if all values are available and not None/NaN
        if (exc_energy_h is not None and gnd_energy_h is not None and lumo_en is not None and
            pd.notna(exc_energy_h) and pd.notna(gnd_energy_h) and pd.notna(lumo_en)):
            
            # Convert Hartree to eV
            exc_energy_ev = exc_energy_h * hartree_to_ev
            gnd_energy_ev = gnd_energy_h * hartree_to_ev
            
            # Calculate energy correction
            energy_correction = exc_energy_ev - gnd_energy_ev - lumo_en
            
            return energy_correction
        else:
            return None
    
    df_result['Energy_Correction (eV)'] = df_result.apply(calculate_correction, axis=1)
    
    return df_result

def apply_energy_corrections(xray_transitions_df, energy_results_df, verbose=True):
    """
    Applies energy corrections from the restructured energy dataframe to the 
    transition energies in the xray_transitions dataframe.
    
    Parameters:
    -----------
    xray_transitions_df : pd.DataFrame
        DataFrame containing X-ray transitions with 'E' and 'Atom' columns
    energy_results_df : pd.DataFrame
        Restructured energy DataFrame with 'Atom' and 'Energy_Correction (eV)' columns
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    pd.DataFrame : Modified xray_transitions dataframe with corrected energies and original energies preserved
    """
    if xray_transitions_df.empty or energy_results_df.empty:
        if verbose:
            print("Warning: Cannot apply energy corrections: one or both dataframes are empty")
        return xray_transitions_df
    
    # Check if Energy_Correction column exists
    if 'Energy_Correction (eV)' not in energy_results_df.columns:
        if verbose:
            print("Warning: Energy_Correction (eV) column not found in energy results. Skipping corrections.")
        return xray_transitions_df
    
    # Create a copy to avoid modifying the original
    corrected_df = xray_transitions_df.copy()
    
    # Store original energies before correction
    corrected_df['E_original'] = corrected_df['E'].copy()
    
    # Create a mapping of atom to energy correction
    energy_corrections = energy_results_df.set_index('Atom')['Energy_Correction (eV)'].to_dict()
    
    # Track corrections applied
    corrections_applied = 0
    atoms_without_corrections = []
    
    # Apply corrections for each atom
    for atom in corrected_df['Atom'].unique():
        if atom in energy_corrections:
            correction = energy_corrections[atom]
            if pd.notna(correction):  # Only apply if correction is not NaN
                # Apply correction: corrected_energy = original_energy + correction
                mask = corrected_df['Atom'] == atom
                corrected_df.loc[mask, 'E'] = corrected_df.loc[mask, 'E'] + correction
                corrections_applied += 1
                if verbose:
                    num_transitions = mask.sum()
                    print(f"Applied correction of {correction:.4f} eV to {num_transitions} transitions for atom {atom}")
            else:
                atoms_without_corrections.append(atom)
        else:
            atoms_without_corrections.append(atom)
    
    # Add a column to indicate which energies were corrected
    corrected_df['Energy_Corrected'] = corrected_df['Atom'].map(
        lambda x: x in energy_corrections and pd.notna(energy_corrections.get(x, None))
    )
    
    # Add the correction amount for reference
    corrected_df['Applied_Correction'] = corrected_df['Atom'].map(
        lambda x: energy_corrections.get(x, 0.0) if x in energy_corrections and pd.notna(energy_corrections.get(x, None)) else 0.0
    )
    
    # Report summary
    if verbose:
        print(f"Energy corrections applied successfully!")
        print(f"- Corrections applied to {corrections_applied} unique atoms")
        print(f"- Total transitions corrected: {corrected_df['Energy_Corrected'].sum()}")
        
        if atoms_without_corrections:
            unique_missing = set(atoms_without_corrections)
            print(f"- No corrections available for atoms: {', '.join(unique_missing)}")
    
    return corrected_df

def load_data(directory, width1, width2, max_energy, verbose=True, n_jobs=None):
    """
    Loads data from a directory, processes it, and returns the results as a dictionary.
    Non-Streamlit version of the data loading function.
    
    Parameters:
    -----------
    directory : str
        Path to directory containing StoBe output folders
    width1, width2 : float
        Gaussian widths for spectral broadening (sigma values)
    max_energy : float
        Maximum energy for processing
    verbose : bool
        Whether to print progress information
    n_jobs : int, optional
        Number of parallel processes to use
        
    Returns:
    --------
    dict : Dictionary containing all processed dataframes
    """
    import time
    
    if not os.path.isdir(directory):
        print(f'Invalid directory: {directory}. Please enter a valid directory path.')
        return None
    
    if verbose:
        print(f'Processing directory: {directory}')
    
    start_time = time.time()
    
    # Process the directory
    basis_sets, energy_results, orbital_alpha, orbital_beta, xray_transitions, atomic_coordinates = process_directory(
        directory, width1, width2, 290, max_energy, verbose=verbose, n_jobs=n_jobs
    )
    
    end_time = time.time()
    
    if verbose:
        print(f'Processing completed in {end_time - start_time:.2f} seconds.')
    
    # Create the data dictionary
    data = {
        'basis_sets': basis_sets,
        'energy_results': energy_results,
        'orbital_alpha': orbital_alpha,
        'orbital_beta': orbital_beta,
        'xray_transitions': xray_transitions,
        'atomic_coordinates': atomic_coordinates
    }
    
    # Print summary if verbose
    if verbose:
        print("\n" + "="*50)
        print("PROCESSING COMPLETE - DATA OVERVIEW")
        print("="*50)
        
        # Summary metrics
        unique_atoms = energy_results['Atom'].nunique() if not energy_results.empty else 0
        total_transitions = len(xray_transitions) if not xray_transitions.empty else 0
        alpha_orbitals = len(orbital_alpha) if not orbital_alpha.empty else 0
        atomic_coords = len(atomic_coordinates) if not atomic_coordinates.empty else 0
        
        print(f"Unique Atoms: {unique_atoms}")
        print(f"X-ray Transitions: {total_transitions}")
        print(f"Alpha Orbitals: {alpha_orbitals}")
        print(f"Atomic Coordinates: {atomic_coords}")
        
        # Show energy corrections info if available
        if not xray_transitions.empty and 'Energy_Corrected' in xray_transitions.columns:
            corrected_count = xray_transitions['Energy_Corrected'].sum()
            total_count = len(xray_transitions)
            print(f"Energy corrections applied to {corrected_count}/{total_count} transitions")
        
        print("="*50)
    
    return data

