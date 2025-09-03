import os
import psutil
import re
import pandas as pd
import streamlit as st
from natsort import natsorted
import time
from concurrent.futures import ProcessPoolExecutor
import py3Dmol
from stmol import showmol
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from scipy.integrate import quad
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import plotly.figure_factory as ff
import plotly.colors
from zipfile import ZipFile
from io import BytesIO
import plotly.express as px
from numba import jit, prange
from numba.typed import Dict
from numba.core import types
import numba

# Control numba parallelization
from numba import set_num_threads
set_num_threads(4)  # Limit to 4 threads to prevent CPU overload

def show_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    st.metric("Memory Usage", f"{memory_mb:.1f} MB")
            
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
    import re
    
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

def restructure_energies_dataframe(df_energies):
    """
    Restructures the energies dataframe from having 3 rows per atom (one for each calculation type)
    to having 1 row per atom with prefixed columns for each calculation type.
    
    Parameters:
    df_energies (DataFrame): Original dataframe with columns including 'Atom', 'Calculation Type', 
                           and various energy columns
    
    Returns:
    DataFrame: Restructured dataframe with one row per atom and prefixed columns
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
    df_restructured (DataFrame): Restructured dataframe with prefixed columns
    
    Returns:
    DataFrame: Dataframe with added 'Energy_Correction (eV)' column
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

# Also need to fix the process_directory function to handle empty xray_transitions
def process_directory_fixed(directory, progress_bar, progress_text, width1, width2, ewid1, ewid2):
    """
    Fixed version of process_directory with better error handling for empty DataFrames.
    """
    valid_folders = {'GND', 'EXC', 'TP', 'NEXAFS'}
    valid_suffixes = {'gnd.out', 'exc.out', 'tp.out', 'nexafs.out'}
    file_paths = []

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
    st.write(f'Total files to process: {total_files}')
    processed_files = 0

    with ProcessPoolExecutor() as executor:
        for result in executor.map(process_file, file_paths):
            if result is not None:  # Check if processing succeeded
                df_combined, df_energies, df_alpha, df_beta, df_xray_transitions, df_atomic_coordinates = result
                combined_results_list.append(df_combined)
                energy_results_list.append(df_energies)
                orbital_alpha_list.append(df_alpha)
                orbital_beta_list.append(df_beta)
                xray_transitions_list.append(df_xray_transitions)
                atomic_coordinates_list.append(df_atomic_coordinates)

            processed_files += 1
            percentage_complete = min((processed_files / total_files), 1.0)
            progress_bar.progress(percentage_complete)
            progress_text.text(f'Processing: {percentage_complete*100:.2f}% completed.')

    # Combine results, handling empty lists
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

    # Process other DataFrames only if they're not empty
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

    # Apply broadening function only if xray_transitions has data and 'E' column exists
    if not xray_transitions.empty and 'E' in xray_transitions.columns:
        def broad(E):
            if E < ewid1:
                return width1
            elif E > ewid2:
                return width2
            else:
                return width1 + (width2 - width1) * (E - ewid1) / (ewid2 - ewid1)

        xray_transitions['width'] = xray_transitions['E'].apply(broad)

        # Calculate normalized oscillator strengths and angles
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
        dot = xray_transitions['normalized_osz']
        theta = np.arccos(np.clip(dot, -1, 1))  # Clip to avoid numerical errors
        theta_deg = np.degrees(theta)
        theta_deg = np.where(theta_deg > 90, 180 - theta_deg, theta_deg)
        xray_transitions['theta'] = theta_deg
    else:
        st.warning("No X-ray transitions found or 'E' column missing. Some features may not work properly.")

    return combined_results, energy_results, orbital_alpha, orbital_beta, xray_transitions, atomic_coordinates

def extract_energy_information_enhanced(lines):
    """
    Enhanced energy extraction with better error handling and more comprehensive parsing.
    """
    energies = {}
    
    energy_patterns = {
        "Total energy (H)": r"Total energy\s+\(H\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)",
        "Nuc-nuc energy (H)": r"Nuc-nuc energy\s+\(H\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)",
        "El-nuc energy (H)": r"El-nuc energy\s+\(H\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)",
        "Kinetic energy (H)": r"Kinetic energy\s+\(H\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)",
        "Coulomb energy (H)": r"Coulomb energy\s+\(H\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)",
        "Ex-cor energy (H)": r"Ex-cor energy\s+\(H\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)",
    }
    
    for line in lines:
        for energy_name, pattern in energy_patterns.items():
            match = re.search(pattern, line)
            if match:
                try:
                    energies[energy_name] = float(match.group(1))
                    # Also create eV version
                    if "(H)" in energy_name:
                        ev_name = energy_name.replace("(H)", "(eV)")
                        energies[ev_name] = float(match.group(1)) * 27.2114
                except ValueError as e:
                    print(f"Error converting energy value {match.group(1)} for {energy_name}: {e}")
    
    return energies

def sort_dataframe_naturally(df, column):
    """
    Sorts a DataFrame naturally by the specified column.
    """
    df[column] = df[column].astype(str)
    sorted_index = natsorted(df[column].tolist())
    df = df.set_index(column).loc[sorted_index].reset_index()
    return df

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
    base = os.path.basename(filename).lower()
    match = re.match(r"([a-z]+[0-9]+)", base)
    return match.group(1).upper() if match else base.split('.')[0].upper()

def process_directory(directory, progress_bar, progress_text, width1, width2, ewid1, ewid2):
    """
    Processes all relevant .out files in GND, EXC, TP, and NEXAFS folders using filenames to identify the atom.
    """
    valid_folders = {'GND', 'EXC', 'TP', 'NEXAFS'}
    valid_suffixes = {'gnd.out', 'exc.out', 'tp.out', 'nexafs.out'}
    file_paths = []

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
    st.write(f'Total files to process: {total_files}')
    processed_files = 0

    with ProcessPoolExecutor() as executor:
        for result in executor.map(process_file, file_paths):
            df_combined, df_energies, df_alpha, df_beta, df_xray_transitions, df_atomic_coordinates = result
            combined_results_list.append(df_combined)
            energy_results_list.append(df_energies)
            orbital_alpha_list.append(df_alpha)
            orbital_beta_list.append(df_beta)
            xray_transitions_list.append(df_xray_transitions)
            atomic_coordinates_list.append(df_atomic_coordinates)

            processed_files += 1
            percentage_complete = min((processed_files / total_files), 1.0)
            progress_bar.progress(percentage_complete)
            progress_text.text(f'Processing: {percentage_complete*100:.2f}% completed.')

    combined_results = pd.concat(combined_results_list, ignore_index=True)
    energy_results = pd.concat(energy_results_list, ignore_index=True)
    orbital_alpha = pd.concat(orbital_alpha_list, ignore_index=True)
    orbital_beta = pd.concat(orbital_beta_list, ignore_index=True)
    xray_transitions = pd.concat(xray_transitions_list, ignore_index=True)
    atomic_coordinates = pd.concat(atomic_coordinates_list, ignore_index=True)

    energy_results = sort_dataframe_naturally(energy_results, 'Atom')
    energy_results['Atom'] = energy_results['Atom'].str.upper()
    energy_results = energy_results.drop_duplicates().reset_index(drop=True)

    for df_name, df in [('orbital_alpha', orbital_alpha), ('orbital_beta', orbital_beta), ('xray_transitions', xray_transitions)]:
        df['Atom'] = df['Originating File'].apply(extract_atom_id)
        df = df[['Atom'] + [col for col in df.columns if col != 'Atom']]
        if df_name == 'orbital_alpha':
            orbital_alpha = df
        elif df_name == 'orbital_beta':
            orbital_beta = df
        elif df_name == 'xray_transitions':
            xray_transitions = df

    def broad(E):
        if E < ewid1:
            return width1
        elif E > ewid2:
            return width2
        else:
            return width1 + (width2 - width1) * (E - ewid1) / (ewid2 - ewid1)

    xray_transitions['width'] = xray_transitions['E'].apply(broad)

    mag = np.sqrt(xray_transitions['osx']**2 + xray_transitions['osy']**2 + xray_transitions['osz']**2)
    xray_transitions['normalized_osx'] = xray_transitions['osx'] / mag
    xray_transitions['normalized_osy'] = xray_transitions['osy'] / mag
    xray_transitions['normalized_osz'] = xray_transitions['osz'] / mag
    xray_transitions['normalized_magnitude'] = np.sqrt(
        xray_transitions['normalized_osx']**2 +
        xray_transitions['normalized_osy']**2 +
        xray_transitions['normalized_osz']**2
    )
    dot = xray_transitions['normalized_osz']
    theta = np.arccos(dot)
    theta_deg = np.degrees(theta)
    theta_deg = np.where(theta_deg > 90, 180 - theta_deg, theta_deg)
    xray_transitions['theta'] = theta_deg
    restructured_energy_results = restructure_energies_dataframe(energy_results)

    print(restructured_energy_results.columns)

    return combined_results, restructured_energy_results, orbital_alpha, orbital_beta, xray_transitions, atomic_coordinates

def dataframe_to_xyz(df, file_name="molecule.xyz"):
    """
    Convert a DataFrame to XYZ format file.
    """
    lines = []
    lines.append(str(len(df)))
    lines.append("Generated by dataframe_to_xyz")
    for idx, row in df.iterrows():
        atom_symbol = ''.join([i for i in row['Atom'] if not i.isdigit()])
        lines.append(f"{atom_symbol} {row['x']} {row['y']} {row['z']}")
    
    with open(file_name, 'w') as f:
        f.write("\n".join(lines))
    print(f"XYZ file '{file_name}' created successfully.")

def visualize_xyz_with_stmol(df, file_name, label_size=14, bond_width=0.1, atom_scale=0.3):
    """
    Visualize an XYZ file using Stmol and py3Dmol.
    """
    with open(file_name, 'r') as f:
        xyz_data = f.read()
    
    view = py3Dmol.view(width=800, height=600)
    view.addModel(xyz_data, 'xyz')
    view.setStyle({'stick': {'radius': bond_width}, 'sphere': {'scale': atom_scale}})
    view.setBackgroundColor('black')
    
    atom_counters = {}
    for i, row in df.iterrows():
        atom_symbol = ''.join([i for i in row['Atom'] if not i.isdigit()])
        if atom_symbol not in atom_counters:
            atom_counters[atom_symbol] = 0
        atom_counters[atom_symbol] += 1
        label = f"{atom_symbol}{atom_counters[atom_symbol]}"
        x, y, z = row['x'], row['y'], row['z']
        view.addLabel(label, {'position': {'x': x, 'y': y, 'z': z}, 'backgroundColor': 'black', 'fontColor': 'white', 'fontSize': label_size})
    
    view.zoomTo()
    return view

def plot_individual_spectra(xray_transitions, E_max):
    """
    Generates and plots spectra for each unique atom in the xray_transitions DataFrame.
    """
    unique_atoms = natsorted(xray_transitions['Atom'].unique())
    
    num_atoms = len(unique_atoms)
    num_cols = 4
    num_rows = (num_atoms + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10))
    axs = axs.flatten()

    for i, atom in enumerate(unique_atoms):
        filtered_df = xray_transitions[xray_transitions['Atom'] == atom]
        
        E_min = filtered_df['E'].min() - 10
        E_range = np.linspace(E_min, E_max, 2000)

        spectrum = np.zeros_like(E_range)

        E_values = filtered_df['E'].values
        width_values = filtered_df['width'].values
        amplitude_values = filtered_df['OS'].values
        
        E_range_2D = E_range[:, np.newaxis]
        
        gaussians = (amplitude_values / (width_values * np.sqrt(2 * np.pi))) * \
                    np.exp(-((E_range_2D - E_values) ** 2) / (2 * width_values ** 2))
        
        spectrum = np.sum(gaussians, axis=1)

        ax1 = axs[i]
        ax1.plot(E_range, spectrum, label='Spectrum')
        ax1.set_xlabel('Energy (eV)')
        ax1.set_ylabel('Intensity')
        ax1.set_xlim([E_min, E_max])
        ax1.set_title(f'Spectrum for {atom}')
        
        ax2 = ax1.twinx()
        ax2.vlines(x=E_values, ymin=0, ymax=amplitude_values, color='r')
        ax2.set_ylabel('OS')
        ax2.set_xlim([E_min, E_max])
        
        if not filtered_df.empty:
            custom_line = Line2D([0], [0], color='r', lw=2, label='OS')
        
        ax1.legend(loc='upper left')
        if not filtered_df.empty:
            ax2.legend(handles=[custom_line], loc='upper right')

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.tight_layout()
    st.pyplot(fig)

def plot_total_spectra(xray_transitions, E_max, molName, os_col, os_ylim=1.0):
    """
    Generates and plots the total spectra from the sum of all the Gaussians for each unique atom in the xray_transitions DataFrame.
    """
    E_min = xray_transitions['E'].min() - 1
    E_range = np.linspace(E_min, E_max, 2000)
    
    total_spectrum = np.zeros_like(E_range)

    unique_atoms = xray_transitions['Atom'].unique()

    for atom in unique_atoms:
        filtered_df = xray_transitions[xray_transitions['Atom'] == atom]
        
        E_values = filtered_df['E'].values
        width_values = filtered_df['width'].values
        amplitude_values = filtered_df[os_col].values
        
        E_range_2D = E_range[:, np.newaxis]
        
        gaussians = (amplitude_values / (width_values * np.sqrt(2 * np.pi))) * \
                    np.exp(-((E_range_2D - E_values) ** 2) / (2 * width_values ** 2))
        
        total_spectrum += np.sum(gaussians, axis=1)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(E_range, total_spectrum, label='Total Spectrum')
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Intensity')
    ax1.set_xlim([E_min, E_max])
    ax1.set_title(f'Total Spectrum from {molName}')
    
    ax2 = ax1.twinx()
    
    for atom in unique_atoms:
        filtered_df = xray_transitions[xray_transitions['Atom'] == atom]
        E_values = filtered_df['E'].values
        amplitude_values = filtered_df['OS'].values
        ax2.vlines(x=E_values, ymin=0, ymax=amplitude_values, color='r')

    ax2.set_ylabel('OS')
    ax2.set_xlim([E_min, E_max])
    if os_col == 'normalized_os':
        ax2.set_ylim([0, os_ylim])
    
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    
    if not xray_transitions.empty:
        custom_line = Line2D([0], [0], color='r', lw=2, label='OS')
        ax2.legend(handles=[custom_line], loc='upper right')

    ax1.legend(loc='upper left')
    
    fig.tight_layout()
    st.pyplot(fig)
    
def find_core_hole_homo_lumo(orbital_alpha):
    """
    Identifies the Core Hole, HOMO, and LUMO for each unique Atom and Originating File.
    """
    results = []
    orbital_alpha['Occup.'] = orbital_alpha['Occup.'].astype(float)

    grouped = orbital_alpha.groupby(['Atom', 'Originating File'])
    
    for (atom, file), group in grouped:
        core_hole, homo, lumo = None, None, None
        
        core_hole_row = group[group['Occup.'] == 0.5]
        if not core_hole_row.empty:
            core_hole = core_hole_row.iloc[0]['MO_Index']
    
        homo_rows = group[group['Occup.'] == 1.0]
        if not homo_rows.empty:
            homo_index = homo_rows.index[-1]
            homo = homo_rows.iloc[-1]['MO_Index']
            lumo_index = homo_index + 1
            if lumo_index in group.index:
                lumo = group.loc[lumo_index]['MO_Index']
        
        results.append({
            'Atom': atom,
            'File': file,
            'Core Hole': core_hole,
            'HOMO': homo,
            'LUMO': lumo
        })
    
    return pd.DataFrame(results)

def plot_density_spectra(xray_transitions, E_max, os_col):
    """
    Generates a 2D density plot for the xray_transitions DataFrame.
    """
    filtered_transitions = xray_transitions[xray_transitions['E'] < E_max]
    
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(
        x=filtered_transitions['E'],
        y=filtered_transitions['theta'],
        weights=filtered_transitions[os_col],
        fill=True,
        cmap="viridis",
        ax=ax
    )
    
    ax.set_xlim([filtered_transitions['E'].min()-10, E_max+10])
    ax.set_ylim([-50, 125])
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Theta (degrees)')
    ax.set_title('2D KDE Plot of X-ray Transitions')

    fig.tight_layout()
    st.pyplot(fig)

def filter_and_normalize_xray(df, maxE, OST):
    """
    Filters the dataframe based on the maxE and OST parameters, and adds a normalized_os column.
    """
    df_filtered = df[df['E'] <= maxE]

    max_os = df_filtered['OS'].max()
    df_filtered['normalized_os'] = df_filtered['OS'] / max_os

    df_filtered = df_filtered[df_filtered['normalized_os'] >= OST / 100]
    df_filtered = df_filtered.sort_values(by='E')
    df_filtered = df_filtered.reset_index(drop=True)
    
    return df_filtered

# Analytical Gaussian overlap - MUCH faster than numerical integration
@jit(nopython=True)
def analytical_gaussian_overlap(mu1, sigma1, amp1, mu2, sigma2, amp2):
    """
    Analytical calculation of overlap between two normalized Gaussians.
    """
    if abs(mu1 - mu2) < 1e-10 and abs(sigma1 - sigma2) < 1e-10:
        return min(amp1, amp2)
    
    sigma_combined_sq = sigma1*sigma1 + sigma2*sigma2
    sigma_combined = np.sqrt(sigma_combined_sq)
    
    delta_mu = abs(mu1 - mu2)
    
    if delta_mu > 5 * sigma_combined:
        return 0.0
    
    prefactor = np.sqrt(2 * np.pi) * sigma1 * sigma2 / sigma_combined
    exp_term = np.exp(-(delta_mu * delta_mu) / (2 * sigma_combined_sq))
    
    overlap_coefficient = prefactor * exp_term / (sigma1 * np.sqrt(2 * np.pi))
    overlap_coefficient = min(overlap_coefficient, 1.0)
    
    return min(amp1, amp2) * overlap_coefficient

@jit(nopython=True, parallel=True)
def calculate_overlap_matrix_vectorized(energies, widths, amplitudes, energy_threshold=10.0):
    """
    Vectorized overlap matrix calculation with early termination.
    """
    n = len(energies)
    overlap_matrix = np.zeros((n, n))
    
    for i in prange(n):
        for j in range(i, n):
            if i == j:
                overlap_matrix[i, j] = 100.0
            else:
                energy_diff = abs(energies[i] - energies[j])
                if energy_diff > energy_threshold:
                    overlap_matrix[i, j] = 0.0
                    overlap_matrix[j, i] = 0.0
                else:
                    overlap = analytical_gaussian_overlap(
                        energies[i], widths[i], amplitudes[i],
                        energies[j], widths[j], amplitudes[j]
                    )
                    
                    smaller_area = min(amplitudes[i], amplitudes[j])
                    percent_overlap = (overlap / smaller_area * 100.0) if smaller_area > 0 else 0.0
                    
                    overlap_matrix[i, j] = percent_overlap
                    overlap_matrix[j, i] = percent_overlap
    
    return overlap_matrix

def calculate_percent_overlap_matrix_optimized(df, energy_threshold=10.0, n_jobs=-1):
    """
    Optimized overlap matrix calculation - orders of magnitude faster.
    """
    df = df.sort_values(by='E').reset_index(drop=True)
    n = len(df)
    
    if n == 0:
        return pd.DataFrame()
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    timer_text = st.empty()
    
    start_time = time.time()
    
    energies = df['E'].values.astype(np.float64)
    widths = df['width'].values.astype(np.float64)
    amplitudes = df['OS'].values.astype(np.float64)
    
    progress_text.text('Calculating overlap matrix (analytical method)...')
    overlap_matrix = calculate_overlap_matrix_vectorized(
        energies, widths, amplitudes, energy_threshold
    )
    
    elapsed_time = time.time() - start_time
    progress_bar.progress(1.0)
    progress_text.text(f'Overlap calculation completed: {n}x{n} matrix')
    timer_text.text(f'Time elapsed: {elapsed_time:.2f} seconds')
    
    time.sleep(1)
    progress_bar.empty()
    progress_text.empty()
    timer_text.empty()
    
    return pd.DataFrame(overlap_matrix, index=df.index, columns=df.index)

# Fast IGOR-style clustering implementations
@jit(nopython=True)
def fast_igor_clustering(overlap_matrix_values, threshold):
    """
    Fast, conservative IGOR-style clustering using numba compilation.
    Only clusters immediately adjacent peaks with sufficient overlap.
    """
    n = overlap_matrix_values.shape[0]
    
    if n == 0:
        return Dict.empty(
            key_type=types.int64,
            value_type=types.int64[:]
        )
    
    used_peaks = np.zeros(n, dtype=numba.boolean)
    
    cluster_starts = np.zeros(n, dtype=numba.int64)
    cluster_lengths = np.zeros(n, dtype=numba.int64)
    all_clustered_peaks = np.zeros(n, dtype=numba.int64)
    
    num_clusters = 0
    peak_write_idx = 0
    
    i = 0
    while i < n:
        if used_peaks[i]:
            i += 1
            continue
        
        cluster_start = peak_write_idx
        cluster_length = 0
        
        all_clustered_peaks[peak_write_idx] = i
        used_peaks[i] = True
        peak_write_idx += 1
        cluster_length += 1
        
        j = i + 1
        max_adjacency = min(i + 3, n)
        
        while j < max_adjacency:
            if not used_peaks[j]:
                overlap = overlap_matrix_values[i, j]
                if overlap >= threshold:
                    all_clustered_peaks[peak_write_idx] = j
                    used_peaks[j] = True
                    peak_write_idx += 1
                    cluster_length += 1
                    
                    i = j
                    j = i + 1
                    max_adjacency = min(i + 3, n)
                else:
                    j += 1
            else:
                j += 1
        
        cluster_starts[num_clusters] = cluster_start
        cluster_lengths[num_clusters] = cluster_length
        num_clusters += 1
        
        i += 1
    
    result = Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:]
    )
    
    for cluster_id in range(num_clusters):
        start_idx = cluster_starts[cluster_id]
        length = cluster_lengths[cluster_id]
        
        cluster_peaks = np.empty(length, dtype=numba.int64)
        for i in range(length):
            cluster_peaks[i] = all_clustered_peaks[start_idx + i]
        
        result[cluster_id] = cluster_peaks
    
    return result

@jit(nopython=True)
def ultra_fast_adjacency_clustering(overlap_matrix_values, threshold):
    """
    Ultra-fast clustering that only checks immediate neighbors (i, i+1).
    """
    n = overlap_matrix_values.shape[0]
    
    if n == 0:
        return Dict.empty(
            key_type=types.int64,
            value_type=types.int64[:]
        )
    
    used_peaks = np.zeros(n, dtype=numba.boolean)
    
    cluster_data = np.zeros((n, 2), dtype=numba.int64)
    cluster_sizes = np.zeros(n, dtype=numba.int64)
    num_clusters = 0
    
    for i in range(n):
        if used_peaks[i]:
            continue
        
        cluster_data[num_clusters, 0] = i
        cluster_size = 1
        used_peaks[i] = True
        
        if i + 1 < n and not used_peaks[i + 1]:
            overlap = overlap_matrix_values[i, i + 1]
            if overlap >= threshold:
                cluster_data[num_clusters, 1] = i + 1
                cluster_size = 2
                used_peaks[i + 1] = True
        
        cluster_sizes[num_clusters] = cluster_size
        num_clusters += 1
    
    result = Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:]
    )
    
    for cluster_id in range(num_clusters):
        size = cluster_sizes[cluster_id]
        cluster_peaks = np.empty(size, dtype=numba.int64)
        
        for i in range(size):
            cluster_peaks[i] = cluster_data[cluster_id, i]
        
        result[cluster_id] = cluster_peaks
    
    return result

@jit(nopython=True)
def fast_conservative_clustering(overlap_matrix_values, threshold, max_cluster_size=500):
    """
    Fast conservative clustering with size limits.
    """
    n = overlap_matrix_values.shape[0]
    
    if n == 0:
        return Dict.empty(
            key_type=types.int64,
            value_type=types.int64[:]
        )
    
    used_peaks = np.zeros(n, dtype=numba.boolean)
    
    cluster_data = np.zeros((n, max_cluster_size), dtype=numba.int64)
    cluster_sizes = np.zeros(n, dtype=numba.int64)
    num_clusters = 0
    
    i = 0
    while i < n:
        if used_peaks[i]:
            i += 1
            continue
        
        cluster_data[num_clusters, 0] = i
        cluster_size = 1
        used_peaks[i] = True
        
        current_peak = i
        while cluster_size < max_cluster_size:
            found_adjacent = False
            for offset in range(1, min(3, n - current_peak)):
                candidate = current_peak + offset
                if candidate >= n or used_peaks[candidate]:
                    continue
                
                overlap = overlap_matrix_values[current_peak, candidate]
                if overlap >= threshold:
                    cluster_data[num_clusters, cluster_size] = candidate
                    used_peaks[candidate] = True
                    cluster_size += 1
                    current_peak = candidate
                    found_adjacent = True
                    break
            
            if not found_adjacent:
                break
        
        cluster_sizes[num_clusters] = cluster_size
        num_clusters += 1
        i += 1
    
    result = Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:]
    )
    
    for cluster_id in range(num_clusters):
        size = cluster_sizes[cluster_id]
        cluster_peaks = np.empty(size, dtype=numba.int64)
        
        for i in range(size):
            cluster_peaks[i] = cluster_data[cluster_id, i]
        
        result[cluster_id] = cluster_peaks
    
    return result

def convert_numba_clusters_to_dict(numba_clusters):
    """Convert numba Dict result to regular Python dict for compatibility."""
    result = {}
    for cluster_id, peaks in numba_clusters.items():
        result[int(cluster_id)] = [int(peak) for peak in peaks]
    return result

def sequential_clustering_optimized(df, overlap_matrix, overlap_threshold):
    """
    Optimized sequential clustering with multiple speed options.
    """
    n_peaks = len(df)
    if n_peaks == 0:
        return {}
    
    if n_peaks > 1000:
        numba_result = ultra_fast_adjacency_clustering(
            overlap_matrix.values.astype(np.float64), 
            float(overlap_threshold)
        )
    elif n_peaks > 100:
        numba_result = fast_conservative_clustering(
            overlap_matrix.values.astype(np.float64), 
            float(overlap_threshold),
            max_cluster_size=300
        )
    else:
        numba_result = fast_igor_clustering(
            overlap_matrix.values.astype(np.float64), 
            float(overlap_threshold)
        )
    
    return convert_numba_clusters_to_dict(numba_result)

def hierarchical_clustering(overlap_matrix, overlap_threshold, method='average'):
    """
    Performs hierarchical clustering on the overlap matrix.
    """
    if len(overlap_matrix) == 0:
        return {}
    
    distance_matrix = 100 - overlap_matrix.values
    np.fill_diagonal(distance_matrix, 0)
    
    condensed_distances = squareform(distance_matrix, checks=False)
    
    linkage_matrix = linkage(condensed_distances, method=method)
    
    distance_threshold = 100 - overlap_threshold
    cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
    
    clusters = {}
    for peak_idx, cluster_label in enumerate(cluster_labels):
        cluster_id = cluster_label - 1
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(peak_idx)
    
    return clusters

def create_cluster_summary(df, clusters):
    """
    Creates a summary DataFrame of clusters.
    """
    summary_data = []
    
    for cluster_id, peak_indices in clusters.items():
        cluster_peaks = df.iloc[peak_indices]
        
        summary_data.append({
            'Cluster_ID': cluster_id,
            'Num_Peaks': len(peak_indices),
            'Peak_Indices': peak_indices,
            'Energy_Range': f"{cluster_peaks['E'].min():.2f} - {cluster_peaks['E'].max():.2f}",
            'Total_OS': cluster_peaks['OS'].sum(),
            'Avg_Energy': cluster_peaks['E'].mean(),
            'Energy_Spread': cluster_peaks['E'].std() if len(peak_indices) > 1 else 0.0
        })
    
    return pd.DataFrame(summary_data)

def plot_overlap_heatmap(overlap_matrix, clusters=None):
    """
    Creates a heatmap visualization of the overlap matrix.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(overlap_matrix, annot=False, cmap='viridis', 
                square=True, ax=ax, cbar_kws={'label': 'Overlap (%)'})
    
    ax.set_title('Peak Overlap Matrix')
    ax.set_xlabel('Peak Index')
    ax.set_ylabel('Peak Index')
    
    if clusters:
        for cluster_peaks in clusters.values():
            if len(cluster_peaks) > 1:
                min_idx = min(cluster_peaks)
                max_idx = max(cluster_peaks)
                rect = plt.Rectangle((min_idx-0.5, min_idx-0.5), 
                                   max_idx-min_idx+1, max_idx-min_idx+1,
                                   fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
    
    return fig

def plot_clustered_spectra(df, clusters, E_max):
    """
    Plots individual cluster spectra.
    """
    n_clusters = len(clusters)
    if n_clusters == 0:
        return None
    
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_clusters == 1:
        axs = [axs]
    elif n_rows == 1:
        axs = axs if n_clusters > 1 else [axs]
    else:
        axs = axs.flatten()
    
    E_min = df['E'].min() - 10
    E_range = np.linspace(E_min, E_max, 2000)
    
    for i, (cluster_id, peak_indices) in enumerate(clusters.items()):
        cluster_peaks = df.iloc[peak_indices]
        
        spectrum = np.zeros_like(E_range)
        
        for _, peak in cluster_peaks.iterrows():
            E_val = peak['E']
            width_val = peak['width']
            amplitude_val = peak['OS']
            
            gaussian_contribution = (amplitude_val / (width_val * np.sqrt(2 * np.pi))) * \
                                  np.exp(-((E_range - E_val) ** 2) / (2 * width_val ** 2))
            spectrum += gaussian_contribution
        
        ax = axs[i] if i < len(axs) else None
        if ax is None:
            break
            
        ax.plot(E_range, spectrum, label=f'Cluster {cluster_id}')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Intensity')
        ax.set_xlim([E_min, E_max])
        ax.set_title(f'Cluster {cluster_id} ({len(peak_indices)} peaks)')
        
        ax2 = ax.twinx()
        ax2.vlines(cluster_peaks['E'], 0, cluster_peaks['OS'], colors='red', alpha=0.7)
        ax2.set_ylabel('OS', color='red')
        ax2.set_xlim([E_min, E_max])
    
    for j in range(len(clusters), len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    return fig

def merge_clustered_peaks(df, clusters):
    """
    Merges peaks within each cluster into representative peaks.
    """
    merged_peaks = []
    peak_mapping = {}
    
    for cluster_id, peak_indices in clusters.items():
        cluster_data = df.iloc[peak_indices]
        
        total_os = cluster_data['OS'].sum()
        if total_os > 0:
            weighted_energy = (cluster_data['E'] * cluster_data['OS']).sum() / total_os
        else:
            weighted_energy = cluster_data['E'].mean()
        
        merged_os = total_os
        avg_width = cluster_data['width'].mean()
        
        merged_peak = {
            'E': weighted_energy,
            'OS': merged_os,
            'width': avg_width,
            'Cluster_ID': cluster_id,
            'Original_Peak_Count': len(peak_indices)
        }
        
        for col in df.columns:
            if col not in ['E', 'OS', 'width']:
                if col in ['normalized_os', 'theta']:
                    if col == 'normalized_os':
                        merged_peak[col] = merged_os / df['OS'].max()
                    elif col == 'theta':
                        merged_peak[col] = cluster_data[col].mean()
                else:
                    merged_peak[col] = cluster_data[col].iloc[0]
        
        merged_peaks.append(merged_peak)
        peak_mapping[len(merged_peaks) - 1] = peak_indices
    
    merged_df = pd.DataFrame(merged_peaks)
    merged_df = merged_df.sort_values('E').reset_index(drop=True)
    
    new_peak_mapping = {}
    for new_idx, (old_idx, original_peaks) in enumerate(zip(merged_df.index, peak_mapping.values())):
        new_peak_mapping[new_idx] = original_peaks
    
    return merged_df, new_peak_mapping

def iterative_clustering(df, overlap_threshold, clustering_method='sequential', max_iterations=10, linkage_method='average'):
    """
    Performs iterative clustering until no overlaps exceed the threshold.
    """
    iteration_results = {
        'iterations': [],
        'final_clusters': {},
        'total_iterations': 0,
        'convergence_reason': ''
    }
    
    current_df = df.copy()
    iteration = 0
    
    original_peak_mapping = {i: [i] for i in range(len(df))}
    
    while iteration < max_iterations:
        st.write(f"**Iteration {iteration + 1}**: Processing {len(current_df)} peaks")
        
        overlap_matrix = calculate_percent_overlap_matrix_optimized(
            current_df, 
            energy_threshold=30.0
        )
        
        max_overlap = np.max(overlap_matrix.values[np.triu_indices(len(overlap_matrix), k=1)])
        convergence_reached = max_overlap < overlap_threshold
        
        if convergence_reached:
            clusters = {i: [i] for i in range(len(current_df))}
        else:
            if clustering_method == 'sequential':
                clusters = sequential_clustering_optimized(current_df, overlap_matrix, overlap_threshold)
            else:
                clusters = hierarchical_clustering(overlap_matrix, overlap_threshold, method=linkage_method)
        
        iteration_data = {
            'iteration': iteration + 1,
            'input_peaks': len(current_df),
            'clusters_formed': len(clusters),
            'overlap_matrix': overlap_matrix.copy(),
            'clusters': clusters.copy(),
            'peak_data': current_df.copy(),
            'max_overlap': max_overlap,
            'peak_mapping': original_peak_mapping.copy(),
            'converged': convergence_reached
        }
        
        iteration_results['iterations'].append(iteration_data)
        
        if convergence_reached:
            iteration_results['convergence_reason'] = f"No overlaps exceed {overlap_threshold}% threshold"
            break
        
        if all(len(cluster_peaks) == 1 for cluster_peaks in clusters.values()):
            iteration_results['convergence_reason'] = "No clusters formed - all peaks are singletons"
            break
        
        merged_df, peak_mapping = merge_clustered_peaks(current_df, clusters)
        
        new_original_mapping = {}
        for new_idx, merged_indices in peak_mapping.items():
            original_peaks = []
            for merged_idx in merged_indices:
                original_peaks.extend(original_peak_mapping[merged_idx])
            new_original_mapping[new_idx] = original_peaks
        
        original_peak_mapping = new_original_mapping
        current_df = merged_df
        iteration += 1
        
        if len(current_df) == 1:
            final_overlap_matrix = pd.DataFrame([[100.0]], index=[0], columns=[0])
            final_iteration_data = {
                'iteration': iteration + 1,
                'input_peaks': 1,
                'clusters_formed': 1,
                'overlap_matrix': final_overlap_matrix,
                'clusters': {0: [0]},
                'peak_data': current_df.copy(),
                'max_overlap': 0.0,
                'peak_mapping': original_peak_mapping.copy(),
                'converged': True
            }
            iteration_results['iterations'].append(final_iteration_data)
            iteration_results['convergence_reason'] = "Single peak remaining"
            break
    
    if iteration >= max_iterations:
        iteration_results['convergence_reason'] = f"Maximum iterations ({max_iterations}) reached"
    
    iteration_results['total_iterations'] = len(iteration_results['iterations'])
    iteration_results['final_peaks'] = current_df.copy()
    iteration_results['final_peak_mapping'] = original_peak_mapping
    
    return iteration_results

def plot_iteration_convergence(iteration_results):
    """
    Plots convergence metrics across iterations.
    """
    if not iteration_results['iterations']:
        return None
    
    iterations = [data['iteration'] for data in iteration_results['iterations']]
    input_peaks = [data['input_peaks'] for data in iteration_results['iterations']]
    clusters_formed = [data['clusters_formed'] for data in iteration_results['iterations']]
    max_overlaps = [data['max_overlap'] for data in iteration_results['iterations']]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    ax1.plot(iterations, input_peaks, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Number of Peaks')
    ax1.set_title('Peak Count Reduction')
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(iterations, clusters_formed, alpha=0.7, color='green')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Clusters Formed')
    ax2.set_title('Clusters per Iteration')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(iterations, max_overlaps, 'ro-', linewidth=2, markersize=6)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Maximum Overlap (%)')
    ax3.set_title('Maximum Overlap Trend')
    ax3.grid(True, alpha=0.3)
    
    reduction_ratios = [inp / clust if clust > 0 else 1 for inp, clust in zip(input_peaks, clusters_formed)]
    ax4.bar(iterations, reduction_ratios, alpha=0.7, color='purple')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Peaks per Cluster')
    ax4.set_title('Clustering Efficiency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_iteration_heatmaps(iteration_results, max_display=5):
    """
    Creates a grid of heatmaps showing overlap matrices for each iteration.
    """
    iterations_to_show = iteration_results['iterations'][:max_display]
    n_iterations = len(iterations_to_show)
    
    if n_iterations == 0:
        return None
    
    n_cols = min(3, n_iterations)
    n_rows = (n_iterations + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_iterations == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_iterations > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, iteration_data in enumerate(iterations_to_show):
        ax = axes[i]
        overlap_matrix = iteration_data['overlap_matrix']
        
        im = ax.imshow(overlap_matrix.values, cmap='viridis', aspect='auto')
        
        plt.colorbar(im, ax=ax)
        
        ax.set_title(f'Iteration {iteration_data["iteration"]}\n({iteration_data["input_peaks"]} peaks)')
        ax.set_xlabel('Peak Index')
        ax.set_ylabel('Peak Index')
    
    for j in range(n_iterations, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    return fig

def plot_cluster_summary_interactive(cluster_summary):
    """
    Creates an interactive plot of cluster summary statistics.
    """
    if cluster_summary.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cluster_summary['Avg_Energy'],
        y=cluster_summary['Total_OS'],
        mode='markers+text',
        marker=dict(
            size=cluster_summary['Num_Peaks'] * 10,
            color=cluster_summary['Cluster_ID'],
            colorscale='viridis',
            showscale=True,
            colorbar=dict(title='Cluster ID')
        ),
        text=cluster_summary['Cluster_ID'],
        textposition='middle center',
        name='Clusters',
        hovertemplate='<b>Cluster %{text}</b><br>' +
                     'Avg Energy: %{x:.2f} eV<br>' +
                     'Total OS: %{y:.4f}<br>' +
                     'Num Peaks: %{marker.size}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Cluster Summary: Average Energy vs Total Oscillator Strength',
        xaxis_title='Average Energy (eV)',
        yaxis_title='Total Oscillator Strength',
        height=500
    )
    
    return fig

def plot_iteration_nexafs_spectra(iteration_results, E_max, molName, show_individual=True, show_combined=True):
    """
    Plots NEXAFS spectra for each iteration of the clustering process.
    """
    if not iteration_results['iterations']:
        return None, None
    
    iterations_data = iteration_results['iterations']
    n_iterations = len(iterations_data)
    
    all_energies = []
    for iter_data in iterations_data:
        all_energies.extend(iter_data['peak_data']['E'].values)
    
    E_min = min(all_energies) - 5
    E_range = np.linspace(E_min, E_max, 2000)
    
    individual_fig = None
    combined_fig = None
    
    if show_individual and n_iterations > 0:
        n_cols = min(3, n_iterations)
        n_rows = (n_iterations + n_cols - 1) // n_cols
        
        individual_fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_iterations == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = axes
        elif n_rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i, iter_data in enumerate(iterations_data):
            if i >= len(axes):
                break
                
            ax = axes[i]
            peak_data = iter_data['peak_data']
            
            spectrum = np.zeros_like(E_range)
            
            for _, peak in peak_data.iterrows():
                E_val = peak['E']
                width_val = peak['width']
                amplitude_val = peak['OS']
                
                gaussian_contrib = (amplitude_val / (width_val * np.sqrt(2 * np.pi))) * \
                                 np.exp(-((E_range - E_val) ** 2) / (2 * width_val ** 2))
                spectrum += gaussian_contrib
            
            ax.plot(E_range, spectrum, 'b-', linewidth=2, label=f'Iteration {iter_data["iteration"]}')
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel('Intensity')
            ax.set_xlim([E_min, E_max])
            ax.set_title(f'Iteration {iter_data["iteration"]} NEXAFS\n({iter_data["input_peaks"]} peaks)')
            ax.grid(True, alpha=0.3)
            
            ax2 = ax.twinx()
            ax2.vlines(peak_data['E'], 0, peak_data['OS'], colors='red', alpha=0.7, linewidth=1)
            ax2.set_ylabel('OS', color='red')
            ax2.set_xlim([E_min, E_max])
        
        for j in range(n_iterations, len(axes)):
            individual_fig.delaxes(axes[j])
        
        individual_fig.suptitle(f'{molName} - NEXAFS Spectra Evolution', fontsize=16)
        individual_fig.tight_layout()
    
    if show_combined and n_iterations > 0:
        combined_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_iterations))
        max_intensity = 0
        
        for i, (iter_data, color) in enumerate(zip(iterations_data, colors)):
            peak_data = iter_data['peak_data']
            
            spectrum = np.zeros_like(E_range)
            
            for _, peak in peak_data.iterrows():
                E_val = peak['E']
                width_val = peak['width']
                amplitude_val = peak['OS']
                
                gaussian_contrib = (amplitude_val / (width_val * np.sqrt(2 * np.pi))) * \
                                 np.exp(-((E_range - E_val) ** 2) / (2 * width_val ** 2))
                spectrum += gaussian_contrib
            
            max_intensity = max(max_intensity, np.max(spectrum))
            
            alpha = 0.7 if n_iterations > 1 else 1.0
            ax1.plot(E_range, spectrum, color=color, linewidth=2, alpha=alpha,
                    label=f'Iter {iter_data["iteration"]} ({iter_data["input_peaks"]} peaks)')
        
        ax1.set_xlabel('Energy (eV)')
        ax1.set_ylabel('Intensity')
        ax1.set_xlim([E_min, E_max])
        ax1.set_ylim([0, max_intensity * 1.1])
        ax1.set_title(f'{molName} - NEXAFS Evolution Overview')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        iterations = [data['iteration'] for data in iterations_data]
        peak_counts = [data['input_peaks'] for data in iterations_data]
        
        total_intensities = []
        for iter_data in iterations_data:
            peak_data = iter_data['peak_data']
            total_os = peak_data['OS'].sum()
            total_intensities.append(total_os)
        
        ax2_twin = ax2.twinx()
        
        bars = ax2.bar(iterations, peak_counts, alpha=0.6, color='skyblue', label='Peak Count')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Number of Peaks', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        line = ax2_twin.plot(iterations, total_intensities, 'ro-', linewidth=2, markersize=6, label='Total OS')
        ax2_twin.set_ylabel('Total Oscillator Strength', color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        
        ax2.set_title('Clustering Progress')
        ax2.grid(True, alpha=0.3)
        
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        combined_fig.tight_layout()
    
    return individual_fig, combined_fig

def plot_iteration_spectra_interactive(iteration_results, E_max, molName):
    """
    Creates an interactive Plotly plot of NEXAFS spectra for each iteration.
    """
    if not iteration_results['iterations']:
        return None
    
    iterations_data = iteration_results['iterations']
    
    all_energies = []
    for iter_data in iterations_data:
        all_energies.extend(iter_data['peak_data']['E'].values)
    
    E_min = min(all_energies) - 5
    E_range = np.linspace(E_min, E_max, 1000)
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1[:len(iterations_data)]
    
    for i, iter_data in enumerate(iterations_data):
        peak_data = iter_data['peak_data']
        
        spectrum = np.zeros_like(E_range)
        
        for _, peak in peak_data.iterrows():
            E_val = peak['E']
            width_val = peak['width']
            amplitude_val = peak['OS']
            
            gaussian_contrib = (amplitude_val / (width_val * np.sqrt(2 * np.pi))) * \
                             np.exp(-((E_range - E_val) ** 2) / (2 * width_val ** 2))
            spectrum += gaussian_contrib
        
        fig.add_trace(go.Scatter(
            x=E_range,
            y=spectrum,
            mode='lines',
            name=f'Iteration {iter_data["iteration"]} ({iter_data["input_peaks"]} peaks)',
            line=dict(width=3, color=colors[i % len(colors)]),
            hovertemplate='Energy: %{x:.2f} eV<br>Intensity: %{y:.4f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=peak_data['E'],
            y=peak_data['OS'],
            mode='markers',
            marker=dict(
                symbol='line-ns',
                size=8,
                color=colors[i % len(colors)],
                line=dict(width=2)
            ),
            name=f'Peaks Iter {iter_data["iteration"]}',
            showlegend=False,
            hovertemplate='Peak Energy: %{x:.2f} eV<br>OS: %{y:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'{molName} - Interactive NEXAFS Evolution',
        xaxis_title='Energy (eV)',
        yaxis_title='Intensity',
        hovermode='x unified',
        height=600,
        showlegend=True
    )
    
    return fig

def save_dataframe_to_csv_in_memory(dataframe, filename):
    """
    Saves a DataFrame to a CSV file in memory.
    """
    buffer = BytesIO()
    if dataframe is not None:
        dataframe.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer, filename

def zip_dataframes(dataframes_dict, zip_filename):
    """
    Zips multiple DataFrames into a single zip file in memory.
    """
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, "a") as zf:
        for name, df in dataframes_dict.items():
            if df is not None:
                csv_buffer, filename = save_dataframe_to_csv_in_memory(df, f"{name}.csv")
                zf.writestr(filename, csv_buffer.read())
    zip_buffer.seek(0)
    return zip_buffer

@st.cache_data(show_spinner=False)
def load_data(directory, width1, width2, maxEnergy):
    """
    Loads data from a directory, processes it, and returns the results as a dictionary.
    """
    if os.path.isdir(directory):
        st.write('Processing directory:', directory)
        progress_bar = st.progress(0)
        progress_text = st.empty()
        start_time = time.time()
        basis_sets, energy_results, orbital_alpha, orbital_beta, xray_transitions, atomic_coordinates = process_directory(
            directory, progress_bar, progress_text, width1, width2, 290, maxEnergy)
        end_time = time.time()
        st.write(f'Processing completed in {end_time - start_time:.2f} seconds.')
        return {
            'basis_sets': basis_sets,
            'energy_results': energy_results,
            'orbital_alpha': orbital_alpha,
            'orbital_beta': orbital_beta,
            'xray_transitions': xray_transitions,
            'atomic_coordinates': atomic_coordinates
        }
    else:
        st.write('Invalid directory. Please enter a valid directory path.')
        return None

def display_initial_data(data):
    """
    Enhanced display function with better layout and interactive filtering options.
    Updated to work with restructured energy data (one row per atom with prefixed columns).
    """
    st.header("Initial Data Analysis")
    
    # Create tabs for better organization
    main_tabs = st.tabs([
        "📊 Overview", 
        "⚛️ Molecular Structure", 
        "📈 Energy Analysis", 
        "🔬 Orbital Data",
        "🧮 Basis Sets"
    ])
    
    with main_tabs[0]:  # Overview Tab
        st.subheader("Dataset Overview")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            unique_atoms = data['energy_results']['Atom'].nunique()
            st.metric("Unique Atoms", unique_atoms)
        
        with col2:
            total_calculations = len(data['energy_results'])
            st.metric("Total Atoms", total_calculations)
                
        with col3:
            total_transitions = len(data['xray_transitions'])
            st.metric("X-ray Transitions", total_transitions)
        
        # Quick preview of energy results
        st.subheader("Restructured Energy Results Summary")
        
        # Add filtering options
        selected_atoms = st.multiselect(
            "Filter by Atoms:",
            options=sorted(data['energy_results']['Atom'].unique()),
            default=sorted(data['energy_results']['Atom'].unique())[:5],  # Show first 5 by default
            key="overview_atom_filter"
        )
        
        # Filter the dataframe
        filtered_energy_df = data['energy_results'][
            data['energy_results']['Atom'].isin(selected_atoms)
        ]
        
        st.dataframe(filtered_energy_df, width='stretch')
    
    with main_tabs[1]:  # Molecular Structure Tab
        st.subheader("Molecular Structure and Coordinates")
        
        # File selection dropdown
        available_files = data['atomic_coordinates']['Originating File'].unique()
        selected_file = st.selectbox(
            "Select Structure File:",
            options=available_files,
            key="structure_file_select"
        )
        
        # Filter coordinates based on selected file
        filtered_coords = data['atomic_coordinates'][
            data['atomic_coordinates']['Originating File'] == selected_file
        ]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("### Atomic Coordinates")
            st.dataframe(filtered_coords, width='stretch')
            
            # Download coordinates as XYZ
            if st.button("Generate XYZ File", key="generate_xyz"):
                dataframe_to_xyz(filtered_coords, "molecule.xyz")
                st.success("XYZ file generated successfully!")
        
        with col2:
            st.write("### 3D Molecular Visualization")
            try:
                view = visualize_xyz_with_stmol(filtered_coords, "molecule.xyz")
                showmol(view, height=500, width=500)
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
                st.info("Please ensure atomic coordinates are properly formatted.")
    
    with main_tabs[2]:  # Energy Analysis Tab
        st.subheader("Detailed Energy Analysis")
        
        # Energy analysis options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            energy_atom_filter = st.selectbox(
                "Select Atom for Analysis:",
                options=["All"] + sorted(data['energy_results']['Atom'].unique()),
                key="energy_atom_select"
            )
        
        with col2:
            # Get energy column options (columns with energy values)
            energy_columns = [col for col in data['energy_results'].columns 
                            if any(x in col.lower() for x in ['energy', 'lumo_en', 'correction'])]
            energy_column_filter = st.selectbox(
                "Select Energy Type:",
                options=["All"] + energy_columns,
                key="energy_column_select"
            )
        
        with col3:
            energy_units = st.selectbox(
                "Energy Units:",
                options=["Hartree", "eV"],
                key="energy_units"
            )
        
        # Filter energy data
        energy_df_filtered = data['energy_results'].copy()
        
        if energy_atom_filter != "All":
            energy_df_filtered = energy_df_filtered[energy_df_filtered['Atom'] == energy_atom_filter]
        
        # Show specific columns if selected
        if energy_column_filter != "All":
            columns_to_show = ['Atom', energy_column_filter]
            energy_df_filtered = energy_df_filtered[columns_to_show]
        
        # Convert units if needed
        if energy_units == "eV":
            hartree_columns = [col for col in energy_df_filtered.columns if "(H)" in col]
            for col in hartree_columns:
                new_col_name = col.replace("(H)", "(eV)")
                energy_df_filtered[new_col_name] = energy_df_filtered[col] * 27.2114  # Hartree to eV conversion
                energy_df_filtered = energy_df_filtered.drop(columns=[col])
        
        st.dataframe(energy_df_filtered, width='stretch')
        
        # Energy comparison plots
        st.subheader("Energy Analysis Plots")
        
        plot_tabs = st.tabs(["Energy Correction", "Total Energy Comparison", "LUMO Energy"])
        
        with plot_tabs[0]:  # Energy Correction Plot
            st.write("### Energy Correction Analysis")
            
            if 'Energy_Correction' in data['energy_results'].columns:
                correction_data = data['energy_results'][['Atom', 'Energy_Correction']].dropna()
                
                if not correction_data.empty:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    bars = ax.bar(range(len(correction_data)), 
                                 correction_data['Energy_Correction'],
                                 color='steelblue', alpha=0.7)
                    
                    ax.set_xlabel('Atom')
                    ax.set_ylabel('Energy Correction (Hartree)')
                    ax.set_title('Energy Correction by Atom (EXC - GND - LUMO_En)')
                    ax.set_xticks(range(len(correction_data)))
                    ax.set_xticklabels(correction_data['Atom'], rotation=45, ha='right')
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, correction_data['Energy_Correction']):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.6f}', ha='center', va='bottom', fontsize=8)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Correction", f"{correction_data['Energy_Correction'].mean():.6f}")
                    with col2:
                        st.metric("Std Deviation", f"{correction_data['Energy_Correction'].std():.6f}")
                    with col3:
                        st.metric("Range", f"{correction_data['Energy_Correction'].max() - correction_data['Energy_Correction'].min():.6f}")
                else:
                    st.warning("No energy correction data available. Make sure EXC, GND, and TP calculations are present.")
        
        with plot_tabs[1]:  # Total Energy Comparison
            st.write("### Total Energy Comparison")
            
            # Get total energy columns
            total_energy_cols = [col for col in data['energy_results'].columns if 'Total energy (H)' in col]
            
            if total_energy_cols:
                energy_comparison = data['energy_results'][['Atom'] + total_energy_cols].set_index('Atom')
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Create grouped bar chart
                x = np.arange(len(energy_comparison.index))
                width = 0.25
                
                for i, col in enumerate(total_energy_cols):
                    calc_type = col.split('_')[0]  # Get GND, EXC, TP prefix
                    ax.bar(x + i*width, energy_comparison[col], width, label=calc_type, alpha=0.7)
                
                ax.set_xlabel('Atom')
                ax.set_ylabel('Total Energy (Hartree)')
                ax.set_title('Total Energy Comparison by Calculation Type')
                ax.set_xticks(x + width)
                ax.set_xticklabels(energy_comparison.index, rotation=45, ha='right')
                ax.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with plot_tabs[2]:  # LUMO Energy
            st.write("### LUMO Energy Analysis")
            
            if 'TP_LUMO_En' in data['energy_results'].columns:
                lumo_data = data['energy_results'][['Atom', 'TP_LUMO_En']].dropna()
                
                if not lumo_data.empty:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    bars = ax.bar(range(len(lumo_data)), 
                                 lumo_data['TP_LUMO_En'],
                                 color='orange', alpha=0.7)
                    
                    ax.set_xlabel('Atom')
                    ax.set_ylabel('LUMO Energy (eV)')
                    ax.set_title('LUMO Energy by Atom')
                    ax.set_xticks(range(len(lumo_data)))
                    ax.set_xticklabels(lumo_data['Atom'], rotation=45, ha='right')
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, lumo_data['TP_LUMO_En']):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.2f}', ha='center', va='bottom', fontsize=8)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
    
    with main_tabs[3]:  # Orbital Data Tab
        st.subheader("Orbital Analysis")
        
        # Orbital data filtering
        col1, col2 = st.columns(2)
        
        with col1:
            orbital_atom_filter = st.selectbox(
                "Select Atom for Orbital Analysis:",
                options=["All"] + sorted(data['orbital_alpha']['Atom'].unique()),
                key="orbital_atom_select"
            )
        
        with col2:
            orbital_file_filter = st.selectbox(
                "Select File:",
                options=["All"] + sorted(data['orbital_alpha']['Originating File'].unique()),
                key="orbital_file_select"
            )
        
        # Filter orbital data
        alpha_filtered = data['orbital_alpha'].copy()
        beta_filtered = data['orbital_beta'].copy()
        
        if orbital_atom_filter != "All":
            alpha_filtered = alpha_filtered[alpha_filtered['Atom'] == orbital_atom_filter]
            beta_filtered = beta_filtered[beta_filtered['Atom'] == orbital_atom_filter]
        
        if orbital_file_filter != "All":
            alpha_filtered = alpha_filtered[alpha_filtered['Originating File'] == orbital_file_filter]
            beta_filtered = beta_filtered[beta_filtered['Originating File'] == orbital_file_filter]
        
        # Display orbital data in sub-tabs
        orbital_subtabs = st.tabs(["Alpha Orbitals", "Beta Orbitals", "Core Hole/HOMO/LUMO", "Orbital Visualization"])
        
        with orbital_subtabs[0]:
            st.write("### Alpha Orbital Data")
            st.dataframe(alpha_filtered, width='stretch')
        
        with orbital_subtabs[1]:
            st.write("### Beta Orbital Data")
            st.dataframe(beta_filtered, width='stretch')
        
        with orbital_subtabs[2]:
            st.write("### Core Hole, HOMO, and LUMO Analysis")
            core_hole_homo_lumo_df = find_core_hole_homo_lumo(alpha_filtered)
            st.dataframe(core_hole_homo_lumo_df, width='stretch')
            
            # Add summary statistics
            if not core_hole_homo_lumo_df.empty:
                st.write("#### Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    core_holes = core_hole_homo_lumo_df['Core Hole'].notna().sum()
                    st.metric("Core Holes Found", core_holes)
                with col2:
                    homos = core_hole_homo_lumo_df['HOMO'].notna().sum()
                    st.metric("HOMOs Identified", homos)
                with col3:
                    lumos = core_hole_homo_lumo_df['LUMO'].notna().sum()
                    st.metric("LUMOs Identified", lumos)
        
        with orbital_subtabs[3]:
            # Energy level diagram
            st.write("### Orbital Energy Visualization")
            
            if not alpha_filtered.empty:
                # Convert energy to numeric
                alpha_filtered['Energy(eV)'] = pd.to_numeric(alpha_filtered['Energy(eV)'], errors='coerce')
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Group by atom and file for separate energy level diagrams
                for i, (group_name, group_data) in enumerate(alpha_filtered.groupby(['Atom', 'Originating File'])):
                    atom, file = group_name
                    energies = group_data['Energy(eV)'].dropna().sort_values()
                    
                    # Plot energy levels as horizontal lines
                    x_pos = i * 0.5
                    for energy in energies:
                        ax.hlines(energy, x_pos - 0.1, x_pos + 0.1, colors='blue', linewidth=2)
                    
                    ax.text(x_pos, energies.min() - 5, f"{atom}\n{file}", 
                           ha='center', va='top', fontsize=8, rotation=0)
                
                ax.set_ylabel('Energy (eV)')
                ax.set_title('Orbital Energy Levels')
                ax.grid(True, alpha=0.3)
                
                # Remove x-axis ticks as they're not meaningful
                ax.set_xticks([])
                
                plt.tight_layout()
                st.pyplot(fig)
    
    with main_tabs[4]:  # Basis Sets Tab
        st.subheader("Basis Set Information")
        
        # Display basis sets with filtering
        col1, col2 = st.columns(2)
        
        with col1:
            basis_atom_filter = st.selectbox(
                "Filter by Atom:",
                options=["All"] + sorted(data['basis_sets']['Atom'].unique()),
                key="basis_atom_select"
            )
        
        with col2:
            basis_file_filter = st.selectbox(
                "Filter by File:",
                options=["All"] + sorted(data['basis_sets']['Originating File'].unique()),
                key="basis_file_select"
            )
        
        # Filter basis sets
        basis_filtered = data['basis_sets'].copy()
        
        if basis_atom_filter != "All":
            basis_filtered = basis_filtered[basis_filtered['Atom'] == basis_atom_filter]
        
        if basis_file_filter != "All":
            basis_filtered = basis_filtered[basis_filtered['Originating File'] == basis_file_filter]
        
        st.dataframe(basis_filtered, width='stretch')
        
        # Basis set summary
        if not basis_filtered.empty:
            st.subheader("Basis Set Summary")
            
            basis_summary_tabs = st.tabs(["Auxiliary Basis", "Orbital Basis", "Model Potential"])
            
            with basis_summary_tabs[0]:
                if 'Auxiliary Basis' in basis_filtered.columns:
                    aux_basis_counts = basis_filtered['Auxiliary Basis'].value_counts()
                    st.write("### Auxiliary Basis Distribution")
                    st.bar_chart(aux_basis_counts)
            
            with basis_summary_tabs[1]:
                if 'Orbital Basis' in basis_filtered.columns:
                    orb_basis_counts = basis_filtered['Orbital Basis'].value_counts()
                    st.write("### Orbital Basis Distribution")
                    st.bar_chart(orb_basis_counts)
            
            with basis_summary_tabs[2]:
                if 'Model Potential' in basis_filtered.columns:
                    model_pot_counts = basis_filtered['Model Potential'].value_counts()
                    st.write("### Model Potential Distribution")
                    st.bar_chart(model_pot_counts)
    
    # Memory usage indicator
    with st.expander("📊 Memory Usage"):
        show_memory_usage()

def display_filtered_data(data, maxEnergy, OST, molName):
    filtered_xray_transitions = filter_and_normalize_xray(data['xray_transitions'], maxEnergy, OST)

    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        st.write('### Energy and OS Filtered X-ray Transitions')
        st.dataframe(filtered_xray_transitions)
    with col2:
        st.write('### Energy and OS Filtered DFT NEXAFS')
        plot_total_spectra(filtered_xray_transitions, maxEnergy, molName, 'normalized_os', (OST / 100) * 2)
    with col3:
        st.write('### Energy and OS Filtered KDE')
        plot_density_spectra(filtered_xray_transitions, maxEnergy, 'normalized_os')
    st.write('### Excitation Centers DFT NEXAFS')
    plot_individual_spectra(data['xray_transitions'], maxEnergy)
    
    return filtered_xray_transitions

@st.fragment
def visualize_gaussian_overlap(filtered_df):
    st.subheader("Gaussian Peak Overlap Visualizer")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("### Filtered Peaks")
        st.dataframe(filtered_df)

    with col2:
        with st.container():
            st.markdown("### Select Peaks")

            c1, c2 = st.columns(2)
            max_index = len(filtered_df) - 1

            with c1:
                index1 = st.number_input("Peak 1 Index", min_value=0, max_value=max_index, value=0, key="peak1_idx")
            with c2:
                index2 = st.number_input("Peak 2 Index", min_value=0, max_value=max_index, value=1, key="peak2_idx")

            def extract_peak_params(df, idx):
                row = df.iloc[idx]
                return row['E'], row['width'], row['OS']

            mu1, sigma1, amp1 = extract_peak_params(filtered_df, index1)
            mu2, sigma2, amp2 = extract_peak_params(filtered_df, index2)

            def gaussian(x, mu, sigma, amplitude):
                return amplitude * np.exp(-((x - mu)**2) / (2 * sigma**2))

            def overlap_area(mu1, sigma1, amp1, mu2, sigma2, amp2):
                integrand = lambda x: np.minimum(
                    gaussian(x, mu1, sigma1, amp1),
                    gaussian(x, mu2, sigma2, amp2)
                )
                lower = min(mu1 - 3*sigma1, mu2 - 3*sigma2)
                upper = max(mu1 + 3*sigma1, mu2 + 3*sigma2)
                area, _ = quad(integrand, lower, upper)
                return area

            max_sigma = max(sigma1, sigma2)
            x_lower = min(mu1, mu2) - 3 * max_sigma
            x_upper = max(mu1, mu2) + 3 * max_sigma
            x = np.linspace(x_lower, x_upper, 1000)

            g1 = gaussian(x, mu1, sigma1, amp1)
            g2 = gaussian(x, mu2, sigma2, amp2)
            overlap = np.minimum(g1, g2)
            ovlp_area = overlap_area(mu1, sigma1, amp1, mu2, sigma2, amp2)

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(x, g1, label=f'Peak {index1}', color='blue')
            ax.plot(x, g2, label=f'Peak {index2}', color='green')
            ax.fill_between(x, overlap, color='purple', alpha=0.4, label='Overlap')
            ax.set_title(f'Overlap Area = {ovlp_area:.4f}')
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel('Oscillator Strength')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

def main():
    """
    Main function to run the Streamlit application for StoBe Loader and Clustering Algorithm.
    """
    st.set_page_config(layout="wide")
    st.title('StoBe Loader for Clustering Algorithm')

    col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 1, 1, 1, 1, 1, 1])
    with col1:
        directory = st.text_input('Enter the directory containing the output folders:', key="dir_input_tab0")
    with col2:
        width1 = st.number_input('Width 1 (eV)', min_value=0.01, max_value=20.0, value=0.5) / 2.355
    with col3:
        width2 = st.number_input('Width 2 (eV)', min_value=0.01, max_value=20.0, value=12.0) / 2.355
    with col4:
        maxEnergy = st.number_input('Maximum DFT Energy (eV)', min_value=0.0, max_value=1000.0, value=320.0)
    with col5:
        molName = st.text_input('Enter the name of your molecule:')
    with col6:
        OST = st.number_input('OS Threshold (%)', min_value=0.0, max_value=100.0, value=10.0)
    with col7:
        OVPT = st.number_input('OVP Threshold (%)', min_value=0.0, max_value=100.0, value=50.0)

    tabs = st.tabs([
        "Initial Data Loading",
        "Initial Data Displays", 
        "Filtered Data Displays",
        "Clustering",
        "Peak Overlap Visualizer"
    ])

    with tabs[0]:  # Initial Data Loading
        st.header("Initial Data Loading")
        if st.button('Process Directory', key="process_button_tab0"):
            data = load_data(directory, width1, width2, maxEnergy)
            if data:
                st.session_state['processed_data'] = data

    with tabs[1]:  # Initial Data Displays
        st.header("Initial Data Displays")
        if 'processed_data' in st.session_state:
            display_initial_data(st.session_state['processed_data'])

    with tabs[2]:  # Filtered Data Displays
        st.header("Filtered Data Displays")
        if 'processed_data' in st.session_state:
            filter_key = f"{maxEnergy}_{OST}_{molName}"
            if 'filter_key' not in st.session_state or st.session_state['filter_key'] != filter_key:
                filtered_data = display_filtered_data(st.session_state['processed_data'], maxEnergy, OST, molName)
                st.session_state['filtered_data'] = filtered_data
                st.session_state['filter_key'] = filter_key
            else:
                st.write('### Energy and OS Filtered X-ray Transitions')
                st.dataframe(st.session_state['filtered_data'])

    with tabs[3]:  # Clustering
        st.header("Clustering Analysis")
        
        if 'filtered_data' in st.session_state and not st.session_state['filtered_data'].empty:
            filtered_df = st.session_state['filtered_data']
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                clustering_mode = st.selectbox(
                    "Choose clustering mode:",
                    ["Single Iteration", "Iterative (until convergence)"],
                    key="clustering_mode_select"
                )
                
                clustering_method = st.selectbox(
                    "Choose clustering method:",
                    ["Hierarchical (Iterative)", "Simple Sequential (IGOR-style)"],
                    key="clustering_method_select"
                )
            
            with col2:
                if clustering_method == "Simple Sequential (IGOR-style)":
                    st.info("Matrix adjacency clustering - only considers directly adjacent peaks")
                else:
                    linkage_method = st.selectbox(
                        "Linkage method:",
                        ["average", "single", "complete", "ward"],
                        key="linkage_method_select"
                    )
            
            with col3:
                if clustering_mode == "Iterative (until convergence)":
                    max_iterations = st.number_input("Max iterations", min_value=1, max_value=20, value=5, 
                                                help="Maximum number of iterations to prevent infinite loops")
            
            with col4:
                run_clustering = st.button("Run Clustering Analysis", key="run_clustering_btn")
            
            if clustering_mode == "Single Iteration":
                overlap_key = f"overlap_{len(filtered_df)}_{OVPT}"
                
                if run_clustering or overlap_key not in st.session_state:
                    with st.spinner("Calculating overlap matrix..."):
                        overlap_matrix = calculate_percent_overlap_matrix_optimized(
                            filtered_df, 
                            energy_threshold=10.0
                        )
                        st.session_state[overlap_key] = overlap_matrix
                        st.session_state['overlap_matrix'] = overlap_matrix
                else:
                    overlap_matrix = st.session_state[overlap_key]
                
                if 'overlap_matrix' in st.session_state and run_clustering:
                    if clustering_method == "Simple Sequential (IGOR-style)":
                        clusters = sequential_clustering_optimized(filtered_df, overlap_matrix, OVPT)
                        st.session_state['clustering_method'] = "Sequential (IGOR-style Matrix Adjacency)"
                    else:
                        clusters = hierarchical_clustering(
                            overlap_matrix, OVPT, method=linkage_method
                        )
                        st.session_state['clustering_method'] = f"Hierarchical ({linkage_method})"
                    
                    st.session_state['clusters'] = clusters
                    st.session_state['cluster_summary'] = create_cluster_summary(filtered_df, clusters)
                    st.session_state['iteration_results'] = None
            
            else:  # Iterative clustering
                if run_clustering:
                    with st.spinner("Running iterative clustering..."):
                        method = 'sequential' if clustering_method == "Simple Sequential (IGOR-style)" else 'hierarchical'
                        linkage_method_to_use = linkage_method if clustering_method != "Simple Sequential (IGOR-style)" else 'average'
                        
                        iteration_results = iterative_clustering(
                            filtered_df, OVPT, method, max_iterations, linkage_method_to_use
                        )
                        
                        st.session_state['iteration_results'] = iteration_results
                        st.session_state['clustering_method'] = f"Iterative {clustering_method}"
                        
                        if iteration_results['iterations']:
                            final_iteration = iteration_results['iterations'][-1]
                            st.session_state['clusters'] = final_iteration['clusters']
                            st.session_state['cluster_summary'] = create_cluster_summary(
                                final_iteration['peak_data'], final_iteration['clusters']
                            )
                            st.session_state['overlap_matrix'] = final_iteration['overlap_matrix']
            
            # Display results
            if 'iteration_results' in st.session_state and st.session_state['iteration_results'] is not None:
                iteration_results = st.session_state['iteration_results']
                
                st.subheader(f"Iterative Clustering Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Iterations", iteration_results['total_iterations'])
                with col2:
                    initial_peaks = len(filtered_df)
                    final_peaks = len(iteration_results['final_peaks']) if 'final_peaks' in iteration_results else 0
                    st.metric("Peak Reduction", f"{initial_peaks} → {final_peaks}")
                with col3:
                    reduction_pct = ((initial_peaks - final_peaks) / initial_peaks * 100) if initial_peaks > 0 else 0
                    st.metric("Reduction %", f"{reduction_pct:.1f}%")
                with col4:
                    st.metric("Convergence", iteration_results['convergence_reason'])
                
                iter_tabs = st.tabs([
                    "Convergence Plot",
                    "NEXAFS Spectra Evolution",
                    "Iteration Heatmaps", 
                    "Final Results"
                ])
                
                with iter_tabs[0]:
                    st.write("### Clustering Convergence Analysis")
                    fig_convergence = plot_iteration_convergence(iteration_results)
                    if fig_convergence:
                        st.pyplot(fig_convergence)
                
                with iter_tabs[1]:
                    st.write("### NEXAFS Spectra Evolution Across Iterations")
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        show_individual = st.checkbox("Show Individual Iterations", value=True)
                    with col2:
                        show_combined = st.checkbox("Show Combined Overview", value=True)
                    with col3:
                        show_interactive = st.checkbox("Show Interactive Plot", value=False)
                    
                    if show_individual or show_combined:
                        fig_individual, fig_combined = plot_iteration_nexafs_spectra(
                            iteration_results, maxEnergy, molName, show_individual, show_combined
                        )
                        
                        if show_individual and fig_individual:
                            st.write("#### Individual Iteration Spectra")
                            st.pyplot(fig_individual)
                        
                        if show_combined and fig_combined:
                            st.write("#### Combined Overview")
                            st.pyplot(fig_combined)
                    
                    if show_interactive:
                        st.write("#### Interactive Spectra Evolution")
                        fig_interactive = plot_iteration_spectra_interactive(iteration_results, maxEnergy, molName)
                        if fig_interactive:
                            st.plotly_chart(fig_interactive, width='stretch')
                
                with iter_tabs[2]:
                    st.write("### Overlap Matrix Evolution")
                    max_display = st.slider("Max iterations to display", 1, min(10, len(iteration_results['iterations'])), 5)
                    fig_heatmaps = plot_iteration_heatmaps(iteration_results, max_display)
                    if fig_heatmaps:
                        st.pyplot(fig_heatmaps)
                
                with iter_tabs[3]:
                    st.write("### Final Clustering State")
                    if 'final_peaks' in iteration_results:
                        st.write("**Final Merged Peaks:**")
                        st.dataframe(iteration_results['final_peaks'])
            
            elif 'clusters' in st.session_state:
                clusters = st.session_state['clusters']
                cluster_summary = st.session_state['cluster_summary']
                
                st.subheader(f"Clustering Results ({st.session_state.get('clustering_method', 'Unknown Method')})")
                st.write(f"**Found {len(clusters)} clusters from {len(filtered_df)} peaks**")
                
                cluster_tabs = st.tabs([
                    "Cluster Summary", 
                    "Overlap Heatmap", 
                    "Cluster Spectra", 
                    "Interactive Summary"
                ])
                
                with cluster_tabs[0]:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("### Cluster Summary Table")
                        st.dataframe(cluster_summary, width='stretch')
                    
                    with col2:
                        st.write("### Statistics")
                        st.metric("Total Clusters", len(clusters))
                        st.metric("Largest Cluster", cluster_summary['Num_Peaks'].max() if not cluster_summary.empty else 0)
                        st.metric("Avg Peaks/Cluster", f"{cluster_summary['Num_Peaks'].mean():.1f}" if not cluster_summary.empty else "0")
                
                with cluster_tabs[1]:
                    st.write("### Peak Overlap Matrix")
                    if 'overlap_matrix' in st.session_state:
                        fig_heatmap = plot_overlap_heatmap(st.session_state['overlap_matrix'], clusters)
                        st.pyplot(fig_heatmap)
                        
                        csv_buffer = st.session_state['overlap_matrix'].to_csv()
                        st.download_button(
                            label="Download Overlap Matrix CSV",
                            data=csv_buffer,
                            file_name=f"{molName}_overlap_matrix.csv",
                            mime="text/csv"
                        )
                
                with cluster_tabs[2]:
                    st.write("### Individual Cluster Spectra")
                    if clusters:
                        fig_spectra = plot_clustered_spectra(filtered_df, clusters, maxEnergy)
                        if fig_spectra:
                            st.pyplot(fig_spectra)
                        else:
                            st.write("No clusters to display.")
                
                with cluster_tabs[3]:
                    st.write("### Interactive Cluster Analysis")
                    if not cluster_summary.empty:
                        fig_interactive = plot_cluster_summary_interactive(cluster_summary)
                        if fig_interactive:
                            st.plotly_chart(fig_interactive, width='stretch')
            
            else:
                st.info("Configure clustering parameters and click 'Run Clustering Analysis' to begin.")
        
        else:
            st.warning("Please load and filter data in the earlier tabs before performing clustering analysis.")

    with tabs[4]:  # Peak Overlap Visualizer
        st.header("Peak Overlap Visualizer")
        if 'filtered_data' in st.session_state and not st.session_state['filtered_data'].empty:
            visualize_gaussian_overlap(st.session_state['filtered_data'])
        else:
            st.warning("Please load and filter data in the earlier tabs before using this visualizer.")

    # Download section
    if 'processed_data' in st.session_state:
        data_to_save = {
            f"{molName}_basis_sets": st.session_state['processed_data'].get('basis_sets'),
            f"{molName}_energy_results": st.session_state['processed_data'].get('energy_results'),
            f"{molName}_orbital_alpha": st.session_state['processed_data'].get('orbital_alpha'),
            f"{molName}_orbital_beta": st.session_state['processed_data'].get('orbital_beta'),
            f"{molName}_xray_transitions": st.session_state['processed_data'].get('xray_transitions'),
            f"{molName}_atomic_coordinates": st.session_state['processed_data'].get('atomic_coordinates'),
            f"{molName}_core_hole_homo_lumo": find_core_hole_homo_lumo(st.session_state['processed_data'].get('orbital_alpha'))
        }
        
        if 'overlap_matrix' in st.session_state:
            data_to_save[f"{molName}_overlap_matrix"] = pd.DataFrame(st.session_state['overlap_matrix'])
        
        zip_filename = f"{molName}_dataframes.zip"
        zip_buffer = zip_dataframes(data_to_save, zip_filename)
        st.download_button(
            label="Download ZIP",
            data=zip_buffer,
            file_name=zip_filename,
            mime='application/zip'
        )

if __name__ == "__main__":
    main()