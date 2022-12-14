"""
Created: 2022/08/31

Useful shortcut functions for nextnano3 postprocessing.

@author: takuma.sato@nextnano.com
"""

# Python includes
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging

# nextnanopy includes
import nextnanopy as nn
import common


software = 'nextnano3'
# model_names = ['Gamma', 'L', 'X', 'Delta', 'HH', 'LH', 'SO', 'kp6', 'kp8']  # in nextnano++
model_names = ['cb1', 'vb1', 'vb2', 'vb3', 'kp6', 'kp8']  # in nextnano3
model_names_conduction = ['cb1']
model_names_valence    = ['vb1', 'vb2', 'vb3', 'kp6', 'kp8']



def detect_quantum_model(filename):
    """
    Detect the quantum model from a file name.

    INPUT:
        filename    name of output data file

    RETURN:
        quantum model string
    """
    found = False

    for model in model_names:
        if model + '_' in filename[0:4]:
            found = True
            quantum_model = model
            break

    if not found:
        raise RuntimeError('Quantum model cannot be detected!')

    return quantum_model


def getKPointsData1D_in_folder(folder_path):

    datafiles = common.getDataFiles_in_folder('k_points', folder_path, software)

    # inplaneK_dict = {
    #     'Gamma': list(),
    #     'kp6': list(),
    #     'kp8': list()
    # }
    inplaneK_dict = {model_name: list() for model_name in model_names}
    # TODO: adjustments needed
    for df in datafiles:
        filename = os.path.split(df.fullpath)[1]
        quantum_model = detect_quantum_model(filename)
        data = np.loadtxt(df.fullpath, skiprows=1)

        if np.ndim(data) == 1:    # if only the zone-center is calculated
            inplaneK = [data[2], data[3]]
            inplaneK_dict[quantum_model] = inplaneK
            continue

        num_kPoints = len(data)
        for kIndex in range(num_kPoints):
            inplaneK = [data[kIndex, 2], data[kIndex, 3]]   # list representing 2D vector
            inplaneK_dict[quantum_model].append(inplaneK)

    # remove quantum model keys that are not simulated
    inplaneK_dict = {key: lis for key, lis in inplaneK_dict.items() if len(lis) >= 1}

    return inplaneK_dict


def plot_inplaneK(inplaneK_dict):
    """
    INPUT:
        inplaneK_dict   dictionary of in-plane k points generated by getKPointsData1D()

    RETURN:
        fig             matplotlib.figure.Figure object
    TODO: this method can be moved to common. All software-dependency is filtered by getKPointsData1D_in_folder().
    """
    if len(list(inplaneK_dict.values())[0]) == 2: return   # if only zone-center has been calculated

    for model, k_vectors in inplaneK_dict.items():
        if np.ndim(k_vectors) == 1: continue    # skip if only the zone-center is calculated

        fig, ax = plt.subplots()
        ax.set_xlabel('$k_y$ [$\mathrm{nm}^{-1}$]')
        ax.set_ylabel('$k_z$ [$\mathrm{nm}^{-1}$]')
        ax.set_title(f'in-plane k points (quantum model: {model})')
        for k_vector in k_vectors:
            ax.scatter(k_vector[0], k_vector[1], color='green')
        ax.axhline(color='grey', linewidth=1.0)
        ax.axvline(color='grey', linewidth=1.0)

        for kIndex, k_vector in enumerate(k_vectors):
            ax.annotate(kIndex, (k_vector[0], k_vector[1]), xytext=(k_vector[0]+0.01, k_vector[1]))
        plt.show()

    return fig


def getDataFile_probabilities_in_folder(folder_path):
    """
    Get single nextnanopy.DataFile of probability_shift data in the specified folder.

    Parameters
    ----------
    folder_path : str
        output folder path in which the datafile should be sought

    Returns
    -------
    dictionary { quantum model key: corresponding list of nn.DataFile() objects for probability_shift }
    """
    datafiles = common.getDataFiles_in_folder('_psi_squared', folder_path, software)  # TODO: is this finding the correct files?

    # probability_dict = {
    #     'Gamma': list(),
    #     'kp6': list(),
    #     'kp8': list()
    # }
    probability_dict = {model_name: list() for model_name in model_names}
    for df in datafiles:
        filename = os.path.split(df.fullpath)[1]
        quantum_model = detect_quantum_model(filename)
        probability_dict[quantum_model].append(df)

    # delete quantum model keys whose probabilities do not exist in output folder
    models_to_be_removed = [model for model, df in probability_dict.items() if len(df) == 0]

    for model in models_to_be_removed:
        probability_dict.pop(model)

    return probability_dict


def getDataFile_amplitudesK0_in_folder(folder_path):
    """
    Get single nextnanopy.DataFile of zone-center amplitude data in the folder of specified name.
    Shifted data is avoided since non-shifted one is used to calculate overlap and matrix elements.

    INPUT:
        folder_path      output folder path in which the datafile should be sought

    RETURN:
        dictionary { quantum model key: list of nn.DataFile() objects for amplitude data }
    """
    datafiles = common.getDataFiles_in_folder('psi', folder_path, software, exclude_keywords='shift')   # return a list of nn.DataFile

    # amplitude_dict = {
    #     'Gamma': list(),
    #     'kp6': list(),
    #     'kp8': list()
    # }
    amplitude_dict = {model_name: list() for model_name in model_names}
    for df in datafiles:
        filename = os.path.split(df.fullpath)[1]
        quantum_model = detect_quantum_model(filename)

        if quantum_model == 'kp8' or quantum_model == 'kp6':
            if '_001_' not in filename: continue   # exclude non k|| = 0 amplitudes   # TODO: check if it's working
        amplitude_dict[quantum_model].append(df)

    # delete quantum model keys whose probabilities do not exist in output folder
    amplitude_dict_trimmed = {model: amplitude_dict[model] for model in amplitude_dict if len(amplitude_dict[model]) > 0}

    if len(amplitude_dict_trimmed) == 0:
        raise common.NextnanoInputFileError("Amplitudes are not output! Modify the input file.")

    return amplitude_dict_trimmed


def get_num_evs(probability_dict):
    """ number of eigenvalues for each quantum model """
    num_evs = dict()
    for model, datafiles in probability_dict.items():
        if len(datafiles) == 0:   # if no k-points are calculated
            num_evs[model] = 0
        else:
            df = datafiles[0]
            num_evs[model] = sum(1 for var in df.variables if ('Psi^2' in var.name))   # this conditional counting is necessary because probability_shift file may contain also eigenvalues.
            logging.debug(f"\nNumber of eigenvalues for {model}: {num_evs[model]}")
    return num_evs


############### plot probabilities ##################################
def get_states_to_be_plotted(datafiles_probability_dict, states_range_dict=None, states_list_dict=None):
    """
    Create dictionaries of
        1) eigenstate indices to be plotted for each quantum model
        2) number of eigenstates for each quantum model

    INPUT:
        datafiles_probability_dict      dict generated by getDataFile_probabilities() method
        states_range_dict               range of state indices to be plotted for each quantum model. dict of the form { 'quantum model': [start index, last index] }
        states_list_dict                list of state indices to be plotted for each quantum model. Alternatively, strings 'lowestElectron' and 'highestHole' are accepted and state indices are set automatically.

    RETURN:
        states_toBePlotted              for each quantum model, array of eigenstate indices to be plotted in the figure. Has the form:
                                        { 'quantum model': list of values }
        num_evs                         for each quantum model, number of eigenstates

    NOTE:
        state index is base 0 (differ from nextnano++ output), state No is base 1 (identical to nextnano++ output)
    TODO:
        this method is probably identical to nnp shortcut. can be merged
    """
    # validate input
    if states_range_dict is not None and not isinstance(states_range_dict, dict):
        raise TypeError("Argument 'states_range_dict' must be a dict")
    if states_list_dict is not None and not isinstance(states_list_dict, dict):
        raise TypeError("Argument 'states_list_dict' must be a dict")
    if (states_range_dict is not None) and (states_list_dict is not None):
        raise ValueError("Only one of 'states_range_dict' or 'states_list_dict' is allowed as an argument")

    # get number of eigenvalues
    num_evs = get_num_evs(datafiles_probability_dict)

    # TODO: nn3 has two output files '_el' and '_hl' also in 8kp calculation
    states_toBePlotted = dict.fromkeys(datafiles_probability_dict.keys())

    # determine index of states to be plotted
    if states_list_dict is None:
        if states_range_dict is None:
            for model in datafiles_probability_dict:
                states_toBePlotted[model] = np.arange(0, num_evs[model])   # by default, plot all the eigenstates
        else:
            # from states_range_dict
            for model in datafiles_probability_dict:
                if model not in states_range_dict:
                    states_toBePlotted[model] = np.arange(0, num_evs[model])   # by default, plot all the eigenstates
                else:
                    startIdx = states_range_dict[model][0] - 1
                    stopIdx  = states_range_dict[model][1] - 1
                    states_toBePlotted[model] = np.arange(startIdx, stopIdx+1, 1)   # np.arange(min, max) stops one step before the max
    else:
        # from states_list_dict
        first_element = list(datafiles_probability_dict.values())[0][0]
        filepath = first_element.fullpath   # take arbitrary quantum model because all of them are in the same folder bias_*/Quantum
        outfolder = os.path.split(filepath)[0]
        for model in datafiles_probability_dict:
            if model not in states_list_dict:
                states_toBePlotted[model] = np.arange(0, num_evs[model])   # by default, plot all the eigenstates
            else:
                states_toBePlotted[model] = list()
                for stateNo in states_list_dict[model]:
                    if stateNo == 'highestHole':
                        if model != 'kp8' and model not in model_names_valence:
                            raise ValueError(f"Quantum model '{model}' does not contain hole states.")
                        
                        # TODO: nn3 has two output files '_el' and '_hl' also in 8kp calculation
                        states_toBePlotted[model].append(find_highest_hole_state_atK0(outfolder, threshold=0.5))
                            
                    elif stateNo == 'lowestElectron':
                        if model != 'kp8' and model not in model_names_conduction:
                            raise ValueError(f"Quantum model '{model}' does not contain electron states.")

                        states_toBePlotted[model].append(find_lowest_electron_state_atK0(outfolder, threshold=0.5))
                        
                    elif stateNo == 'occupied':
                        if 'cutoff_occupation' not in states_list_dict.keys():
                            raise ValueError("cutoff_occupation must be specified in 'states_list_dict'")

                        # WARNING: state selection based on k||=0 occupation
                        df = common.getDataFile_in_folder(['occupation', model], outfolder, software)
                        try:
                            cutoff_occupation = np.double(states_list_dict['cutoff_occupation'])
                        except ValueError as e:
                            raise Exception("cutoff_occupation must be a real number!") from e
                        if cutoff_occupation < 0: 
                            raise ValueError("cutoff_occupation must be positive!")

                        states_toBePlotted[model] += [int(stateNo) - 1 for stateNo, occupation in zip(df.coords['no.'].value, df.variables['Occupation'].value) if occupation >= cutoff_occupation]
                    elif isinstance(stateNo, int):
                        if stateNo > num_evs[model]: 
                            raise ValueError("State index greater than number of eigenvalues calculated!")
                        states_toBePlotted[model].append(stateNo - 1)
    return states_toBePlotted, num_evs


############### find ground states from kp8 result ############################
def find_lowest_electron_state_atK0(output_folder, threshold=0.5):
    """
    TODO: Since nn3 outputs e- and h-like eigenstates in separate output files, 
    1. this method should return 0
    2. methods that use this method should distinguish two output files kp8_psi_squared_el and kp8_psi_squared_hl. 

    From spinor composition data, determine the lowest electron state in an 8-band k.p simulation.
    This method should be able to detect it properly even when the effective bandgap is negative,
    i.e. when the lowest electron state is below the highest hole state.

    Note
    ----
    Nonzero k points may become important for TI phase and camel-back dispersion.

    Parameters
    ----------
    output_folder : str
        output folder path
    threshold : real, optional
        If electron fraction in the spinor composition is greater than this value, the state is assumed to be an electron state.
        The default is 0.5.

    Returns
    -------
    state index (base 0) of the lowest electron state at in-plane k = 0

    """
    # get nn.DataFile object   # TODO: where is the output of spinor composition in nn3?
    try:
        datafile = common.getDataFile_in_folder(['eigenvalues', '_info'], output_folder, software)   # spinor composition at in-plane k = 0
    except FileNotFoundError:
        warnings.warn("Spinor components output in CbHhLhSo basis is not found. Assuming decoupling of the conduction and valence bands...", category=common.NextnanoInputFileWarning)
        return int(0)

    # check if it is an 8-band k.p simulation result
    filename = os.path.split(datafile.fullpath)[1]
    if detect_quantum_model(filename) != 'kp8':
        raise common.NextnanopyScriptError("This method only applies to 8-band k.p model!")

    # find the lowest electron state
    num_evs = len(datafile.variables['cb1'].value)
    for stateIndex in range(num_evs):
        electronFraction = datafile.variables['cb1'].value[stateIndex] + datafile.variables['cb2'].value[stateIndex]
        if electronFraction > threshold:
            return stateIndex

    raise RuntimeError(f"No electron states found in: {output_folder}")


def find_highest_hole_state_atK0(output_folder, threshold=0.5):
    """
    TODO: Since nn3 outputs e- and h-like eigenstates in separate output files, 
    1. this method should return 0
    2. methods that use this method should distinguish two output files kp8_psi_squared_el and kp8_psi_squared_hl. 

    From spinor composition data, determine the highest hole state in an 8-band k.p simulation.
    This method should be able to detect it properly even when the effective bandgap is negative,
    i.e. when the highest hole state is above the lowest electron state.

    Note
    ----
    Nonzero k points may become important for TI phase and camel-back dispersion.

    Parameters
    ----------
    output_folder : str
        output folder path
    threshold : real, optional
        If electron fraction in the spinor composition is less than this value, the state is assumed to be a hole state.
        The default is 0.5.

    Returns
    -------
    state index (base 0) of the highest hole state at in-plane k = 0

    """
    # get nn.DataFile object  # TODO: where is the output of spinor composition in nn3?
    try:
        datafile = common.getDataFile_in_folder(['eigenvalues', '_info'], output_folder, software)   # spinor composition at in-plane k = 0
    except FileNotFoundError:
        warnings.warn("Spinor components output in CbHhLhSo basis is not found. Assuming decoupling of the conduction and valence bands...", category=common.NextnanoInputFileWarning)
        return int(0)

    # check if it is an 8-band k.p simulation result
    filename = os.path.split(datafile.fullpath)[1]
    if detect_quantum_model(filename) != 'kp8':
        raise common.NextnanopyScriptError("This method only applies to 8-band k.p model!")

    # find the highest hole state
    num_evs = len(datafile.variables['cb1'].value)
    for stateIndex in reversed(range(num_evs)):
        electronFraction = datafile.variables['cb1'].value[stateIndex] + datafile.variables['cb2'].value[stateIndex]
        if electronFraction < threshold:
            return stateIndex

    raise RuntimeError(f"No hole states found in: {output_folder}")


################ optics analysis #############################################
kp8_basis = ['cb1', 'cb2', 'hh1', 'hh2', 'lh1', 'lh2', 'so1', 'so2']
kp6_basis = ['hh1', 'hh2', 'lh1', 'lh2', 'so1', 'so2']

def calculate_overlap(output_folder, force_lightHole=False):
    """
    Calculate envelope overlap between the lowest electron and highest hole states from wavefunction output data

    Parameters
    ----------
    output_folder : str
        Absolute path of output folder that contains the amplitude data
    force_lightHole : bool, optional
        If True, always use the overlap of Gamma and light-hole states. Effective only for single-band models. 
        Default is False

    Returns
    -------
    overlap : np.ndarray with dtype=cdouble
        Spatial overlap of envelope functions

    Requires
    --------
    SciPy package installation

    """
    from scipy.integrate import simps

    # load amplitude data
    # TODO: nn3 has two output files '_el' and '_hl' also in 8kp calculation
    datafile_amplitude_at_k0 = getDataFile_amplitudesK0_in_folder(output_folder)   # returns a dict of nn.DataFile

    if 'kp8' in datafile_amplitude_at_k0.keys():
        electron_state_is_multiband = True
        hole_state_is_multiband = True
        df_e = datafile_amplitude_at_k0['kp8'][0]
        df_h = datafile_amplitude_at_k0['kp8'][0]
        h_state_basis = kp8_basis
    elif 'kp6' in datafile_amplitude_at_k0.keys():
        electron_state_is_multiband = False
        hole_state_is_multiband = True
        df_e = datafile_amplitude_at_k0['Gamma'][0]
        df_h = datafile_amplitude_at_k0['kp6'][0]
        h_state_basis = kp6_basis
    else:  # single band
        electron_state_is_multiband = False
        hole_state_is_multiband = False
        df_e = datafile_amplitude_at_k0['Gamma'][0]
        df_HH = datafile_amplitude_at_k0['HH'][0]
        df_LH = datafile_amplitude_at_k0['LH'][0]
        
        if force_lightHole:
            # always take Gamma-LH
            df_h = df_LH
            h_state_basis = ['LH']
        else:
            # take the highest of HH and LH eigenstates
            E_HH = common.getDataFile_in_folder(['energy_spectrum', '_HH'], output_folder, software).variables['Energy'].value[0]   # energy of the first eigenstate
            E_LH = common.getDataFile_in_folder(['energy_spectrum', '_LH'], output_folder, software).variables['Energy'].value[0]   # energy of the first eigenstate
            if E_HH >= E_LH:
                df_h = df_HH
                h_state_basis = ['HH']
            else:
                df_h = df_LH
                h_state_basis = ['LH']
        
    x = df_e.coords['x'].value
    iLowestElectron = find_lowest_electron_state_atK0(output_folder, threshold=0.5)
    iHighestHole    = find_highest_hole_state_atK0(output_folder, threshold=0.5)
    

    # extract amplitude of electron-like state
    if electron_state_is_multiband:
        amplitude_e = np.zeros((len(kp8_basis), len(x)), dtype=np.cdouble)
        for mu, band in enumerate(kp8_basis):
            amplitude_e[mu,:].real = df_e.variables[f'Psi_{iLowestElectron+1}_{band}_real'].value
            amplitude_e[mu,:].imag = df_e.variables[f'Psi_{iLowestElectron+1}_{band}_imag'].value
    else:
        amplitude_e = np.zeros(len(x), dtype=np.double)
        amplitude_e[:] = df_e.variables[f'Psi_{iLowestElectron+1}'].value
    
    # extract amplitude of hole-like state
    if hole_state_is_multiband:
        amplitude_h = np.zeros((len(h_state_basis), len(x)), dtype=np.cdouble)
        for nu, band in enumerate(h_state_basis):
            amplitude_h[nu,:].real = df_h.variables[f'Psi_{iHighestHole+1}_{band}_real'].value
            amplitude_h[nu,:].imag = df_h.variables[f'Psi_{iHighestHole+1}_{band}_imag'].value
    else:
        amplitude_h = np.zeros(len(x), dtype=np.double)
        amplitude_h[:] = df_h.variables[f'Psi_{iHighestHole+1}'].value

    overlap = 0.

    # calculate overlap
    if electron_state_is_multiband and hole_state_is_multiband:
        for mu in range(len(kp8_basis)):
            for nu in range(len(kp8_basis)):
                prod = np.multiply(np.conjugate(amplitude_h[nu,:]), amplitude_e[mu,:])   # multiply arguments element-wise
                overlap += simps(prod, x)
    elif hole_state_is_multiband:
        for nu in range(len(kp6_basis)):
            prod = np.multiply(np.conjugate(amplitude_h[nu,:]), amplitude_e[:])   # multiply arguments element-wise
            overlap += simps(prod, x)
    else:
        prod = np.multiply(np.conjugate(amplitude_h[:]), amplitude_e[:])   # multiply arguments element-wise
        overlap += simps(prod, x)

    return overlap


def get_transition_energy(output_folder, force_lightHole=False):
    """ 
    Get the transition energy = energy separation between the lowest electron and highest hole states.
    Unit: eV
    """
    # TODO: make it compatible with single-band & kp6 models. See nnp implementation
    # NOTE: nn3 has two output files '_el' and '_hl' also in 8kp calculation.
    datafile_el = common.getDataFile_in_folder(["eigenvalues", "el"], output_folder, software, exclude_keywords=["info", "pos"])
    datafile_hl = common.getDataFile_in_folder(["eigenvalues", "hl"], output_folder, software, exclude_keywords=["info", "pos"])
    if 'kp8' not in datafile_el.fullpath or 'kp8' not in datafile_hl.fullpath:
        raise NotImplementedError("This method is currently limited to kp8!")

    iLowestElectron = 0
    iHighestHole    = 0

    # get the energy separation dE = (electron state level) - (hole state level)
    E_el = datafile_el.variables['energy'].value[iLowestElectron]
    E_hl = datafile_hl.variables['energy'].value[iHighestHole]
    dE = E_el - E_hl

    # logging.debug(f"get_transition_energy: using data {df.fullpath}")

    return dE


def get_hole_energy_difference(output_folder):
    """ 
    Get the hole energy difference = energy separation between the highest HH and highest LH states.
    Unit: eV
    """
    E_HH = common.getDataFile_in_folder(['energy_spectrum', '_HH'], output_folder, software).variables['Energy'].value[0]   # energy of the first eigenstate
    E_LH = common.getDataFile_in_folder(['energy_spectrum', '_LH'], output_folder, software).variables['Energy'].value[0]   # energy of the first eigenstate
        
    return E_HH - E_LH
