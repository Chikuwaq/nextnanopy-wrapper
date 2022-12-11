"""
Created: 2021/05/27

Basic toolbox (shortcut, module) for nextnanopy. 
Applicable to all nextnano products.

@author: takuma.sato@nextnano.com
"""

# Python includes
from decimal import getcontext
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
# from PIL import Image   # for gif
# from celluloid import Camera   # for gif
# from IPython.display import HTML   # for HTML display of gif

# nextnanopy includes
import nextnanopy as nn
from nextnanopy.utils.misc import mkdir_if_not_exist



# -------------------------------------------------------
# Fundamental physical constants 
# https://physics.nist.gov/cgi-bin/cuu
# -------------------------------------------------------
Planck = 6.62607015E-34  # Planck constant [J.s]
hbar = 1.054571817E-34   # Planck constant / 2Pi in [J.s]
electron_mass = 9.1093837015E-31   # in [kg]
elementary_charge  = 1.602176634*10**(-19)   # [C] elementary charge
speed_of_light = 2.99792458E8   # [m/s]
vacuum_permittivity = 8.854187e-12   # [F/m] 1F = 1 kg^{-1} m^{-2} s^2 C^2 = 1 C^2 / J
Boltzmann = 1.380649e-23   # [J/K]



# -------------------------------------------------------
# Output default formats
# -------------------------------------------------------
figFormat_list = ['.pdf', '.png', '.jpg', '.svg']
figFormat_list_display = ['pdf', 'png', 'jpg', 'svg']

band_colors = {
    'Gamma': 'tomato',
    'CB': 'tomato',
    'HH': 'royalblue',
    'LH': 'forestgreen',
    'SO': 'goldenrod',
    'kp6': 'blueviolet',
    'kp8': 'blueviolet'
    }

labelsize_default = 16
ticksize_default = 14


# -------------------------------------------------------
# Exceptions
# -------------------------------------------------------
class NextnanoInputFileError(Exception):
    """ Exception when the user's nextnano input file contains issue """
    pass

class NextnanoInputFileWarning(Warning):
    """ Warns when the user's nextnano input file contains potential issue """
    pass

# -------------------------------------------------------
# Math
# -------------------------------------------------------
def get_num_of_decimals(x : float):
    from math import floor
    from decimal import Decimal

    getcontext().prec = 16
    y = Decimal(str(x)) - Decimal(str(floor(x)))
    return len(str(y)) - 2

def is_half_integer(x : float):
    from math import floor

    if x < 0: x = -x
    if get_num_of_decimals(x) != 1: return False

    return x - floor(x) == 0.5


def find_maximum(arr):
    """
    Find the maximum of given multidimensional data and return the info.

    Parameters
    ----------
    arr : numpy.ndarray
        2D array

    Returns
    -------
    max_val : double
        maximum value in the flattened data
    indices : tuple
        indices of the maximal element

    Note
    ----
    arr[indices] = max_val holds.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input is not numpy.ndarray!")

    max_val = np.amax(arr)
    indices = np.unravel_index(np.argmax(arr), np.shape(arr))  # get index of the maximum

    return max_val, indices


def find_minimum(arr):
    """
    Find the minimum of given multidimensional data and return the info.

    Parameters
    ----------
    arr : numpy.ndarray
        2D array

    Returns
    -------
    min_val : double
        minimum value in the flattened data
    indices : tuple
        indices of the minimal element

    Note
    ----
    arr[indices] = min_val holds.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input is not numpy.ndarray!")

    min_val = np.amin(arr)
    indices = np.unravel_index(np.argmin(arr), np.shape(arr))  # get index of the minimum

    return min_val, indices


def absolute_squared(x):
    return np.abs(x)**2


# -------------------------------------------------------
# Conversion
# -------------------------------------------------------
scale1ToKilo = 1e-3
scale1ToCenti = 1e2
scale1ToMilli = 1e3
scale1ToMicro = 1e6
scale1ToNano = 1e9
scale1ToPico = 1e12

scale_Angstrom_to_nm = 0.1
scale_eV_to_J = elementary_charge

def electronvolt_to_micron(E):
    """
    Convert energy in eV to micrometer.

    E : array-like
        energy in eV
    """
    energy_in_J = E * elementary_charge   # eV to J
    wavelength_in_meter = Planck * speed_of_light / energy_in_J   # J to m
    return wavelength_in_meter * scale1ToMicro   # m to micron

def wavenumber_to_energy(sound_velocity, k_in_inverseMeter):
    """
    For linear dispersion, convert wavenumber in [1/m] to energy [eV].
    E = hbar * omega = hbar * c * k
    """
    E_in_Joules = hbar * sound_velocity * k_in_inverseMeter
    return E_in_Joules * scale_eV_to_J**(-1)



# -------------------------------------------------------
# Simulation preprocessing
# -------------------------------------------------------

def separateFileExtension(filename):
    """
    Separate file extension from file name.
    Returns the original filename and empty string if extension is absent.
    """
    filename = os.path.split(filename)[1]   # remove paths if present

    if filename[-3:] == '.in':
        extension = '.in'
    elif filename[-4:] == '.xml':
        extension = '.xml'
    elif filename[-5:] == '.negf':
        extension = '.negf'
    else:
        extension = ''

    filename_no_extension = filename.replace(extension, '')
    return filename_no_extension, extension



def detect_software(folder_path, filename):
    """
    Detect software from input file for the argument of nextnanopy.DataFile() and sweep output folder names.
    Useful when the user does not execute nn.InputFile but want to postprocess only.

    Parameters
    ----------
    folder_path : str
        input file folder
    filename : str
        input file name

    Returns
    -------
    software : str
        nextnano solver
    software_short : str
        shorthand of nextnano solver
    extension : str
        file extension
    """

    extension = separateFileExtension(filename)[1]
    if extension == '':
        raise ValueError('Please specify input file with extension (.in, .xml or .negf)')

    InputPath = os.path.join(folder_path, filename)
    try:
        with open(InputPath,'r') as file:
            for line in file:
                if 'simulation-flow-control' in line:
                    software = 'nextnano3'
                    software_short = '_nn3'
                    break
                elif 'run{' in line:
                    software = 'nextnano++'
                    software_short = '_nnp'
                    break
                elif '<nextnano.NEGF' in line or 'nextnano.NEGF{' in line:
                    software = 'nextnano.NEGF'
                    software_short = '_nnNEGF'
                    break
                elif '<nextnano.MSB' in line or 'nextnano.MSB{' in line:
                    software = 'nextnano.MSB'
                    software_short = '_nnMSB'
    except FileNotFoundError:
        raise FileNotFoundError(f'Input file {InputPath} not found!')

    if not software:   # if the variable is empty
        raise NextnanoInputFileError('Software cannot be detected! Please check your input file.')
    else:
        print('\nSoftware detected: ', software)

    return software, software_short, extension



def detect_software_new(inputfile):
    """
    Detect software from nextnanopy.InputFile() object.
    The return value software will be needed for the argument of nextnanopy.DataFile() and sweep output folder names.

    This function is more compact than detect_software() because it makes use of the attributes of nextnanopy.InputFile() object.
    If the object is not executed in the script, it does not have execute_info attributes.
    In that case, you have to explicitly give the output folder name to load output data.
    Therefore, THIS METHOD DOES NOT WORK IF YOU RUN SIMULATIONS WITH nextnanopy.Sweep()!
    """
    try:
        with open(inputfile.fullpath, 'r') as file:
            for line in file:
                if 'simulation-flow-control' in line:
                    software = 'nextnano3'
                    extension = '.in'
                    break
                elif 'run{' in line:
                    software = 'nextnano++'
                    extension = '.in'
                    break
                elif '<nextnano.NEGF' in line:
                    software = 'nextnano.NEGF'
                    extension = '.xml'
                    break
                elif 'nextnano.NEGF{' in line:
                    software = 'nextnano.NEGF'
                    extension = '.negf'
                elif '<nextnano.MSB' in line:
                    software = 'nextnano.MSB'
                    extension = '.xml'
                elif 'nextnano.MSB{' in line:
                    software = 'nextnano.MSB'
                    extension = '.negf'
    except FileNotFoundError:
        raise FileNotFoundError(f'Input file {inputfile.fullpath} not found!')

    if not software:   # if the variable is empty
        raise NextnanoInputFileError('Software cannot be detected! Please check your input file.')
    else:
        print('\nSoftware detected: ', software)

    return software, extension



def prepareInputFile(folderPath, originalFilename, modifiedParamString='', newValue=0, filename_appendix=''):
    """
    Modify parameter in the input file, append specified string to the file name, save the file.

    RETURN:
        new file name
        modified nextnanopy.InputFile object
    """
    InputPath     = os.path.join(folderPath, originalFilename)
    input_file    = nn.InputFile(InputPath)

    if modifiedParamString == '':
        print('\nUsing the default parameters in the input file...\n')
        return originalFilename, input_file

    input_file.set_variable(modifiedParamString, value=newValue)
    name = input_file.get_variable(modifiedParamString).name
    value = input_file.get_variable(modifiedParamString).value
    print(f'\nUsing modified input parameter:\t${name} = {value}')

    filename_no_extension, extension = separateFileExtension(originalFilename)
    if extension == '':
        raise ValueError('Include file extension to the input file name!')
    newFilename = filename_no_extension + filename_appendix + extension
    print(f'Saving input file as:\t{newFilename}\n')
    input_file.save(os.path.join(folderPath, newFilename), overwrite=True)   # update input file name

    return newFilename, input_file


# -------------------------------------------------------
# Bandedge and k.p parameters
# -------------------------------------------------------
def get_bandgap_at_T(bandgap_at_0K, alpha, beta, T):
    """ Varshni formula """
    return bandgap_at_0K - alpha * T**2 / (T + beta)


def get_factor_zb(Eg, deltaSO):
    """
    Temperature-dependent factor for the conversion among effective mass, S and Ep
    """
    return (Eg + 2. * deltaSO / 3.) / Eg / (Eg + deltaSO)

def mass_from_kp_parameters(Ep, S, Eg, deltaSO):
    factor = get_factor_zb(Eg, deltaSO)   # independent of S and Ep, but temperature-dependent
    mass = 1. / (S + Ep * factor)
    return mass


def Ep_from_mass_and_S(mass, S, Eg, deltaSO):
    factor = get_factor_zb(Eg, deltaSO)   # independent of S and Ep, but temperature-dependent
    Ep = (1./mass - S) / factor
    return Ep


def Ep_from_P(P):
    """
    Convert the Kane parameter P [eV nm] into energy Ep [eV].
    #NOTE: nextnano++ output is in units of [eV Angstrom].
    """
    P_in_SI = P * scale_eV_to_J * scale1ToNano**(-1)
    Ep_in_SI = P_in_SI**2 * 2 * electron_mass / (hbar**2)
    return Ep_in_SI / scale_eV_to_J


def P_from_Ep(Ep):
    """
    Convert the Kane energy Ep [eV] into P [eV nm].
    #NOTE: nextnano++ output is in units of [eV Angstrom].
    """
    from math import sqrt
    Ep_in_SI = Ep * scale_eV_to_J
    P_in_SI = hbar * sqrt(Ep_in_SI / 2 / electron_mass)
    return P_in_SI * scale_eV_to_J**(-1) * scale1ToNano


# TODO: extend to WZ
def evaluate_and_rescale_S(db_Ep, db_S, db_L, db_N, eff_mass, Eg, deltaSO, evaluateS, rescaleS, rescaleSTo):
    """
    Identical to nnp implementation. 'db_' denotes database values.
    massFromKpParameters changes mass, but it isn't used to rescale S here because
    (*) old_S + old_Ep * factor = new_S + new_Ep * factor
    NEGF uses mass when rescaling S.
    """
    factor = get_factor_zb(Eg, deltaSO)   # independent of S and Ep, but temperature-dependent

    if evaluateS:
        S = 1. / eff_mass - db_Ep * factor
    else:
        S = db_S


    if rescaleS:   # rescale S to given value, and adjust Ep to preserve the effective mass
        new_S = rescaleSTo
        new_Ep = db_Ep + (S - new_S) / factor   # using formula (*)
    else:
        new_Ep = db_Ep
        new_S = S

    if (isinstance(new_Ep, float) and new_Ep < 0) or (hasattr(new_Ep, '__iter__') and any(x < 0 for x in new_Ep)):
        raise RuntimeError('Ep parameter has become negative while rescaling S!')

    # L' and N' get modified by the change of Ep
    cSchroedinger = hbar**2 / 2 / electron_mass
    new_L = db_L + cSchroedinger * (new_Ep - db_Ep) / Eg
    new_N = db_N + cSchroedinger * (new_Ep - db_Ep) / Eg

    return new_S, new_Ep, new_L, new_N



# TODO: extend to WZ
def rescale_Ep_and_get_S(old_Ep, old_S, old_L, old_N, rescaleEpTo, Eg, deltaSO):
    """
    Rescale Ep to given value, and adjust S to preserve the electron effective mass:
    (*) old_S + old_Ep * factor = new_S + rescaleEpTo * factor
    """
    factor = get_factor_zb(Eg, deltaSO)   # independent of S and Ep, but temperature-dependent
    new_S = old_S + (old_Ep - rescaleEpTo) * factor

    # L' and N' get modified by the change of Ep
    # This keeps the kp6 values at T=0K intact.
    cSchroedinger = hbar**2 / 2 / electron_mass
    new_L = old_L + cSchroedinger * (rescaleEpTo - old_Ep) / Eg
    new_N = old_N + cSchroedinger * (rescaleEpTo - old_Ep) / Eg
    return new_S, new_L, new_N


def get_8kp_from_6kp_NEGF(mass, rescaleSTo, Eg_0K, Eg_finiteTemp, deltaSO, L_6kp, N_6kp):
    """
    Imitate NEGF implementation.
    1. Calculate Ep by rescaling at T=0
    2. Calculate L', M, N', S from this Ep but using Eg at nonzero T
    """
    Ep = Ep_from_mass_and_S(mass, rescaleSTo, Eg_0K, deltaSO)
    
    correction = Ep / Eg_finiteTemp   # L, N in the database are in units of hbar^2/2m0
    Lprime = L_6kp + correction
    Nprime = N_6kp + correction
    
    factor = get_factor_zb(Eg_finiteTemp, deltaSO)   # independent of S and Ep, but temperature-dependent
    new_S = 1./mass - Ep*factor

    return Lprime, Nprime, new_S




# -------------------------------------------------------
# Access to output data
# -------------------------------------------------------

def check_if_simulation_has_run(input_file):
    """
    Check if simulation has been run. If not, ask the user if the postprocessing should continue.

    INPUT:
        input_file      nn.InputFile() object
    """
    try:
        folder = input_file.folder_output
    except:
        determined = False
        while not determined:
            choice = input('Simulation has not been executed. Continue? [y/n]')
            if choice == 'n':
                raise RuntimeError('Terminated nextnanopy.')
            elif choice == 'y':
                determined = True
            else:
                print("Invalid input.")
                continue


def getSweepOutputFolderName(filename, *args):
    """
    nextnanopy.sweep.execute_sweep() generates output folder with this name

    INPUT:
        filename
        args = SweepVariableString1, SweepVariableString2, ...

    RETURN:
        string of sweep output folder name

    """
    filename_no_extension = separateFileExtension(filename)[0]
    output_folderName = filename_no_extension + '_sweep'

    for sweepVar in args:
        if not isinstance(sweepVar, str):
            raise TypeError(f'Argument {sweepVar} must be a string!')
        output_folderName += '__' + sweepVar

    return output_folderName


def getSweepOutputFolderPath(filename, software, *args):
    """
    Get the output folder path generated by nextnanopy.sweep.execute_sweep().

    Parameters
    ----------
    filename : str
        input file name (may include absolute/relative paths)
    software : str
        nextnano solver
    *args : str
        SweepVariableString1, SweepVariableString2, ...

    Returns
    -------
    output_folder_path : str
        sweep output folder path

    """
    filename_no_extension = separateFileExtension(filename)[0]
    output_folder_path = os.path.join(nn.config.get(software, 'outputdirectory'), filename_no_extension + '_sweep')

    if len(args) == 0: raise ValueError("Sweep variable string is missing in the argument!")

    for sweepVar in args:
        if not isinstance(sweepVar, str):
            raise TypeError(f'Argument {sweepVar} must be a string!')
        output_folder_path += '__' + sweepVar

    return output_folder_path


def get_output_subfolder_path(sweep_output_folder_path, input_file_name):
    """
    Return output folder path corresponding to the input file

    Parameters
    ----------
    input_file_name : str
        input file name or path

    Returns
    -------
    str
        path to output folder

    """
    subfolder_name = separateFileExtension(input_file_name)[0]
    return os.path.join(sweep_output_folder_path, subfolder_name)


def getSweepOutputSubfolderName(filename, sweepCoordinates):
    """
    nextnanopy.sweep.execute_sweep() generates output subfolders with this name

    INPUT:
        filename
        {sweepVariable1: value1, sweepVariable2: value2, ...}

    RETURN:
        string of sweep output subfolder name

    """
    filename_no_extension = separateFileExtension(filename)[0]
    output_subfolderName = filename_no_extension + '__'

    for sweepVar, value in sweepCoordinates.items():
        if not isinstance(sweepVar, str):
            raise TypeError('key must be a string!')
        try:
            val = str(value)
        except ValueError:
            print('value cannot be converted to string!')
            raise
        else:
            output_subfolderName +=  sweepVar + '_' + val + '_'

    return output_subfolderName



def getDataFile(keywords, name, software, exclude_keywords=None):
    """
    Get single nextnanopy.DataFile of output data in the directory matching name with the given string keyword.

    Parameters
    ----------
    keywords : str or list of str
        Find output data file with the names containing single keyword or multiple keywords (AND search)
    name : str
        input file name (= output subfolder name). May contain extensions and/or fullpath.
    software : str
        nextnano solver.
    exclude_keywords : str or list of str, optional
        Files containing these keywords in the file name are excluded from search.

    """
    outputFolder = nn.config.get(software, 'outputdirectory')
    filename_no_extension = separateFileExtension(name)[0]
    outputSubfolder = os.path.join(outputFolder, filename_no_extension)

    return getDataFile_in_folder(keywords, outputSubfolder, software, exclude_keywords=exclude_keywords)


def getDataFile_in_folder(keywords, folder_path, software, exclude_keywords=None):
    """
    Get single nextnanopy.DataFile of output data with the given string keyword(s) in the specified folder.

    Parameters
    ----------
    keywords : str or list of str
        Find output data file with the names containing single keyword or multiple keywords (AND search)
    folder_path : str
        absolute path of output folder in which the datafile should be sought
    software : str
        nextnano solver.
    exclude_keywords : str or list of str, optional
        Files containing these keywords in the file name are excluded from search.

    Returns
    -------
    nextnanopy.DataFile object of the simulation data

    """
    # if only one keyword is provided, make a list with single element to simplify code
    if isinstance(keywords, str):
        keywords = [keywords]
    elif not isinstance(keywords, list):
        raise TypeError("Argument 'keywords' must be either str or list")
    if isinstance(exclude_keywords, str):
        exclude_keywords = [exclude_keywords]
    elif not isinstance(exclude_keywords, list) and exclude_keywords is not None:
        raise TypeError("Argument 'exclude_keywords' must be either str or list")

    if exclude_keywords is None:
        message = " '" + "', '".join(keywords) + "'"
    else:
        message = " '" + "', '".join(keywords) + "', excluding keyword(s) '" + "', '".join(exclude_keywords) + "'"

    print(f'\nSearching for output data {message}...')

    # Search output data using nn.DataFolder.find(). If multiple keywords are provided, find the intersection of files found with each keyword.
    list_of_sets = [set(nn.DataFolder(folder_path).find(keyword, deep=True)) for keyword in keywords]
    candidates = list_of_sets[0]
    for s in list_of_sets:
        candidates = s.intersection(candidates)
    list_of_files = list(candidates)

    def should_be_excluded(filepath):
        filename = os.path.split(filepath)[1]
        return any(key in filename for key in exclude_keywords)

    if exclude_keywords is not None:
        list_of_files = [file for file in list_of_files if not should_be_excluded(file)]


    # validate the search result
    if len(list_of_files) == 0:
        raise FileNotFoundError(f"No output file found!")
    elif len(list_of_files) == 1:
        file = list_of_files[0]
    else:
        print(f"More than one output files found!")
        for count, file in enumerate(list_of_files):
            filename = os.path.split(file)[1]
            print(f"Choice {count}: {filename}")
        determined = False
        while not determined:
            choice = input('Enter the index of data you need: ')
            if choice == 'q':
                raise RuntimeError('Terminated nextnanopy.')
            try:
                choice = int(choice)
            except ValueError:
                print("Invalid input. (Type 'q' to quit)")
                continue
            else:
                if choice < 0 or choice >= len(list_of_files):
                    print("Index out of bounds. Type 'q' to quit")
                    continue
                else:
                    determined = True
        file = list_of_files[choice]

    if __debug__: print("Found:\n", file)

    try:
        return nn.DataFile(file, product=software)
    except NotImplementedError:
        raise NotImplementedError(f'Nextnanopy does not support datafile for {file}')


def getDataFiles(keywords, name, software, exclude_keywords=None):
    """
    Get multiple nextnanopy.DataFiles of output data with the given string keyword(s).

    Parameters
    ----------
    keywords : str or list of str
        Find output data file with the names containing single keyword or multiple keywords (AND search).
    name : str
        input file name (= output subfolder name) without folder paths. May contain extension '.in' or '.xml'.
    software : str
        nextnano solver
    exclude_keywords : str or list of str, optional
        Files containing these keywords in the file name are excluded from search.

    """
    outputFolder = nn.config.get(software, 'outputdirectory')
    filename_no_extension = separateFileExtension(name)[0]
    outputSubFolder = os.path.join(outputFolder, filename_no_extension)

    return getDataFiles_in_folder(keywords, outputSubFolder, software, exclude_keywords=exclude_keywords)


def getDataFiles_in_folder(keywords, folder_path, software, exclude_keywords=None):
    """
    Get multiple nextnanopy.DataFiles of output data with the given string keyword(s) in the specified folder.

    Parameters
    ----------
    keywords : str or list of str
        Find output data file with the names containing single keyword or multiple keywords (AND search)
    folder_path : str
        absolute path of output folder in which the datafile should be sought
    software : str
        nextnano solver
    exclude_keywords : str or list of str, optional
        Files containing these keywords in the file name are excluded from search.

    Returns
    -------
    list of nextnanopy.DataFile objects of the simulation data

    """
    # if only one keyword is provided, make a list with single element to simplify code
    if isinstance(keywords, str):
        keywords = [keywords]
    elif not isinstance(keywords, list):
        raise TypeError("Argument 'keywords' must be either str or list")
    if isinstance(exclude_keywords, str):
        exclude_keywords = [exclude_keywords]
    elif not isinstance(exclude_keywords, list) and exclude_keywords is not None:
        raise TypeError("Argument 'exclude_keywords' must be either str or list")

    if exclude_keywords is None:
        message = "with keyword(s) '" + "', '".join(keywords) + "'"
    else:
        message = "with keyword(s) '" + "', '".join(keywords) + "', excluding '" + "', '".join(exclude_keywords) + "'"

    print(f'\nSearching for output data {message}...')

    # Search output data using nn.DataFolder.find(). If multiple keywords are provided, find the intersection of files found with each keyword.
    list_of_sets = [set(nn.DataFolder(folder_path).find(keyword, deep=True)) for keyword in keywords]
    candidates = list_of_sets[0]
    for s in list_of_sets:
        candidates = s.intersection(candidates)
    list_of_files = list(candidates)

    def should_be_excluded(filepath):
        filename = os.path.split(filepath)[1]
        return any(key in filename for key in exclude_keywords)

    if exclude_keywords is not None:
        list_of_files = [file for file in list_of_files if not should_be_excluded(file)]

    # validate the search result
    if len(list_of_files) == 0:
        raise FileNotFoundError(f"No output file found!")
    elif len(list_of_files) == 1:
        warnings.warn("getDataFiles_in_folder(): Only one output file found!", category=RuntimeWarning)

    if __debug__: print("Found:\n", list_of_files)

    try:
        datafiles = [nn.DataFile(file, product=software) for file in list_of_files]
    except NotImplementedError:
        raise NotImplementedError('Nextnanopy does not support datafile')

    return datafiles




# -------------------------------------------------------
# Data postprocessing
# -------------------------------------------------------

def convert_grid(arr, old_grid, new_grid):
    """
    Convert grid of an array.
    Needed if two physical quantities that you want to overlay are on a different grid.

    Parameters
    ----------
    arr : array-like
        array to be converted
    old_grid : array-like
        grid points on which arr is defined
    new_grid : array-like
        grid points on which new arr should sit

    Returns
    -------
    arr_new : array-like
        array on the new grid

    Requires
    --------
    SciPy
    """

    from scipy.interpolate import splev, splrep

    spl = splrep(old_grid, arr)     # interpolate
    arr_new = splev(new_grid, spl)  # map to new grid
    return arr_new



def cutOff_edges1D(arr, x_grid, start_position, end_position):
    """
    Cut off the edges of 1D real space array.
    If the specified limits extend beyond the original grid, no cut-off is performed.

    Parameters
    ----------
    arr : array-like
        1D array to be processed
    x_grid : array-like
        grid points on which arr is defined
    start_position : real
        new array starts from this position
    end_position : real
        new array ends at this position

    Returns
    -------
    array-like
        arr without edges

    """
    if np.ndim(arr) != 1: raise ValueError("Array must be one-dimensional!")

    num_gridPoints = len(x_grid)

    # input validation
    if len(arr) != num_gridPoints:  # 'averaged = yes' 'boxed = yes' may lead to inconsistent number of grid points
        print(len(arr), num_gridPoints)
        raise ValueError('Array size does not match the number of real space grid points')
    if end_position < start_position:
        raise ValueError('Illegal start and end positions!')

    # find start & end index
    start_index = 0
    end_index = num_gridPoints - 1
    for i in range(num_gridPoints-1):
        if x_grid[i] <= start_position < x_grid[i + 1]:
            start_index = i
        if x_grid[i] < end_position <= x_grid[i + 1]:
            end_index = i + 1

    return arr[start_index : end_index + 1]



def findCell(arr, wanted_value):
    """
    Find the grid cell that contains given wanted position and return index.
    #TODO: numpy.array.argmin() can be applied also to non-monotonic arrays
    """
    num_nodes = len(arr)
    cnt = 0
    for i in range(num_nodes-1):
        if arr[i] <= wanted_value < arr[i + 1]:
            start_index = i
            end_index = i + 1
            cnt = cnt + 1

    if cnt == 0:
        raise RuntimeError('No grid cells found that contain the point x = {wanted_value}')
    if cnt > 1:
        raise RuntimeError(f'Multiple grid cells found that contain the point x = {wanted_value}')
    return start_index, end_index



def getValueAtPosition(quantity_arr, position_arr, wantedPosition):
    """
    Get value at given position.
    If the position does not match any of array elements due to inconsistent gridding, interpolate array and return the value at wanted position.
    """
    if len(quantity_arr) != len(position_arr):
        raise ValueError('Array size does not match!')

    start_idx, end_idx = findCell(position_arr, wantedPosition)

    # linear interpolation
    x_start = position_arr[start_idx]
    x_end   = position_arr[end_idx]
    y_start = quantity_arr[start_idx]
    y_end   = quantity_arr[end_idx]
    tangent = (y_end - y_start) / (x_end - x_start)
    return tangent * (wantedPosition - x_start) + y_start




# -------------------------------------------------------
# Plotting
# -------------------------------------------------------


def getRowColumnForDisplay(num_elements):
    """
    Determine arrangement of multiple plots in one sheet.

    INPUT:
        number of plots

    RETURN:
        number of rows
        number of columns
        (which satisfy number of rows <= number of columns)
    """
    import math
    n = num_elements

    # if not n % 2 == 0: n = n+1   # avoid failure of display when n is odd
    num_rows = int(n)
    num_columns = 1
    # if __debug__: print(num_rows, num_columns)

    if n < 3: return num_rows, num_columns

    while (np.double(num_columns) / np.double(num_rows) < 0.7):   # try to make it as square as possible
        # if __debug__: print('n=', n)
        k = math.floor(math.sqrt(n))

        while not n % k == 0: k -= 1
        # if __debug__: print('k=', k)
        num_rows    = int(n / k)
        num_columns = int(k)
        # if __debug__: print(num_rows, num_columns)
        n += 1

    return num_rows, num_columns



def get_maximum_points(quantity_arr, position_arr):
    if isinstance(quantity_arr, int) or isinstance(quantity_arr, float):
        warnings.warn(f"get_maximum_points(): Only one point exists in the array {quantity_arr}", category=RuntimeWarning)
        return position_arr[0], quantity_arr

    if len(quantity_arr) != len(position_arr):
        raise ValueError('Array size does not match!')
    ymax = np.amax(quantity_arr)
    if np.size(ymax) > 1:
        print("Multiple maxima found. Taking the first...")
        ymax = ymax[0]
    xmaxIndex = np.where(quantity_arr == ymax)[0]
    xmax = position_arr[xmaxIndex.item(0)]             # type(xmaxIndex.item(0)) is 'int'

    return xmax, ymax



def generateColorscale(colormap, minValue, maxValue):
    """
    Generate a color scale with given colormap and range of values.
    """
    return plt.cm.ScalarMappable( cmap=colormap, norm=plt.Normalize(vmin=minValue, vmax=maxValue) )



def mask_part_of_array(arr, string='flat', tolerance=1e-4, cut_range=[]):
    """
    Mask some elements in an array to plot limited part of data.

    INPUT:
        arr           data array to be masked
        string        specify mask method. 'flat' masks flat part of the data, while 'range' masks the part specified by the index range.
        tolerance     for 'flat' mask method
        cut_range     list: range of index to define indies to be masked

    RETURN:
        masked array

    """
    if not isinstance(arr, np.ndarray):
        raise TypeError('Given array is not numpy.ndarray')

    arr_size = len(arr)
    new_arr = np.ma.array(arr, mask = [0 for i in range(arr_size)])   # non-masked np.ma.array with given data arr

    if string == 'flat':  # mask data points where the values are almost flat
        num_neighbours = 10   # number of neighbouring points to be tested

        def isFlat_forward(i, k):
            forward = arr[(i + k) % arr_size]   # using modulo to avoid index out of bounds
            return np.abs(arr[i] - forward) < tolerance

        def isFlat_backward(i, k):
            backward = arr[(i - k) % arr_size]   # using modulo to avoid index out of bounds
            return np.abs(arr[i] - backward) < tolerance

        for i in range(arr_size):
            flat_nearby = all(isFlat_forward(i, k) and isFlat_backward(i, k) for k in range(num_neighbours))

            if flat_nearby:
                new_arr.mask[i] = True
                continue

            if i in range(num_neighbours):
                if all(isFlat_forward(i, k) for k in range(num_neighbours)):
                    new_arr.mask[i] = True
                    continue

            if (arr_size-1) - i in range(num_neighbours):
                if all(isFlat_backward(i, k) for k in range(num_neighbours)):
                    new_arr.mask[i] = True
                    continue

            # old method
            # if i == 0 or i == arr_size-1:  continue  # edge points

            # # check nearest neighbours
            # flat_before = np.abs(arr[i] - arr[i-1]) < tolerance
            # flat_after = np.abs(arr[i] - arr[i+1]) < tolerance

            # if i == 1 or i == arr_size-2:  # second edge points
            #     if flat_before and flat_after:
            #         new_arr.mask[i] = True
            #     continue

            # check next nearest neighbours
            # flat_nextNearest = np.abs(arr[i] - arr[i-2]) < tolerance and np.abs(arr[i] - arr[i+2]) < tolerance

            # if i == 2 or arr_size-3:  # third edge points
            #     if flat_before and flat_after and flat_nextNearest:
            #         new_arr.mask[i] = True
            #     continue



    if string == 'range':
        if cut_range == []: raise ValueError('Specify the range to cut!')

        cut_indices = np.arange(cut_range[0], cut_range[1], 1)
        for i in cut_indices:
            new_arr.mask[i] = True

    return new_arr


def set_plot_labels(ax, x_label, y_label, title):
    """
    Set the labels and optimize their sizes (matplotlib default of font size is too small!)
    """
    ax.set_xlabel(x_label, fontsize=labelsize_default)  # r with superscript works
    ax.set_ylabel(y_label, fontsize=labelsize_default)
    ax.set_title(title, fontsize=labelsize_default)
    ax.tick_params(axis='x', labelsize=ticksize_default)
    ax.tick_params(axis='y', labelsize=ticksize_default)
    return ax


def getPlotTitle(originalTitle):
    """
    If the title is too long for display, omit the intermediate letters
    """
    title = separateFileExtension(originalTitle)[0]   # remove extension if present

    if len(title) > 25:
        beginning = title[:10]
        last  = title[-10:]
        title = beginning + ' ... ' + last

    return title



def export_figs(figFilename, figFormat, software, outputSubfolderName='nextnanopy', output_folder_path='', fig=None):
    """
    Export all the matplotlib.pyplot objects in multi-page PDF file or other image formats with a given file name.

    Parameters
    ----------
    figFilename : str
        file name of the exported figure

    figFormat : str
        PDF = vector graphic
        PNG = high quality, lossless compression, large size (recommended)
        JPG = lower quality, lossy compression, small size (not recommended)
        SVG = supports animations and image editing for e.g. Adobe Illustrator

    software : str
        nextnano solver. Used to get the output directory specified in the config

    outputSubfolderName : str, optional
        subfolder name in the output directory specified in the config
        The default is 'nextnanopy'

    output_folder_path : str, optional
        If present, the file will be saved to this path and outputSubfolderName will be ignored.
        The default is ''.

    fig : matplotlib.subplot object, optional
        Needed if non-PDF format is desired. The default is None.

    Returns
    -------
    None.

    NOTE:
        fig, ax = plt.subplots() must exist, i.e. subplot object(s) must be instantiated before calling this method.
        specify image format in the argument of this function if non-PDF format is desired.

    LIMITATION:
        PNG and other non-PDF formats cannot generate multiple pages and ends up with one plot when multiple subplots instances exist.

    """
    import matplotlib.backends.backend_pdf as backendPDF

    # validate arguments
    if '.' not in figFormat:
        figFormat = '.' + figFormat
    if figFormat not in figFormat_list:
        raise ValueError(f"Non-supported figure format! It must be one of the following:\n{figFormat_list_display}")

    if fig is None and not figFormat == '.pdf':
        raise ValueError("Argument 'fig' must be specified to export non-PDF images!")

    if isinstance(fig, list) and len(fig) > 1 and not figFormat == '.pdf':
        raise RuntimeError("Non-PDF formats cannot generate multiple pages.")

    # prepare output subfolder path
    if output_folder_path:
        outputSubfolder = os.path.join(output_folder_path, "nextnanopy")
    else:
        outputSubfolderName = separateFileExtension(outputSubfolderName)[0]   # chop off file extension if any
        outputSubfolder = os.path.join(nn.config.get(software, 'outputdirectory'), outputSubfolderName)

    mkdir_if_not_exist(outputSubfolder)
    export_fullpath = os.path.join(outputSubfolder, figFilename + figFormat)
    print(f'\nExporting figure to: \n{export_fullpath}\n')

    if figFormat == '.pdf':
        with backendPDF.PdfPages(export_fullpath, False) as pdf:
            for figure in range(1, plt.gcf().number + 1):
                pdf.savefig(figure)
    else:
        fig.savefig(export_fullpath, dpi=200)


