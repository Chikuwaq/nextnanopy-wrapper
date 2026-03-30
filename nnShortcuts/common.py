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
from matplotlib.colors import ListedColormap
import warnings
import logging
# from PIL import Image   # for gif
# from celluloid import Camera   # for gif
# from IPython.display import HTML   # for HTML display of gif

# nextnanopy includes
import nextnanopy as nn
from nextnanopy.utils.misc import mkdir_if_not_exist

# my includes
from nnShortcuts.default_colors import DefaultColors
from nnShortcuts.scientific_plotter import ScientificPlotter
from nnShortcuts.path_handler import PathHandler


# -------------------------------------------------------
# Exceptions
# -------------------------------------------------------
class NextnanopyScriptError(Exception):
    """ 
    Exception when the user's nextnanopy script contains an issue
    Should only be raised when the cause is certainly in the user's Python script/command and not in nextnanopy or its wrapper libraries.
    """
    pass

class NextnanoInputFileError(Exception):
    """ Exception when the user's nextnano input file contains an issue """
    pass

class NextnanoInputFileWarning(Warning):
    """ Warns when the user's nextnano input file contains potential issue """
    pass


class CommonShortcuts:
    # nextnano solver
    product_name = 'common'

    # -------------------------------------------------------
    # Fundamental physical constants 
    # https://physics.nist.gov/cuu/Constants/index.html
    # -------------------------------------------------------
    Planck = 6.62607015E-34  # Planck constant [J.s]
    hbar = 1.054571817E-34   # Planck constant / 2Pi in [J.s]
    electron_mass = 9.1093837139E-31   # in [kg]
    elementary_charge  = 1.602176634*10**(-19)   # [C] elementary charge
    speed_of_light = 2.99792458E8   # [m/s]
    vacuum_permittivity = 8.8541878188e-12   # [F/m] 1F = 1 kg^{-1} m^{-2} s^2 C^2 = 1 C^2 / J
    Boltzmann = 1.380649e-23   # [J/K]

    position_min = -10000.
    position_max = 10000.

    DUMMYVALUE = 546578653183735435

    # -------------------------------------------------------
    # Output default formats
    # -------------------------------------------------------
    figFormat_list = ['.pdf', '.eps', '.png', '.jpg', '.svg']
    figFormat_list_display = ['pdf', 'eps', 'png', 'jpg', 'svg']

    labelsize_default = 16
    ticksize_default = 14

    units_LDOS = r'$\mathrm{nm}^{-1}\,\mathrm{eV}^{-1}$'
    units_2d_carrier_density = r'$10^{18}\,\mathrm{cm}^{-3}\,\mathrm{eV}^{-1}$'
    units_2d_inplane_resolved_carrier_density= r'$10^{18}\mathrm{cm}^{-3}\,\mathrm{nm}^{2}\,\mathrm{eV}^{-1}$'
    axis_label_position = "Position $z$ ($\mathrm{nm}$)"
    axis_label_energy = "Energy ($\mathrm{eV}$)"
    axis_label_temperature = "Temperature ($\mathrm{K}$)"
    axis_label_inplane_k = "$k_x$ ($\mathrm{nm}^{-1}$)"

    # -------------------------------------------------------
    # Constructor
    # -------------------------------------------------------
    def __init__(self, loglevel=logging.INFO):
        self.position_axis_key = None
        self.bandedge_filename = None
        self.conduction_bandedge_key = None
        self.heavy_hole_bandedge_key = None
        self.light_hole_bandedge_key = None
        self.SO_hole_bandedge_key = None
        self.wavefunction_name = None

        self.band_names = {
            'Gamma': 'Gamma', 
            'CB': 'CB', 
            'HH': 'HH', 
            'LH': 'LH', 
            'SO': 'SO',
            'kp6': 'kp6', 
            'kp8': 'kp8'
        }
        self.default_colors = DefaultColors(self.band_names)

        print("loglevel = ", loglevel)
        # log setting
        fmt = '[%(levelname)s] %(message)s'
        logging.basicConfig(
            level=loglevel, 
            format=fmt
        )
        logging.captureWarnings(True)

        # customize warning format
        def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
            # return '%s:%s:\n%s: %s' % (filename, lineno, category.__name__, message)  # TODO: how to color the warning? Maybe useful https://github.com/Delgan/loguru
            return f"{category.__name__}: {message} ({filename}:{lineno})"
            # return "%(filename)s:%(lineno)d:\n %(category.__name__)s: %(message)s"
        warnings.formatwarning = warning_on_one_line
        logging.getLogger('matplotlib.font_manager').disabled = True  # suppress excessive 'findfont' warnings


        logging.info(f'{self.product_name} shortcuts initialized.')

    # -------------------------------------------------------
    # nextnano corporate colormap for matplotlib
    # -------------------------------------------------------
    """
    @author: herbert, niklas.pichel, takuma.sato
    """
    def get_nn_colormap(num: int = 90, bright_scheme=False):
        """
        Get nextnano corporate colormap for matplotlib
        """
        color_dark = np.empty([256,4])      # dark color scheme
        color_bright = np.empty([256,4])    # bright color scheme
        
        for i in range(256):
            if i <= num and num != 0:
                color_dark[i,:] = np.array([17/256*i/num, 173/256*i/num, 181/256*i/num, 1])
            else:
                color_dark[i,:] = np.array([17/256, 173/256, 181/256, 1-(i-num)/(256-num)]) 
        
        for i in range(256):
            color_bright[i,:] = np.array([17/256, 173/256, 181/256, i/256])
    
        if bright_scheme:
            return ListedColormap(color_bright)
        else:
            return ListedColormap(color_dark)



    # -------------------------------------------------------
    # Math
    #
    # We make methods static because:
    # - these utility functions do not depend on the class state but makes sense that they belong to the class
    # - we want to make this method available without instantiation of an object.
    # -------------------------------------------------------
    @staticmethod
    def is_half_integer(x : float):
        from math import floor

        def get_num_of_decimals(x : float):
            from math import floor
            from decimal import Decimal

            getcontext().prec = 16
            y = Decimal(str(x)) - Decimal(str(floor(x)))
            return len(str(y)) - 2

        if x < 0: x = -x
        if get_num_of_decimals(x) != 1: return False

        return x - floor(x) == 0.5


    @staticmethod
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
            raise TypeError(f"Input must be numpy.ndarray, but is {type(arr)}")

        max_val = np.amax(arr)
        indices = np.unravel_index(np.argmax(arr), np.shape(arr))  # get index of the maximum

        return max_val, indices


    @staticmethod
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
            raise TypeError(f"Input must be numpy.ndarray, but is {type(arr)}")

        min_val = np.amin(arr)
        indices = np.unravel_index(np.argmin(arr), np.shape(arr))  # get index of the minimum

        return min_val, indices


    def absolute_squared(x):
        return np.abs(x)**2


    # -------------------------------------------------------
    # Conversion of units
    #
    # We make methods static because:
    # - these utility functions do not depend on the class state but makes sense that they belong to the class
    # - we want to make this method available without instantiation of an object.
    # -------------------------------------------------------
    scale1ToKilo = 1e-3
    scale1ToCenti = 1e2
    scale1ToMilli = 1e3
    scale1ToMicro = 1e6
    scale1ToNano = 1e9
    scale1ToPico = 1e12

    scale_Angstrom_to_nm = 0.1
    scale_eV_to_J = elementary_charge
    scale_J_to_eV = 1.0 / elementary_charge

    @staticmethod
    def electronvolt_to_micron(E):
        """
        Convert energy in eV to micrometer.

        E : array-like
            energy in eV
        """
        energy_in_J = E * CommonShortcuts.elementary_charge   # eV to J
        wavelength_in_meter = CommonShortcuts.Planck * CommonShortcuts.speed_of_light / energy_in_J   # J to m
        return wavelength_in_meter * CommonShortcuts.scale1ToMicro   # m to micron

    @staticmethod
    def micron_to_electronvolt(wavelength_in_micron):
        """
        Convert energy in micron to eV.

        wavelength_in_micron : array-like
            energy in micron
        """
        wavelength_in_meter = wavelength_in_micron / CommonShortcuts.scale1ToMicro
        energy_in_J = CommonShortcuts.Planck * CommonShortcuts.speed_of_light / wavelength_in_meter
        return energy_in_J / CommonShortcuts.elementary_charge
    
    @staticmethod
    def wavenumber_to_energy(sound_velocity, k_in_inverseMeter):
        """
        For linear dispersion, convert wavenumber in [1/m] to energy [eV].
        E = hbar * omega = hbar * c * k
        """
        E_in_Joules = CommonShortcuts.hbar * sound_velocity * k_in_inverseMeter
        return E_in_Joules * CommonShortcuts.scale_eV_to_J**(-1)



    # -------------------------------------------------------
    # Conversion for bandedge and k.p parameters
    #
    # We make methods static because:
    # - these utility functions do not depend on the class state but makes sense that they belong to the class
    # - we want to make this method available without instantiation of an object.
    # -------------------------------------------------------
    @staticmethod
    def get_bandgap_at_T(bandgap_at_0K, alpha, beta, T):
        """ Varshni formula """
        return bandgap_at_0K - alpha * T**2 / (T + beta)

    @staticmethod
    def get_factor_zb(Eg, deltaSO):
        """
        Temperature-dependent factor for the conversion among effective mass, S and Ep
        """
        return (Eg + 2. * deltaSO / 3.) / Eg / (Eg + deltaSO)

    @staticmethod
    def mass_from_kp_parameters(Ep, S, Eg, deltaSO):
        factor = CommonShortcuts.get_factor_zb(Eg, deltaSO)   # independent of S and Ep, but temperature-dependent
        mass = 1. / (S + Ep * factor)
        return mass

    @staticmethod
    def Ep_from_mass_and_S(mass, S, Eg, deltaSO):
        factor = CommonShortcuts.get_factor_zb(Eg, deltaSO)   # independent of S and Ep, but temperature-dependent
        Ep = (1./mass - S) / factor
        return Ep

    @staticmethod
    def Ep_from_P(P):
        """
        Convert the Kane parameter P [eV nm] into energy Ep [eV].
        #NOTE: nextnano++ output is in units of [eV Angstrom].
        """
        P_in_SI = P * CommonShortcuts.scale_eV_to_J * CommonShortcuts.scale1ToNano**(-1)
        Ep_in_SI = P_in_SI**2 * 2 * CommonShortcuts.electron_mass / (CommonShortcuts.hbar**2)
        return Ep_in_SI / CommonShortcuts.scale_eV_to_J

    @staticmethod
    def P_from_Ep(Ep):
        """
        Convert the Kane energy Ep [eV] into P [eV nm].
        #NOTE: nextnano++ output is in units of [eV Angstrom].
        """
        from math import sqrt
        Ep_in_SI = Ep * CommonShortcuts.scale_eV_to_J
        P_in_SI = CommonShortcuts.hbar * sqrt(Ep_in_SI / 2 / CommonShortcuts.electron_mass)
        return P_in_SI * CommonShortcuts.scale_eV_to_J**(-1) * CommonShortcuts.scale1ToNano

    @staticmethod
    def evaluate_and_rescale_S( 
            db_Ep, 
            db_S, 
            db_L, 
            db_N, 
            eff_mass, 
            Eg, 
            deltaSO, 
            evaluateS, 
            rescaleS, 
            rescaleSTo
            ):
        """
        Identical to nnp implementation. 'db_' denotes database values.
        massFromKpParameters changes mass, but it isn't used to rescale S here because
        (*) old_S + old_Ep * factor = new_S + new_Ep * factor
        NEGF uses mass when rescaling S.
        """
        factor = CommonShortcuts.get_factor_zb(Eg, deltaSO)   # independent of S and Ep, but temperature-dependent

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
        cSchroedinger = CommonShortcuts.hbar**2 / 2 / CommonShortcuts.electron_mass
        new_L = db_L + cSchroedinger * (new_Ep - db_Ep) / Eg
        new_N = db_N + cSchroedinger * (new_Ep - db_Ep) / Eg

        return new_S, new_Ep, new_L, new_N

    @staticmethod
    def shift_DKK_as_nnp(DKK_parameter, Eg_T, old_Ep, new_Ep):
        """ 
        Shift the 8-band DKK parameters as in nn++/nn3.
        
        Return
        ------
            In units of hbar^2 / 2m_0
        """
        return DKK_parameter + (new_Ep - old_Ep) / Eg_T

    @staticmethod
    def shift_DKK_properly(DKK_parameter, Eg_0K, Eg_T, old_Ep, new_Ep):
        """ 
        Shift the 8-band DKK parameters considering the temperature-dependence of the bandgap.
        This is appropriate IF 6-band DKK parameters are independent of temperature, which might not be the case.

        In NEGF++ so far, we have not obtained smooth wavefunctions using this shift.
        
        Return
        ------
            In units of hbar^2 / 2m_0
        """
        return DKK_parameter - old_Ep / Eg_0K + new_Ep / Eg_T

    @staticmethod
    def rescale_Ep_and_get_S(
            old_Ep, 
            old_S, 
            old_L, 
            old_N, 
            rescaleEpTo, 
            Eg, 
            deltaSO
            ):
        """
        Rescale Ep to given value, and adjust S to preserve the electron effective mass:
        (*) old_S + old_Ep * factor = new_S + rescaleEpTo * factor
        """
        factor = CommonShortcuts.get_factor_zb(Eg, deltaSO)   # independent of S and Ep, but temperature-dependent
        new_S = old_S + (old_Ep - rescaleEpTo) * factor

        # L' and N' get modified by the change of Ep
        # This keeps the kp6 values at T=0K intact.
        cSchroedinger = CommonShortcuts.hbar**2 / 2 / CommonShortcuts.electron_mass
        new_L = old_L + cSchroedinger * (rescaleEpTo - old_Ep) / Eg
        new_N = old_N + cSchroedinger * (rescaleEpTo - old_Ep) / Eg
        return new_S, new_L, new_N

    @staticmethod
    def get_8kp_from_6kp_NEGF(mass, rescaleSTo, Eg_0K, Eg_finiteTemp, deltaSO, L_6kp, N_6kp):
        """
        Imitate NEGF implementation.
        1. Calculate Ep by rescaling at T=0
        2. Calculate L', M, N', S from this Ep but using Eg at nonzero T
        """
        Ep = CommonShortcuts.Ep_from_mass_and_S(mass, rescaleSTo, Eg_0K, deltaSO)
        print(f"Calculated Ep from mass, S, and Eg(0): {Ep}")

        correction = Ep / Eg_finiteTemp   # L, N in the database are in units of hbar^2/2m0
        Lprime = L_6kp + correction
        Nprime = N_6kp + correction
        
        factor = CommonShortcuts.get_factor_zb(Eg_finiteTemp, deltaSO)   # independent of S and Ep, but temperature-dependent
        new_S = 1./mass - Ep*factor

        return Lprime, Nprime, new_S
    
    @staticmethod
    def get_ratio_inplaneK_over_KaneMomentum(inplaneK_in_inverseNm, Kane_energy_in_eV):
        """
        momentum 
        = m_0 / (i hbar) * Kane_P 
        = m_0 / (i hbar) * (hbar * sqrt(Kane_energy / 2m_0))
        = -i * sqrt(m_0 * Kane_energy / 2)
        """
        Kane_energy_in_J = Kane_energy_in_eV * CommonShortcuts.scale_eV_to_J
        Kane_momentum_kgmPerSec = np.sqrt(CommonShortcuts.electron_mass * Kane_energy_in_J / 2.)
        return CommonShortcuts.hbar * (inplaneK_in_inverseNm * CommonShortcuts.scale1ToNano) / Kane_momentum_kgmPerSec



    # -------------------------------------------------------
    # Simulation preprocessing
    #
    # We make methods static because:
    # - these utility functions do not depend on the class state but makes sense that they belong to the class
    # - we want to make this method available without instantiation of an object.
    # -------------------------------------------------------
    @staticmethod
    def detect_software(folder_path, filename):
        """
        DEPRECATED: Use get_shortcut() instead.

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
        product_name : str
            nextnano solver
        product_name_short : str
            shorthand of nextnano solver
        extension : str
            file extension
        """

        extension = PathHandler.separate_extension(filename)[1]
        if extension == '':
            raise ValueError('Please specify input file with extension (.in, .xml or .negf)')

        InputPath = os.path.join(folder_path, filename)
        try:
            with open(InputPath,'r') as file:
                for line in file:
                    if 'simulation-flow-control' in line:
                        product_name = 'nextnano3'
                        product_name_short = '_nn3'
                        break
                    elif 'run{' in line:
                        product_name = 'nextnano++'
                        product_name_short = '_nnp'
                        break
                    elif ('<nextnano.QCL' in line) or ('<nextnano.NEGF' in line) or ('nextnano.NEGF{' in line):
                        product_name = 'nextnano.NEGF'
                        product_name_short = '_nnNEGF'
                        break
                    elif '<nextnano.MSB' in line or 'nextnano.MSB{' in line:
                        product_name = 'nextnano.MSB'
                        product_name_short = '_nnMSB'
        except FileNotFoundError as e:
            raise Exception(f'Input file {InputPath} not found!') from e

        if not product_name:   # if the variable is empty
            raise NextnanoInputFileError('Software cannot be detected! Please check your input file.')
        else:
            logging.info(f'Software detected: {product_name}')

        return product_name, product_name_short, extension


    @staticmethod
    def detect_software_new(inputfile):
        """
        DEPRECATED: Use get_shortcut() instead.

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
                        product_name = 'nextnano3'
                        extension = '.in'
                        break
                    elif 'run{' in line:
                        product_name = 'nextnano++'
                        extension = '.in'
                        break
                    elif '<nextnano.NEGF' in line:
                        product_name = 'nextnano.NEGF'
                        extension = '.xml'
                        break
                    elif 'nextnano.NEGF{' in line:
                        product_name = 'nextnano.NEGF++'
                        extension = '.negf'
                    elif '<nextnano.MSB' in line:
                        product_name = 'nextnano.MSB'
                        extension = '.xml'
                    elif 'nextnano.MSB{' in line:
                        product_name = 'nextnano.MSB'
                        extension = '.negf'
        except FileNotFoundError as e:
            raise Exception(f'Input file {inputfile.fullpath} not found!') from e

        if not product_name:   # if the variable is empty
            raise NextnanoInputFileError('Software cannot be detected! Please check your input file.')
        else:
            logging.info(f'Software detected: {product_name}')

        return product_name, extension
    

    @staticmethod
    def get_shortcut(inputfile, loglevel=logging.INFO):
        """
        Detect software from nextnanopy.InputFile() object and returns corresponding shortcut object.

        Parameters
        ----------
        inputfile : nextnanopy.InputFile object

        Returns
        -------
        object of the class nnShortcuts.nnp_shortcuts / nnShortcuts.nn3_shortcuts / nnShortcuts.NEGF_shortcuts

        """
        try:
            with open(inputfile.fullpath, 'r') as file:
                for line in file:
                    if 'simulation-flow-control' in line:
                        from nnShortcuts.nn3_shortcuts import nn3Shortcuts
                        return nn3Shortcuts(loglevel=loglevel)
                    elif 'run{' in line:
                        from nnShortcuts.nnp_shortcuts import nnpShortcuts
                        return nnpShortcuts(loglevel=loglevel)
                    elif '<nextnano.NEGF' in line:
                        from nnShortcuts.NEGF_shortcuts import NEGFShortcuts
                        return NEGFShortcuts(True, loglevel=loglevel)
                    elif 'nextnano.NEGF{' in line:
                        from nnShortcuts.NEGF_shortcuts import NEGFShortcuts
                        return NEGFShortcuts(False, loglevel=loglevel)
                    elif '<nextnano.MSB' in line or 'nextnano.MSB{' in line:
                        raise NotImplementedError("MSB shortcuts are not implemented")
                raise NextnanoInputFileError('Software cannot be detected! Please check your input file.')
        
        except FileNotFoundError as e:
            raise Exception(f'Input file {inputfile.fullpath} not found!') from e


    @staticmethod
    def prepare_InputFile(
            folderPath, 
            originalFilename, 
            modifiedParamString=None, 
            newValue=0, 
            filename_appendix=''
            ):
        """
        Modify parameter in the input file, append specified string to the file name, save the file.

        RETURN:
            new file name
            modified nextnanopy.InputFile object
        """
        InputPath     = os.path.join(folderPath, originalFilename)
        input_file    = nn.InputFile(InputPath)

        if modifiedParamString is None:
            logging.info('Using the default parameters in the input file...\n')
            return originalFilename, input_file

        input_file.set_variable(modifiedParamString, value=newValue)
        name = input_file.get_variable(modifiedParamString).name
        value = input_file.get_variable(modifiedParamString).value
        logging.info(f'Using modified input parameter:\t${name} = {value}')

        filename_no_extension, extension = PathHandler.separate_extension(originalFilename)
        if extension == '':
            raise ValueError('Input file name must include file extension!')
        newFilename = filename_no_extension + filename_appendix + extension
        logging.info(f'Saving input file as:\t{newFilename}\n')
        input_file.save(os.path.join(folderPath, newFilename), overwrite=True)   # update input file name

        return newFilename, input_file


    # -------------------------------------------------------
    # Access to output data
    # -------------------------------------------------------
    def compose_sweep_output_folder_path(self, filename, *args):
        """
        Get the output folder path generated by nextnanopy.sweep.execute_sweep().

        Parameters
        ----------
        filename : str
            input file name (may include absolute/relative paths)
        *args : str
            SweepVariableString1, SweepVariableString2, ...

        Returns
        -------
        output_folder_path : str
            sweep output folder path

        """
        filename_no_extension = PathHandler.separate_extension(filename)[0]
        output_root_dir = nn.config.get(self.product_name, 'outputdirectory')

        if len(args) == 0: 
            return output_root_dir

        output_folder_path = os.path.join(output_root_dir, filename_no_extension + '_sweep')
        for sweepVar in args:
            if not isinstance(sweepVar, str):
                raise TypeError(f'Argument {sweepVar} must be a string, but is {type(sweepVar)}')
            output_folder_path += '__' + sweepVar

        return output_folder_path


    def get_DataFile(self, keywords, name, exclude_keywords=None):
        """
        Get single nextnanopy.DataFile of output data in the directory matching name with the given string keyword.

        Parameters
        ----------
        keywords : str or list of str
            Find output data file with the names containing single keyword or multiple keywords (AND search)
        name : str
            input file name (= output subfolder name). May contain extensions and/or fullpath.
        exclude_keywords : str or list of str, optional
            Files containing these keywords in the file name are excluded from search.

        """
        output_subfolder = self.__compose_output_subfolder_path(name)
        return self.get_DataFile_in_folder(keywords, output_subfolder, exclude_keywords=exclude_keywords)


    def get_DataFile_in_folder(self, keywords, folder_path, exclude_keywords=None, exclude_folders=None, allow_folder_name_suffix=False):
        """
        Get single nextnanopy.DataFile of output data with the given string keyword(s) in the specified folder.

        Parameters
        ----------
        keywords : str or list of str
            Find output data file with the names containing single keyword or multiple keywords (AND search)
        folder_path : str
            absolute path of output folder in which the datafile should be sought
        exclude_keywords : str or list of str, optional
            Files containing these keywords in the file name are excluded from search.
        exclude_folders : str or list of str, optional
            Files with these folder names in their paths are excluded from search.
        allow_folder_name_suffix : bool, optional
            If True, search for the folder name starting with 'folder_path'.
            If False, search for exact match of the folder name with 'folder_path'.

        Returns
        -------
        nextnanopy.DataFile object of the simulation data

        """
        folder_path = PathHandler.expect_single_folder_to_exist(folder_path, allow_folder_name_suffix)

        # if only one keyword is provided, make a list with single element to simplify code
        if isinstance(keywords, str):
            keywords = [keywords]
        elif not isinstance(keywords, list):
            raise TypeError(f"Argument 'keywords' must be either str or list, but is {type(keywords)}")
        if isinstance(exclude_keywords, str):
            exclude_keywords = [exclude_keywords]
        elif exclude_keywords is not None and not isinstance(exclude_keywords, list):
            raise TypeError(f"Argument 'exclude_keywords' must be either str or list, but is {type(exclude_keywords)}")
        if isinstance(exclude_folders, str):
            exclude_folders = [exclude_folders]
        elif exclude_folders is not None and not isinstance(exclude_folders, list):
            raise TypeError(f"Argument 'exclude_subfolders' must be either str or list, but is {type(exclude_folders)}")

        if exclude_keywords is None:
            if exclude_folders is None:
                message = " '" + "', '".join(keywords) + "'"
            else:
                message = " '" + "', '".join(keywords) + "', excluding keyword(s) '" + "', '".join(exclude_folders) + "'"
        else:
            if exclude_folders is None:
                message = " '" + "', '".join(keywords) + "', excluding keyword(s) '" + "', '".join(exclude_keywords) + "'"
            else:
                message = " '" + "', '".join(keywords) + "', excluding keyword(s) '" + "', '".join(exclude_keywords + exclude_folders) + "'"

        logging.info(f'Searching for output data {message}...')

        # Search output data using nn.DataFolder.find(). If multiple keywords are provided, find the intersection of files found with each keyword.
        list_of_sets = [set(nn.DataFolder(folder_path).find(keyword, deep=True)) for keyword in keywords]
        candidates = list_of_sets[0]
        for s in list_of_sets:
            candidates = s.intersection(candidates)
        list_of_files = list(candidates)

        def should_be_excluded(filepath):
            folder, filename = os.path.split(filepath)
            if exclude_folders is not None:
                contain_folder_name = any(key in folder for key in exclude_folders)
            else:
                contain_folder_name = False
            if exclude_keywords is not None:
                contain_keyword = any(key in filename for key in exclude_keywords)
            else:
                contain_keyword = False
            return (contain_keyword or contain_folder_name)

        if (exclude_keywords is not None) or (exclude_folders is not None):
            list_of_files = [file for file in list_of_files if not should_be_excluded(file)]


        # validate the search result
        if len(list_of_files) == 0:
            raise FileNotFoundError(f"No output file found in the folder {folder_path}")
        elif len(list_of_files) == 1:
            file = list_of_files[0]
        else:
            logging.warning(f"More than one output files found!")
            choice = PathHandler.ask_user_to_choose_one(list_of_files)
            file = list_of_files[choice]

        logging.debug(f"Found:\n{file}")

        try:
            return nn.DataFile(file, product=self.product_name)
        except NotImplementedError as e:
            raise Exception(f'Nextnanopy does not support datafile for {file}') from e


    def __compose_output_subfolder_path(self, name):
        outputFolder = nn.config.get(self.product_name, 'outputdirectory')
        filename_no_extension = PathHandler.separate_extension(name)[0]
        return os.path.join(outputFolder, filename_no_extension)


    def get_DataFiles(self, keywords, name, exclude_keywords=None):
        """
        Get multiple nextnanopy.DataFiles of output data with the given string keyword(s).

        Parameters
        ----------
        keywords : str or list of str
            Find output data file with the names containing single keyword or multiple keywords (AND search).
        name : str
            input file name (= output subfolder name) without folder paths. May contain extension '.in' or '.xml'.
        exclude_keywords : str or list of str, optional
            Files containing these keywords in the file name are excluded from search.

        """
        output_subfolder = self.__compose_output_subfolder_path(name)
        return self.get_DataFiles_in_folder(keywords, output_subfolder, exclude_keywords=exclude_keywords)


    def get_DataFiles_in_folder(self, keywords, folder_path, exclude_keywords=None, allow_folder_name_suffix=False):
        """
        Get multiple nextnanopy.DataFiles of output data with the given string keyword(s) in the specified folder.

        Parameters
        ----------
        keywords : str or list of str
            Find output data file with the names containing single keyword or multiple keywords (AND search)
        folder_path : str
            absolute path of output folder in which the datafile should be sought
        exclude_keywords : str or list of str, optional
            Files containing these keywords in the file name are excluded from search.
        allow_folder_name_suffix : bool, optional
            If True, search for the folder name starting with 'folder_path'.
            If False, search for exact match of the folder name with 'folder_path'.

        Returns
        -------
        list of nextnanopy.DataFile objects of the simulation data

        """
        # validate the path
        folder_path = PathHandler.expect_single_folder_to_exist(folder_path, allow_folder_name_suffix)

        # if only one keyword is provided, make a list with single element to simplify code
        if isinstance(keywords, str):
            keywords = [keywords]
        elif not isinstance(keywords, list):
            raise TypeError(f"Argument 'keywords' must be either str or list, but is {type(keywords)}")
        if isinstance(exclude_keywords, str):
            exclude_keywords = [exclude_keywords]
        elif not isinstance(exclude_keywords, list) and exclude_keywords is not None:
            raise TypeError(f"Argument 'exclude_keywords' must be either str or list, but is {type(exclude_keywords)}")

        if exclude_keywords is None:
            message = "with keyword(s) '" + "', '".join(keywords) + "'"
        else:
            message = "with keyword(s) '" + "', '".join(keywords) + "', excluding '" + "', '".join(exclude_keywords) + "'"

        logging.info(f'Searching for output data {message}...')

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
            raise FileNotFoundError(f"No output file found in the folder {folder_path}")
        elif len(list_of_files) == 1:
            warnings.warn("get_DataFiles_in_folder(): Only one output file found!", category=RuntimeWarning)

        logging.debug(f"Found:\n{list_of_files}")

        try:
            datafiles = [nn.DataFile(file, product=self.product_name) for file in list_of_files]
        except NotImplementedError as e:
            raise Exception('Nextnanopy does not support datafile') from e

        return datafiles


    def get_DataFile_probabilities_with_name(self, name, bias=None):
        """
        Get single nextnanopy.DataFile of probability_shift data in the folder of specified name.

        INPUT:
            name : string
                input file name (= output subfolder name). May contain extensions and fullpath
            bias : real, optional
                If not None, that bias is used to search for the energy eigenstates output folder. 
                If None, output is sought in the Init folder.
        
        RETURN:
            dictionary { quantum model key: corresponding list of nn.DataFile() objects for probability_shift }
        """
        output_subfolder = self.__compose_output_subfolder_path(name)
        return self.get_DataFile_probabilities_in_folder(output_subfolder, bias=bias)
        

    def get_DataFile_probabilities_in_folder(self):
        raise NotImplementedError("There is no common implementation")


    def get_DataFile_amplitudesK0_in_folder(self):
        raise NotImplementedError("There is no common implementation")
    

    def __get_num_evs(self, probability_dict):
        """ number of eigenvalues for each quantum model """
        num_evs = dict()
        for model, datafiles in probability_dict.items():
            if isinstance(datafiles, list):
                if len(datafiles) == 0:
                    warnings.warn(f"No k-points were calculated for model = {model}")
                    num_evs[model] = 0
                else:
                    df = datafiles[0]
                    num_evs[model] = sum(1 for var in df.variables if (self.wavefunction_name in var.name))   # this conditional counting is necessary because probability output may contain also eigenvalues and/or bandedges.
                    logging.debug(f"Number of eigenvalues for {model}: {num_evs[model]}")
            elif isinstance(datafiles, nn.DataFile):
                num_evs[model] = sum(1 for var in datafiles.variables if (self.wavefunction_name in var.name))   # this conditional counting is necessary because probability output may contain also eigenvalues and/or bandedges.
        return num_evs


    def get_states_to_be_plotted(self,
            datafiles_probability_dict, 
            states_range_dict=None, 
            states_list_dict=None
            ):
        """
        Create dictionaries of
            1) eigenstate indices to be plotted for each quantum model
            2) number of eigenstates for each quantum model

        INPUT:
            datafiles_probability_dict      dict generated by get_DataFile_probabilities() method
            states_range_dict               range of state indices to be plotted for each quantum model. dict of the form { 'quantum model': [start index, last index] }
            states_list_dict                list of state indices to be plotted for each quantum model. Alternatively, strings 'lowestElectron' and 'highestHole' are accepted and state indices are set automatically.

        RETURN:
            states_toBePlotted              for each quantum model, array of eigenstate indices to be plotted in the figure. Has the form:
                                            { 'quantum model': list of values }
            num_evs                         for each quantum model, number of all eigenstates existing in the output data

        NOTE:
            state index is base 0 (differ from nextnano++ output), state No is base 1 (identical to nextnano++ output)
        """
        # validate input
        if states_range_dict is not None and not isinstance(states_range_dict, dict):
            raise TypeError(f"Argument 'states_range_dict' must be a dict, but is {type(states_range_dict)}")
        if states_list_dict is not None and not isinstance(states_list_dict, dict):
            raise TypeError(f"Argument 'states_list_dict' must be a dict, but is {type(states_list_dict)}")
        if (states_range_dict is not None) and (states_list_dict is not None):
            raise ValueError("Only one of 'states_range_dict' or 'states_list_dict' is allowed as an argument")

        # get number of eigenvalues
        num_evs = self.__get_num_evs(datafiles_probability_dict)

        # TODO: nn3 has two output files '_el' and '_hl' also in 8kp calculation
        states_toBePlotted = dict.fromkeys(datafiles_probability_dict.keys())

        # determine index of states to be plotted
        if states_list_dict is None:
            if states_range_dict is None:
                for model in datafiles_probability_dict:
                    states_toBePlotted[model] = list(np.arange(0, num_evs[model]))   # by default, plot all the eigenstates
            else:
                # from states_range_dict
                for model in datafiles_probability_dict:
                    if model not in states_range_dict:
                        states_toBePlotted[model] = list(np.arange(0, num_evs[model]))   # by default, plot all the eigenstates
                    else:
                        startIdx = states_range_dict[model][0] - 1
                        stopIdx  = states_range_dict[model][1] - 1
                        states_toBePlotted[model] = list(np.arange(startIdx, stopIdx+1, 1))   # np.arange(min, max) stops one step before the max
        else:
            # from states_list_dict
            first_element = list(datafiles_probability_dict.values())[0][0]
            filepath = first_element.fullpath   # take arbitrary quantum model because all of them are in the same folder bias_*/Quantum
            outfolder = os.path.split(filepath)[0]
            for model in datafiles_probability_dict:
                if model not in states_list_dict:
                    states_toBePlotted[model] = list(np.arange(0, num_evs[model]))   # by default, plot all the eigenstates
                else:
                    states_toBePlotted[model] = list()
                    for stateNo in states_list_dict[model]:
                        if stateNo == 'highestHole':
                            if model != 'kp8' and model not in self.model_names_valence:
                                raise ValueError(f"Quantum model '{model}' does not contain hole states.")
                            
                            # TODO: nn3 has two output files '_el' and '_hl' also in 8kp calculation
                            states_toBePlotted[model].append(self.find_highest_valence_state_atK0(outfolder, threshold=0.5))
                                
                        elif stateNo == 'lowestElectron':
                            if model != 'kp8' and model not in self.model_names_conduction:
                                raise ValueError(f"Quantum model '{model}' does not contain electron states.")

                            states_toBePlotted[model].append(self.find_lowest_conduction_state_atK0(outfolder, threshold=0.5))
                            
                        elif stateNo == 'occupied':
                            if self.product_name == 'nextnano.NEGF':
                                raise NotImplementedError("Plotting only occupied states not yet implemented for NEGF")
                            if 'cutoff_occupation' not in states_list_dict.keys():
                                raise ValueError("cutoff_occupation must be specified in 'states_list_dict'")

                            # WARNING: state selection based on k||=0 occupation
                            df = self.get_DataFile_in_folder(['occupation', model], outfolder)
                            try:
                                cutoff_occupation = np.double(states_list_dict['cutoff_occupation'])
                            except ValueError as e:
                                raise Exception("cutoff_occupation must be a real number!") from e
                            if cutoff_occupation < 0: 
                                raise ValueError("cutoff_occupation must be positive!")

                            states_toBePlotted[model] += [int(stateNo) - 1 for stateNo, occupation in zip(df.coords['no.'].value, df.variables['SignedOccupation'].value) if occupation >= cutoff_occupation]
                        elif isinstance(stateNo, int):
                            if stateNo > num_evs[model]: 
                                raise ValueError("State index greater than number of eigenvalues calculated!")
                            states_toBePlotted[model].append(stateNo - 1)   
        logging.debug(f"states_toBePlotted (index base 0): {states_toBePlotted}")
        return states_toBePlotted, num_evs

    def find_highest_valence_state_atK0(self):
        raise NotImplementedError("There is no common implementation")

    def find_lowest_conduction_state_atK0(self):
        raise NotImplementedError("There is no common implementation")
    
    def calculate_overlap(self):
        raise NotImplementedError("There is no common implementation")
    
    def get_transition_energy(self):
        raise NotImplementedError("There is no common implementation")
    
    def get_HH1_LH1_energy_difference(self):
        raise NotImplementedError("There is no common implementation")

    def get_HH1_HH2_energy_difference(self):
        raise NotImplementedError("There is no common implementation")
    
    def get_absorption_at_transition_energy(self):
        raise NotImplementedError("There is no common implementation")

    # -------------------------------------------------------
    # Data postprocessing
    #
    # We make methods static because:
    # - these utility functions do not depend on the class state but makes sense that they belong to the class
    # - we want to make this method available without instantiation of an object.
    # -------------------------------------------------------
    @staticmethod
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


    @staticmethod
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
        if np.ndim(arr) != 1: 
            raise ValueError("Array must be one-dimensional!")

        num_gridPoints = len(x_grid)

        # input validation
        if len(arr) != num_gridPoints:  # 'averaged = yes' 'boxed = yes' may lead to inconsistent number of grid points
            raise ValueError(f'Array size {len(arr)} does not match the number of real space grid points {num_gridPoints}')
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


    # -------------------------------------------------------
    # Plotting
    #
    # We make some methods static because:
    # - these utility functions do not depend on the class state but makes sense that they belong to the class
    # - we want to make this method available without instantiation of an object.
    # -------------------------------------------------------
    def plot_IV(self):
        raise NotImplementedError("There is no common implementation")

    def plot_DOS(self):
        raise NotImplementedError("There is no common implementation")

    def plot_carrier_density(self):
        raise NotImplementedError("There is no common implementation")
    
    def plot_current_density(self):
        raise NotImplementedError("There is no common implementation")
    
    def plot_gain(self):
        raise NotImplementedError("There is no common implementation")


    @staticmethod
    def get_rowColumn_for_display(num_elements):
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
        # logging.debug(num_rows, num_columns)

        if n < 3: return num_rows, num_columns

        while (np.double(num_columns) / np.double(num_rows) < 0.7):   # try to make it as square as possible
            # logging.debug('n=', n)
            k = math.floor(math.sqrt(n))

            while not n % k == 0: k -= 1
            # logging.debug('k=', k)
            num_rows    = int(n / k)
            num_columns = int(k)
            # logging.debug(num_rows, num_columns)
            n += 1

        return num_rows, num_columns


    @staticmethod
    def generate_colorscale(colormap, minValue, maxValue):
        """
        Generate a color scale with given colormap and range of values.
        """
        return plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=minValue, vmax=maxValue) )



    @staticmethod
    def mask_part_of_array(arr, method='flat', tolerance=1e-4, cut_range=[]):
        """
        Mask some elements in an array to plot limited part of data.

        INPUT:
            arr           data array to be masked
            method        specify mask method. 'flat' masks flat part of the data, while 'range' masks the part specified by the index range.
            tolerance     for 'flat' mask method
            cut_range     list: range of index to define indies to be masked

        RETURN:
            masked array

        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Array must be numpy.ndarray, but is {type(arr)}")

        arr_size = len(arr)
        new_arr = np.ma.array(arr, mask = [0 for i in range(arr_size)])   # non-masked np.ma.array with given data arr

        if method == 'flat':  # mask data points where the values are almost flat
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



        if method == 'range':
            if cut_range == []: raise ValueError('Specify the range to cut!')

            cut_indices = np.arange(cut_range[0], cut_range[1], 1)
            for i in cut_indices:
                new_arr.mask[i] = True

        return new_arr


    @staticmethod
    def set_plot_labels(ax, x_label, y_label, title):
        """
        Set the labels and optimize their sizes (matplotlib default of font size is too small!)
        """
        ax.set_xlabel(x_label, fontsize=CommonShortcuts.labelsize_default)  # r with superscript works
        ax.set_ylabel(y_label, fontsize=CommonShortcuts.labelsize_default)
        ax.set_title(title, fontsize=CommonShortcuts.labelsize_default)
        ax.tick_params(axis='x', labelsize=CommonShortcuts.ticksize_default)
        ax.tick_params(axis='y', labelsize=CommonShortcuts.ticksize_default)
        return ax


    @staticmethod
    def adjust_plot_title(originalTitle):
        """
        If the title is too long for display, omit the intermediate letters
        """
        title = PathHandler.separate_extension(originalTitle)[0]   # remove extension if present

        if len(title) > 25:
            beginning = title[:10]
            last  = title[-10:]
            title = beginning + ' ... ' + last

        return title


    @staticmethod
    def draw_inplane_dispersion(
            ax, 
            kPoints, 
            dispersions, 
            states_toBePlotted, 
            flip_xAxis, 
            set_ylabel, 
            labelsize, 
            titlesize,
            ticksize, 
            annotatesize,
            markersize=3,
            Emin=None,
            Emax=None,
            title='Inplane dispersion', 
            lattice_temperature=None, 
            show_kBT_at_energy=None
            ):
        """
        flip_xAxis : bool
            if True, invert the x axis.
        labelsize : float
        """
        CVD_aware = True
        ax.set_xlabel(CommonShortcuts.axis_label_inplane_k, fontsize=labelsize)
        kmargin = 0.05 * (np.amax(kPoints) - np.amin(kPoints))
        kmin = np.amin(kPoints) - kmargin
        kmax = np.amax(kPoints) + kmargin
        ax.set_xlim(kmin, kmax)
        ax.set_ylim(Emin, Emax)

        if CVD_aware:
            color = 'black'
        else:
            color = 'orange'
        for index in states_toBePlotted:
            ax.plot(kPoints, dispersions[index, ], linestyle='', marker='.', markersize=markersize, label=f'Band_{index+1}', color=color) #linewidth=0.7
        ax.tick_params(labelsize=ticksize)
        
        if flip_xAxis:
            ax.invert_xaxis()
            ax.grid(axis='y')
        if set_ylabel:
            ax.set_ylabel(CommonShortcuts.axis_label_energy, fontsize=labelsize)
        if lattice_temperature is not None:
            kBT = CommonShortcuts.Boltzmann * lattice_temperature * CommonShortcuts.scale_J_to_eV
            
            ymin, ymax = ax.get_ylim()
            relative_position_k = 0.6
            x = (1-relative_position_k)*kmin + relative_position_k*kmax
            if show_kBT_at_energy is None:
                relative_position_y = 0.4
                y_from = (1-relative_position_y)*ymin + relative_position_y*ymax
            else:
                y_from = show_kBT_at_energy
            y_to = y_from + kBT

            relative_position_k_text = relative_position_k + 0.35
            x_text = (1-relative_position_k_text)*kmin + relative_position_k_text*kmax
            ax.vlines(x, y_from, y_to, colors='black')
            ax.annotate("$k_\mathrm{B}T$", xy=(x, (y_to + y_from)/2.0), xytext=(x_text, (y_to + y_from)/2.0), fontsize=annotatesize)
        # ax.legend(labels=states_toBePlotted+1, bbox_to_anchor=(1.05, 1))
        ax.set_title(title, fontsize=titlesize)


    def draw_bandedges(self, ax, plot_title, model, x, CBBandedge, want_valence_band, HHBandedge, LHBandedge, shorten_data_label):
        CommonShortcuts.set_plot_labels(ax, CommonShortcuts.axis_label_position, CommonShortcuts.axis_label_energy, plot_title)

        CVD_aware = True
        color_CB, color_HH, color_LH = self.default_colors.get_linecolor_bandedges(CVD_aware, False)
        linestyle_CB, linestyle_HH, linestyle_LH = CommonShortcuts.get_linestyle_bandedges()
        lw_CB, lw_HH, lw_LH = CommonShortcuts.get_linewidths_bandedges(0.6)

        if model == self.band_names['Gamma'] or model == self.band_names['kp8']:
            if shorten_data_label:
                CBLabel = 'CB'
            else:
                CBLabel = 'Conduction band'
            if len(CBBandedge) == 0:
                raise ValueError("CBBandedge is empty!")
            ax.plot(x, CBBandedge, label=CBLabel, linewidth=lw_CB, color=color_CB, linestyle=linestyle_CB)
        if want_valence_band:
            if model == self.band_names['HH'] or model == self.band_names['kp6'] or model == self.band_names['kp8']:
                if shorten_data_label:
                    HHLabel = 'HH'
                else:
                    HHLabel = 'Heavy hole'
                if len(HHBandedge) == 0:
                    raise ValueError("HHBandedge is empty!")
                ax.plot(x, HHBandedge, label=HHLabel, linewidth=lw_HH, color=color_HH, linestyle=linestyle_HH)
            if model == self.band_names['LH'] or model == self.band_names['kp6'] or model == self.band_names['kp8']:
                if shorten_data_label:
                    LHLabel = 'HH'
                else:
                    LHLabel = 'Light hole'
                ax.plot(x, LHBandedge, label=LHLabel, linewidth=lw_LH, color=color_LH, linestyle=linestyle_LH)
            # if model == 'SO' or model == 'kp6' or model == 'kp8':
            #     ax.plot(x, SOBandedge, label='split-off hole', linewidth=0.6, color=self.default_colors.bands['SO'])
            # if model == 'LH' or model == 'kp6' or model == 'kp8':
            #     ax.plot(x, VBTop, label='VB top without strain', linewidth=0.6, color=self.default_colors.bands['LH'])


    def plot_inplaneK(self):
        raise NotImplementedError("There is no common implementation")


    @staticmethod
    def get_linestyle_bandedges():
        linestyle_CB = (0, (3, 3)) # (offset, (on_off_seq)) # more distinguishable than 'dashed' from 'solid'
        linestyle_HH = 'solid'
        linestyle_LH = 'solid' # 'dotted' is indistinguishable from 'solid' in small figures --> distinguish by thickness
        return linestyle_CB, linestyle_HH, linestyle_LH


    @staticmethod
    def get_linewidths_bandedges(linewidth):
        """ 
        Returns magnificed linewidths for CB, HH, and LH edges.
        'dotted' is indistinguishable from 'solid' in small figures --> distinguish by thickness
        """
        return linewidth, 3*linewidth, linewidth


    def draw_probabilities(self, ax, state_indices, x, psiSquared, model, kIndex, show_state_index, color_by_fraction_of, scalarmappable, compositions):
        if model != 'kp8' and color_by_fraction_of is not None:
            warnings.warn(f"Option 'color_by_fraction_of' is only effective in 8kp simulations, but {model} results are being used")
        if model == 'kp8' and color_by_fraction_of is None:
            color_by_fraction_of = 'conduction_band'  # default
        skip_annotation = False
        for cnt, stateIndex in enumerate(state_indices):
            if model == 'kp8':
                if scalarmappable is None:
                    raise ValueError("scalarmappable is None! Cannot draw probabilities with colors")
                if compositions is None:
                    raise ValueError("Spinor compositions are None!")
                # color according to spinor compositions
                if color_by_fraction_of == 'conduction_band':
                    plot_color = scalarmappable.to_rgba(compositions['kp8'][stateIndex, kIndex, 0])
                elif color_by_fraction_of == 'heavy_hole':
                    plot_color = scalarmappable.to_rgba(compositions['kp8'][stateIndex, kIndex, 1])
            else:
                # color according to the quantum model that yielded the solution
                plot_color = self.default_colors.bands[model]
            ax.plot(x, psiSquared[model][cnt][kIndex], color=plot_color, lw=2)

            if show_state_index:
                xmax, ymax = ScientificPlotter.get_maximum_points(psiSquared[model][cnt][kIndex], x)
                if skip_annotation:   # if annotation was skipped in the previous iteration, annotate
                    # ax.annotate(f'n={stateIndex},{stateIndex+1}', xy=(xmax, ymax), xytext=(xmax-0.05*simLength, ymax+0.07))
                    ax.annotate(f'{stateIndex},{stateIndex+1}', xy=(xmax, ymax), xytext=(xmax, ymax+0.07))
                    skip_annotation = False   # wavefunction degeneracy is atmost 2
                elif cnt < len(state_indices)-1:  # if not the last state
                    xmax_next, ymax_next = ScientificPlotter.get_maximum_points(psiSquared[model][cnt][kIndex], x)  # TODO: should 'cnt' be 'cnt+1'?
                    if abs(xmax_next - xmax) < 1.0 and abs(ymax_next - ymax) < 1e-1:
                        skip_annotation = True
                    else:
                        skip_annotation = False
                        # ax.annotate(f'n={stateIndex+1}', xy=(xmax, ymax), xytext=(xmax-0.05*simLength, ymax+0.07))
                        ax.annotate(f'{stateIndex+1}', xy=(xmax, ymax), xytext=(xmax, ymax+0.07))
                else:
                    # ax.annotate(f'n={stateIndex+1}', xy=(xmax, ymax), xytext=(xmax-0.05*simLength, ymax+0.07))
                    ax.annotate(f'{stateIndex+1}', xy=(xmax, ymax), xytext=(xmax, ymax+0.07))


############### plot probabilities ##################################
    def plot_probabilities(self,
                input_file,
                states_range_dict   = None,
                states_list_dict    = None,
                start_position      = None,
                end_position        = None,
                hide_tails          = False,
                only_k0             = True,
                show_spinor         = False,
                show_state_index    = False,
                color_by_fraction_of = '',
                plot_title          = '',
                verbose_title       = True,
                labelsize           = None,
                ticksize            = None,
            ):
        if start_position is None:
            start_position = CommonShortcuts.position_min
        if end_position is None:
            end_position = CommonShortcuts.position_max

        # load output data files
        datafiles_probability_dict = self.get_DataFile_probabilities_with_name(input_file.fullpath)
        output_subfolder = self.__compose_output_subfolder_path(input_file.fullpath)

        return self.plot_probabilities_core(
                output_subfolder,
                datafiles_probability_dict,
                states_range_dict,
                states_list_dict,
                start_position,
                end_position,
                hide_tails,
                only_k0,
                show_spinor,
                show_state_index,
                color_by_fraction_of,
                plot_title,
                verbose_title,
                labelsize,
                ticksize,
            )
    

    def plot_probabilities_by_folderpath(self,
                output_folder_path,
                shift_by_eigenvalue,
                states_range_dict   = None,
                states_list_dict    = None,
                start_position      = None,
                end_position        = None,
                hide_tails          = False,
                only_k0             = True,
                show_spinor         = False,
                show_state_index    = False,
                color_by_fraction_of = '',
                plot_title          = '',
                verbose_title       = True,
                labelsize           = None,
                ticksize            = None,
            ):
        """
        output_folder_path : str
            Location of simulation outputs.

        Returns
        -------
            matplotlib Figure object
        """
        if start_position is None:
            start_position = CommonShortcuts.position_min
        if end_position is None:
            end_position = CommonShortcuts.position_max
            
        # load output data files
        datafiles_probability_dict = self.get_DataFile_probabilities_in_folder(output_folder_path, shift_by_eigenvalue, bias=None)  # TODO: 'allow_folder_name_suffix' option not yet supported by derived classes 'nnpShortcuts' and 'nn3Shortcuts'
        
        return self.plot_probabilities_core(
                output_folder_path,
                datafiles_probability_dict,
                states_range_dict,
                states_list_dict,
                start_position,
                end_position,
                hide_tails,
                only_k0,
                show_spinor,
                show_state_index,
                color_by_fraction_of,
                plot_title,
                verbose_title,
                labelsize,
                ticksize,
            )
    

    def plot_probabilities_core(self, 
            output_folder_path,
            datafiles_probability_dict,
            states_range_dict,
            states_list_dict,
            start_position,
            end_position,
            hide_tails,
            only_k0,
            show_spinor,
            show_state_index,
            color_by_fraction_of,
            plot_title,
            verbose_title,
            labelsize,
            ticksize,
            ):
        """
        Plot probability distribution on top of bandedges.
        Properly distinguishes the results from the single band effective mass model, 6- and 8-band k.p models.
        If both electrons and holes have been simulated, also plot both probabilities in one figure.

        The probability distributions are coloured according to
        quantum model that yielded the solution (for single-band effective mass and kp6 models) or
        electron fraction (for kp8 model).

        Parameters
        ----------
        input_file : nextnanopy.InputFile object
            nextnano3 input file.
        states_range_dict : dict, optional
            range of state indices to be plotted for each quantum model. The default is None.
        states_list_dict : dict, optional
            list of state indices to be plotted for each quantum model.
            Alternatively, strings 'lowestElectron', 'highestHole' and 'occupied' are accepted. For 'occupied', another key 'cutoff_occupation' must be set in this dict.
            The default is None.
        start_position : real, optional
            left edge of the plotting region. The default is -10000.
        end_position : real, optional
            right edge of the plotting region. The default is 10000.
        hide_tails : bool, optional
            hide the probability tails. The default is False.
        only_k0 : bool, optional
            suppress the output of nonzero in-plane k states. The default is True.
        show_spinor : bool, optional
            plot pie chart of spinor composition for all eigenvalues and k points. The default is False.
        show_state_index : bool, optional
            indicate eigenstate indices on top of probability plot. The default is False.
        color_by_fraction_of : str, optional
            If 8-band k.p simulation, colour the probabilities by the spinor fraction of the specified band. The default is 'conduction_band'.
        plot_title : str, optional
            title of the probability plot. The default is ''.
        verbose_title : bool
            If True, print the quantum models used and in-plane k information in the plot title.
        labelsize : int, optional
            font size of xlabel, ylabel and colorbar label
        ticksize : int, optional
            font size of xtics, ytics and colorbar tics
        
        Returns
        -------
        matplotlib Figure object

        """
        if labelsize is None: labelsize = self.labelsize_default
        if ticksize is None: ticksize = self.ticksize_default
        if color_by_fraction_of is not None and color_by_fraction_of not in ['conduction_band', 'heavy_hole']:
            raise ValueError(f"color_by_fraction_of '{color_by_fraction_of}' is not supported")

        from matplotlib import colors
        from matplotlib.gridspec import GridSpec

        for model, datafiles in datafiles_probability_dict.items():
            if len(datafiles) == 0: 
                continue

            datafile_probability = datafiles[0]
            x_probability  = datafile_probability.coords[self.position_axis_key].value
        if not datafile_probability:
            raise NextnanoInputFileError('Probabilities are not output! Modify the input file.')


        # store data in arrays (independent of quantum models)
        datafile_bandedge = self.get_DataFile_in_folder([self.bandedge_filename], output_folder_path)
        x             = datafile_bandedge.coords[self.position_axis_key].value
        CBBandedge    = datafile_bandedge.variables[self.conduction_bandedge_key].value
        LHBandedge    = datafile_bandedge.variables[self.light_hole_bandedge_key].value
        HHBandedge    = datafile_bandedge.variables[self.heavy_hole_bandedge_key].value
        SOBandedge    = datafile_bandedge.variables[self.SO_hole_bandedge_key].value

        states_toBePlotted, num_evs = self.get_states_to_be_plotted(datafiles_probability_dict, states_range_dict=states_range_dict, states_list_dict=states_list_dict)


        # visualize the in-plane k point_maxts at which Schroedinger eq. has been solved
        if only_k0:
            num_kPoints = dict()
            for model in states_toBePlotted:
                num_kPoints[model] = 1
        else:
            inplaneK_dict = self.getKPointsData1D(input_file)
            self.plot_inplaneK(inplaneK_dict)
            num_kPoints = self.get_num_kPoints(inplaneK_dict)


        # dictionary containing quantum model keys and 2-dimensional list for each key that stores psi^2 for all (eigenstate, kIndex)
        psiSquared = dict.fromkeys(datafiles_probability_dict.keys())
        for model in states_toBePlotted:
            psiSquared[model] = [ [ 0 for kIndex in range(num_kPoints[model]) ] for stateIndex in range(num_evs[model]) ]  # stateIndex in states_toBePlotted[model] would give a list of the same size

        for model, dfs in datafiles_probability_dict.items():
            if len(dfs) == 0: 
                continue

            for cnt, stateIndex in enumerate(states_toBePlotted[model]):
                for kIndex in range(num_kPoints[model]):
                    psiSquared_oldgrid = dfs[kIndex].variables[self.wavefunction_name + f'^2_{stateIndex+1}'].value
                    psiSquared[model][cnt][kIndex] = CommonShortcuts.convert_grid(psiSquared_oldgrid, x_probability, x)   # grid interpolation needed because of 'output_bandedges{ averaged=no }'


        # chop off edges of the simulation region
        CBBandedge = CommonShortcuts.cutOff_edges1D(CBBandedge, x, start_position, end_position)
        HHBandedge = CommonShortcuts.cutOff_edges1D(HHBandedge, x, start_position, end_position)
        LHBandedge = CommonShortcuts.cutOff_edges1D(LHBandedge, x, start_position, end_position)
        SOBandedge = CommonShortcuts.cutOff_edges1D(SOBandedge, x, start_position, end_position)


        for model in states_toBePlotted:
            for cnt, stateIndex in enumerate(states_toBePlotted[model]):
                for kIndex in range(num_kPoints[model]):
                    psiSquared[model][cnt][kIndex] = CommonShortcuts.cutOff_edges1D(psiSquared[model][cnt][kIndex], x, start_position, end_position)   # chop off edges of the simulation region

        x = CommonShortcuts.cutOff_edges1D(x, x, start_position, end_position)
        # simLength = x[-1]-x[0]   # [nm]


        # mask psiSquared data where it is flat
        if hide_tails:
            for model in states_toBePlotted:
                for cnt, stateIndex in enumerate(states_toBePlotted[model]):
                    for kIndex in range(num_kPoints[model]):
                        psiSquared[model][cnt][kIndex] = CommonShortcuts.mask_part_of_array(psiSquared[model][cnt][kIndex], 'flat', 1e-3)


        if 'kp6' in datafiles_probability_dict.keys() or 'kp8' in datafiles_probability_dict.keys():
            # output data of spinor composition at all in-plane k
            datafiles_spinor = {
                'kp6': list(),
                'kp8': list()
            }
            datafiles = self.get_DataFiles(['spinor_composition', 'CbHhLhSo'], input_file.fullpath)
            datafiles = [df for cnt in range(len(datafiles)) for df in datafiles if str(cnt).zfill(5) + '_CbHhLhSo' in os.path.split(df.fullpath)[1]]   # sort spinor composition datafiles in ascending kIndex
            for df in datafiles:
                filename = os.path.split(df.fullpath)[1]
                quantum_model = self.detect_quantum_model(filename)
                if quantum_model == 'kp6':
                    datafiles_spinor['kp6'].append(df)
                elif quantum_model == 'kp8':
                    datafiles_spinor['kp8'].append(df)
                else:
                    raise RuntimeError("Unknown quantum model in spinor composition!")
            del datafiles

            # dictionary containing quantum model keys and 1-dimensional np.ndarrays for each key that stores spinor composition for all (eigenstate, kIndex)
            compositions = dict()

            for model, state_indices in states_toBePlotted.items():
                if model not in ['kp6', 'kp8']: 
                    continue

                compositions[model] = np.zeros((num_evs[model], num_kPoints[model], 4))   # compositions[quantum model][eigenvalue index][k index][spinor index]

                for stateIndex in state_indices:
                    for kIndex in range(num_kPoints[model]):
                        assert model in os.path.split(datafiles_spinor[model][kIndex].fullpath)[1]  # filename contains correct quantum model
                        assert str(kIndex) in os.path.split(datafiles_spinor[model][kIndex].fullpath)[1]  # filename contains correct kIndex
                        # store spinor composition data
                        if model == 'kp8':
                            compositions[model][stateIndex, kIndex, 0] = datafiles_spinor[model][kIndex].variables['cb1'].value[stateIndex] + datafiles_spinor[model][kIndex].variables['cb2'].value[stateIndex]

                        compositions[model][stateIndex, kIndex, 1] = datafiles_spinor[model][kIndex].variables['hh1'].value[stateIndex] + datafiles_spinor[model][kIndex].variables['hh2'].value[stateIndex]
                        compositions[model][stateIndex, kIndex, 2] = datafiles_spinor[model][kIndex].variables['lh1'].value[stateIndex] + datafiles_spinor[model][kIndex].variables['lh2'].value[stateIndex]
                        compositions[model][stateIndex, kIndex, 3] = datafiles_spinor[model][kIndex].variables['so1'].value[stateIndex] + datafiles_spinor[model][kIndex].variables['so2'].value[stateIndex]
        else:
            compositions = None

        # define plot title
        # title = CommonShortcuts.adjust_plot_title(plot_title)
        title = plot_title

        def draw_spinor_pie_charts(gs_spinor, state_indices, model, stateIndex, kIndex, show_state_index):
            num_rows, num_columns = self.get_rowColumn_for_display(len(state_indices))  # determine arrangement of spinor composition plots
            list_of_colors = [self.default_colors.bands[model] for model in ['CB', 'HH', 'LH', 'SO']]
            for i in range(num_rows):
                for j in range(num_columns):
                    subplotIndex = j + num_columns * i
                    if subplotIndex >= len(state_indices): break

                    ax = fig.add_subplot(gs_spinor[i, j])
                    stateIndex = state_indices[subplotIndex]
                    ax.pie(compositions[model][stateIndex, kIndex, :], colors=list_of_colors, normalize=True, startangle=90, counterclock=False)   # compositions[quantum model][eigenvalue index][k index][spinor index]
                    if show_state_index: ax.set_title(f'n={stateIndex+1}', color='grey')


        # instantiate matplotlib subplot objects for bandedge & probability distribution & spinor pie charts
        for model, state_indices in states_toBePlotted.items():
            for kIndex in range(num_kPoints[model]):
                if only_k0 and kIndex > 0: break

                if show_spinor and (model == 'kp6' or model == 'kp8'):
                    num_rows, num_columns = self.get_rowColumn_for_display(len(state_indices))

                    fig = plt.figure(figsize=plt.figaspect(0.4))
                    grid_probability = GridSpec(1, 1, figure=fig, left=0.10, right=0.48, bottom=0.15, wspace=0.05)
                    grid_spinor = GridSpec(num_rows, num_columns, figure=fig, left=0.55, right=0.98, bottom=0.15, hspace=0.2)
                    ax_probability = fig.add_subplot(grid_probability[0,0])
                else:
                    fig, ax_probability = plt.subplots()
                if only_k0:
                    ax_probability.set_title(f'{title} (quantum model: {model})', color=self.default_colors.bands[model])
                else:
                    ax_probability.set_title(f'{title} (quantum model: {model}), k index: {kIndex}', color=self.default_colors.bands[model])
                want_valence_band = (model != 'Gamma')
                self.draw_bandedges(ax_probability, title, model, x, CBBandedge, want_valence_band, HHBandedge, LHBandedge, True)


                if model == 'kp8':
                    # define colorbar representing electron fraction
                    divnorm = colors.TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0)
                    scalarmappable = plt.cm.ScalarMappable(cmap=self.colormap['divergent'], norm=divnorm)
                    cbar = fig.colorbar(scalarmappable)
                    cbar.set_label("Electron fraction", fontsize=labelsize)
                    cbar.ax.tick_params(labelsize=ticksize)
                else:
                    scalarmappable = None

                self.draw_probabilities(ax_probability, state_indices, x, psiSquared, model, kIndex, show_state_index, color_by_fraction_of, scalarmappable, compositions)

                if show_spinor and (model == 'kp6' or model == 'kp8'):
                    draw_spinor_pie_charts(grid_spinor, state_indices, model, stateIndex, kIndex, show_state_index)
                else:
                    fig.tight_layout()


        # if both electrons and holes have been calculated separately (but not by kp8),
        # plot bandedge and probabilities at k|| = 0
        calculated_e_models = [model for model in states_toBePlotted if model in self.model_names_conduction]
        calculated_h_models = [model for model in states_toBePlotted if model in self.model_names_valence]

        models = [model in self.model_names_conduction for model in states_toBePlotted]
        models.append([model in self.model_names_conduction for model in states_toBePlotted])

        if len(calculated_e_models) >= 1 and len(calculated_h_models) >= 1:
            fig, ax_combi = plt.subplots()
            if verbose_title:
                title_str = f"{title} ({calculated_e_models}+{calculated_h_models}), zone-center"
            else:
                title_str = title
            ax_combi.set_title(title_str)
            want_valence_band = True
            self.draw_bandedges(ax_combi, title, self.band_names['kp8'], x, CBBandedge, want_valence_band, HHBandedge, LHBandedge, True)
            # TODO: Do we need to draw SO band edge?

            for model in calculated_e_models + calculated_h_models:
                self.draw_probabilities(ax_combi, states_toBePlotted[model], x, psiSquared, model, 0, show_state_index, color_by_fraction_of, scalarmappable, compositions)
            fig.tight_layout()

        return fig
    

    def export_figs(self, 
            figFilename, 
            figFormat, 
            dpi=300,
            output_subfolder_name='nextnanopy', 
            output_folder_path='', 
            fig=None
            ):
        """
        Export all the matplotlib.pyplot objects in multi-page PDF file or other image formats with a given file name.

        Parameters
        ----------
        figFilename : str
            file name of the exported figure

        figFormat : str
            PDF = vector graphics
            EPS = vector graphics, suitable for LaTeX documents
            PNG = high quality, lossless compression, large size (recommended)
            JPG = lower quality, lossy compression, small size (not recommended)
            SVG = supports animations and image editing for e.g. Adobe Illustrator

        dpi : float, optional
            DPI value for the image
            The default is 300.
            
        output_subfolder_name : str, optional
            subfolder name in the output directory specified in the config
            The default is 'nextnanopy'

        output_folder_path : str, optional
            If present, the file will be saved to this path and output_subfolder_name will be ignored.
            The default is ''.

        fig : matplotlib.subplot object, optional
            Needed if non-PDF format is desired. The default is None.

        Returns
        -------
        None.

        Details
        -------
            fig, ax = plt.subplots() must exist, i.e. subplot object(s) must be instantiated before calling this method.
            specify image format in the argument of this function if non-PDF format is desired.

            This method must be invoked before calling plt.show(). Otherwise, the figure will be displayed on the screen and you won't be able to save it as an image.

            PNG and other non-PDF formats cannot generate multiple pages and ends up with one plot when multiple subplots instances exist.

        """
        import matplotlib.backends.backend_pdf as backendPDF

        # validate arguments
        if '.' not in figFormat:
            figFormat = '.' + figFormat
        if figFormat not in CommonShortcuts.figFormat_list:
            raise ValueError(f"Non-supported figure format! It must be one of the following:\n{CommonShortcuts.figFormat_list}")

        if fig is None and not figFormat == '.pdf':
            raise ValueError("Argument 'fig' must be specified to export non-PDF images!")

        if isinstance(fig, list) and len(fig) > 1 and not figFormat == '.pdf':
            raise NotImplementedError("Non-PDF formats cannot generate multiple pages.")

        # prepare output subfolder path
        if output_folder_path:
            output_subfolder = os.path.join(output_folder_path, "nextnanopy")
        else:
            output_subfolder_name = PathHandler.separate_extension(output_subfolder_name)[0]   # chop off file extension if any
            output_subfolder = os.path.join(nn.config.get(self.product_name, 'outputdirectory'), output_subfolder_name)

        mkdir_if_not_exist(output_subfolder)
        export_fullpath = os.path.join(output_subfolder, figFilename + figFormat)
        logging.info(f'Exporting figure to: \n{export_fullpath}\n')

        if figFormat == '.pdf':
            with backendPDF.PdfPages(export_fullpath, False) as pdf:
                for figure in range(1, plt.gcf().number + 1):
                    pdf.savefig(figure)
        else:
            fig.savefig(export_fullpath, dpi=dpi, facecolor='w', bbox_inches='tight')  # 'tight' bbox ensures that legends outside of axes frame is included in the image


