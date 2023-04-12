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



class CommonShortcuts:
    # nextnano solver
    product_name = 'common'

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

    def __init__(self, loglevel=logging.INFO):
        # log setting
        fmt = '[%(levelname)s] %(message)s'
        logging.basicConfig(level=loglevel, format=fmt)
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
        'kp8': 'black'
        }

    labelsize_default = 16
    ticksize_default = 14


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

    # -------------------------------------------------------
    # Math
    # -------------------------------------------------------
    def is_half_integer(self, x : float):
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
            raise TypeError("Input must be numpy.ndarray!")

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
            raise TypeError("Input must be numpy.ndarray!")

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

    def electronvolt_to_micron(self, E):
        """
        Convert energy in eV to micrometer.

        E : array-like
            energy in eV
        """
        energy_in_J = E * self.elementary_charge   # eV to J
        wavelength_in_meter = self.Planck * self.speed_of_light / energy_in_J   # J to m
        return wavelength_in_meter * self.scale1ToMicro   # m to micron

    def wavenumber_to_energy(self, sound_velocity, k_in_inverseMeter):
        """
        For linear dispersion, convert wavenumber in [1/m] to energy [eV].
        E = hbar * omega = hbar * c * k
        """
        E_in_Joules = self.hbar * sound_velocity * k_in_inverseMeter
        return E_in_Joules * self.scale_eV_to_J**(-1)



    # -------------------------------------------------------
    # Simulation preprocessing
    # -------------------------------------------------------
    # We make it a static method because:
    # - this utility function doesn't access any properties of the class but makes sense that it belongs to the class
    # - we want to forbid method override in the inherited classes
    # - we want to make this method available without instantiation of an object.
    @staticmethod
    def separate_extension(filename):
        """
        Separate file extension from file name.
        Returns the original filename and empty string if extension is absent.
        """
        filename = os.path.split(filename)[1]   # remove paths if present
        filename_no_extension, extension = os.path.splitext(filename)

        if extension not in ['', '.in', '.xml', '.negf']: 
            raise RuntimeError(f"File extension {extension} is not supported by nextnano.")

        return filename_no_extension, extension



    def detect_software(self, folder_path, filename):
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
        product_name : str
            nextnano solver
        product_name_short : str
            shorthand of nextnano solver
        extension : str
            file extension
        """

        extension = self.separate_extension(filename)[1]
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
            raise self.NextnanoInputFileError('Software cannot be detected! Please check your input file.')
        else:
            logging.info(f'Software detected: {product_name}')

        return product_name, product_name_short, extension



    def detect_software_new(self, inputfile):
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
            raise self.NextnanoInputFileError('Software cannot be detected! Please check your input file.')
        else:
            logging.info(f'Software detected: {product_name}')

        return product_name, extension
    


    def get_shortcut(self, inputfile):
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
                        return nn3Shortcuts()
                    elif 'run{' in line:
                        from nnShortcuts.nnp_shortcuts import nnpShortcuts
                        return nnpShortcuts()
                    elif '<nextnano.NEGF' in line or 'nextnano.NEGF{' in line:
                        from nnShortcuts.NEGF_shortcuts import NEGFShortcuts
                        return NEGFShortcuts()
                    elif '<nextnano.MSB' in line or 'nextnano.MSB{' in line:
                        raise NotImplementedError("MSB shortcuts are not implemented")
                raise self.NextnanoInputFileError('Software cannot be detected! Please check your input file.')
        
        except FileNotFoundError as e:
            raise Exception(f'Input file {inputfile.fullpath} not found!') from e



    def prepare_InputFile(self,
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

        filename_no_extension, extension = self.separate_extension(originalFilename)
        if extension == '':
            raise ValueError('Input file name must include file extension!')
        newFilename = filename_no_extension + filename_appendix + extension
        logging.info(f'Saving input file as:\t{newFilename}\n')
        input_file.save(os.path.join(folderPath, newFilename), overwrite=True)   # update input file name

        return newFilename, input_file


    # -------------------------------------------------------
    # Bandedge and k.p parameters
    # -------------------------------------------------------
    # We make it a static method because:
    # - this utility function doesn't access any properties of the class but makes sense that it belongs to the class
    # - we want to forbid method override in the inherited classes
    # - we want to make this method available without instantiation of an object.
    @staticmethod
    def get_bandgap_at_T(bandgap_at_0K, alpha, beta, T):
        """ Varshni formula """
        return bandgap_at_0K - alpha * T**2 / (T + beta)


    # We make it a static method because:
    # - this utility function doesn't access any properties of the class but makes sense that it belongs to the class
    # - we want to forbid method override in the inherited classes
    # - we want to make this method available without instantiation of an object.
    @staticmethod
    def get_factor_zb(Eg, deltaSO):
        """
        Temperature-dependent factor for the conversion among effective mass, S and Ep
        """
        return (Eg + 2. * deltaSO / 3.) / Eg / (Eg + deltaSO)

    def mass_from_kp_parameters(self, Ep, S, Eg, deltaSO):
        factor = self.get_factor_zb(Eg, deltaSO)   # independent of S and Ep, but temperature-dependent
        mass = 1. / (S + Ep * factor)
        return mass


    def Ep_from_mass_and_S(self, mass, S, Eg, deltaSO):
        factor = self.get_factor_zb(Eg, deltaSO)   # independent of S and Ep, but temperature-dependent
        Ep = (1./mass - S) / factor
        return Ep


    def Ep_from_P(self, P):
        """
        Convert the Kane parameter P [eV nm] into energy Ep [eV].
        #NOTE: nextnano++ output is in units of [eV Angstrom].
        """
        P_in_SI = P * self.scale_eV_to_J * self.scale1ToNano**(-1)
        Ep_in_SI = P_in_SI**2 * 2 * self.electron_mass / (self.hbar**2)
        return Ep_in_SI / self.scale_eV_to_J


    def P_from_Ep(self, Ep):
        """
        Convert the Kane energy Ep [eV] into P [eV nm].
        #NOTE: nextnano++ output is in units of [eV Angstrom].
        """
        from math import sqrt
        Ep_in_SI = Ep * self.scale_eV_to_J
        P_in_SI = self.hbar * sqrt(Ep_in_SI / 2 / self.electron_mass)
        return P_in_SI * self.scale_eV_to_J**(-1) * self.scale1ToNano


    # TODO: extend to WZ
    def evaluate_and_rescale_S(self, 
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
        factor = self.get_factor_zb(Eg, deltaSO)   # independent of S and Ep, but temperature-dependent

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
        cSchroedinger = self.hbar**2 / 2 / self.electron_mass
        new_L = db_L + cSchroedinger * (new_Ep - db_Ep) / Eg
        new_N = db_N + cSchroedinger * (new_Ep - db_Ep) / Eg

        return new_S, new_Ep, new_L, new_N

    # We make it a static method because:
    # - this utility function doesn't access any properties of the class but makes sense that it belongs to the class
    # - we want to forbid method override in the inherited classes
    # - we want to make this method available without instantiation of an object.
    @staticmethod
    def shift_DKK_as_nnp(DKK_parameter, Eg_T, old_Ep, new_Ep):
        """ 
        Shift the 8-band DKK parameters as in nn++/nn3.
        
        Return
        ------
            In units of hbar^2 / 2m_0
        """
        return DKK_parameter + (new_Ep - old_Ep) / Eg_T

    # We make it a static method because:
    # - this utility function doesn't access any properties of the class but makes sense that it belongs to the class
    # - we want to forbid method override in the inherited classes
    # - we want to make this method available without instantiation of an object.
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

    # TODO: extend to WZ
    def rescale_Ep_and_get_S(self,
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
        factor = self.get_factor_zb(Eg, deltaSO)   # independent of S and Ep, but temperature-dependent
        new_S = old_S + (old_Ep - rescaleEpTo) * factor

        # L' and N' get modified by the change of Ep
        # This keeps the kp6 values at T=0K intact.
        cSchroedinger = self.hbar**2 / 2 / self.electron_mass
        new_L = old_L + cSchroedinger * (rescaleEpTo - old_Ep) / Eg
        new_N = old_N + cSchroedinger * (rescaleEpTo - old_Ep) / Eg
        return new_S, new_L, new_N


    def get_8kp_from_6kp_NEGF(self, mass, rescaleSTo, Eg_0K, Eg_finiteTemp, deltaSO, L_6kp, N_6kp):
        """
        Imitate NEGF implementation.
        1. Calculate Ep by rescaling at T=0
        2. Calculate L', M, N', S from this Ep but using Eg at nonzero T
        """
        Ep = self.Ep_from_mass_and_S(mass, rescaleSTo, Eg_0K, deltaSO)
        print(f"Calculated Ep from mass, S, and Eg(0): {Ep}")

        correction = Ep / Eg_finiteTemp   # L, N in the database are in units of hbar^2/2m0
        Lprime = L_6kp + correction
        Nprime = N_6kp + correction
        
        factor = self.get_factor_zb(Eg_finiteTemp, deltaSO)   # independent of S and Ep, but temperature-dependent
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
        except KeyError:
            determined = False
            while not determined:
                choice = input('Simulation has not been executed. Continue? [y/n]')
                if choice == 'n':
                    raise RuntimeError('Terminated nextnanopy.') from None
                elif choice == 'y':
                    determined = True
                else:
                    print("Invalid input.")
                    continue


    def get_sweep_output_folder_name(self, filename, *args):
        """
        nextnanopy.sweep.execute_sweep() generates output folder with this name

        INPUT:
            filename
            args = SweepVariableString1, SweepVariableString2, ...

        RETURN:
            string of sweep output folder name

        """
        filename_no_extension = self.separate_extension(filename)[0]
        output_folderName = filename_no_extension + '_sweep'

        for sweepVar in args:
            if not isinstance(sweepVar, str):
                raise TypeError(f'Argument {sweepVar} must be a string!')
            output_folderName += '__' + sweepVar

        return output_folderName


    def get_sweep_output_folder_path(self, filename, *args):
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
        filename_no_extension = self.separate_extension(filename)[0]
        output_folder_path = os.path.join(nn.config.get(self.product_name, 'outputdirectory'), filename_no_extension + '_sweep')

        if len(args) == 0: raise ValueError("Sweep variable string is missing in the argument!")

        for sweepVar in args:
            if not isinstance(sweepVar, str):
                raise TypeError(f'Argument {sweepVar} must be a string!')
            output_folder_path += '__' + sweepVar

        return output_folder_path


    def get_output_subfolder_path(self, sweep_output_folder_path, input_file_name):
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
        subfolder_name = self.separate_extension(input_file_name)[0]
        return os.path.join(sweep_output_folder_path, subfolder_name)


    def get_sweep_output_subfolder_name(self, filename, sweepCoordinates):
        """
        nextnanopy.sweep.execute_sweep() generates output subfolders with this name

        INPUT:
            filename
            {sweepVariable1: value1, sweepVariable2: value2, ...}

        RETURN:
            string of sweep output subfolder name

        """
        filename_no_extension = self.separate_extension(filename)[0]
        output_subfolderName = filename_no_extension + '__'

        for sweepVar, value in sweepCoordinates.items():
            if not isinstance(sweepVar, str):
                raise TypeError('key must be a string!')
            try:
                val = str(value)
            except ValueError as e:
                raise Exception(f'value {value} cannot be converted to string!') from e
            else:
                output_subfolderName +=  sweepVar + '_' + val + '_'

        return output_subfolderName



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
        outputFolder = nn.config.get(self.product_name, 'outputdirectory')
        filename_no_extension = self.separate_extension(name)[0]
        outputSubfolder = os.path.join(outputFolder, filename_no_extension)

        return self.get_DataFile_in_folder(keywords, outputSubfolder, exclude_keywords=exclude_keywords)


    def get_DataFile_in_folder(self, keywords, folder_path, exclude_keywords=None):
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
            raise FileNotFoundError(f"No output file found!")
        elif len(list_of_files) == 1:
            file = list_of_files[0]
        else:
            logging.warning(f"More than one output files found!")
            for count, file in enumerate(list_of_files):
                filename = os.path.split(file)[1]
                print(f"Choice {count}: {filename}")
            determined = False
            while not determined:
                choice = input('Enter the index of data you need: ')
                if choice == 'q':
                    raise RuntimeError('Terminated nextnanopy.') from None
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

        logging.debug(f"Found:\n{file}")

        try:
            return nn.DataFile(file, product=self.product_name)
        except NotImplementedError as e:
            raise Exception(f'Nextnanopy does not support datafile for {file}') from e


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
        outputFolder = nn.config.get(self.product_name, 'outputdirectory')
        filename_no_extension = self.separate_extension(name)[0]
        outputSubFolder = os.path.join(outputFolder, filename_no_extension)

        return self.get_DataFiles_in_folder(keywords, outputSubFolder, exclude_keywords=exclude_keywords)


    def get_DataFiles_in_folder(self, keywords, folder_path, exclude_keywords=None):
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
            raise FileNotFoundError(f"No output file found!")
        elif len(list_of_files) == 1:
            warnings.warn("get_DataFiles_in_folder(): Only one output file found!", category=RuntimeWarning)

        logging.debug(f"Found:\n{list_of_files}")

        try:
            datafiles = [nn.DataFile(file, product=self.product_name) for file in list_of_files]
        except NotImplementedError as e:
            raise Exception('Nextnanopy does not support datafile') from e

        return datafiles


    def plot_probabilities(self):
        raise NotImplementedError("There is no common implementation")


    def get_DataFile_probabilities_with_name(self, name):
        """
        Get single nextnanopy.DataFile of probability_shift data in the folder of specified name.

        INPUT:
            name      input file name (= output subfolder name). May contain extensions and fullpath

        RETURN:
            dictionary { quantum model key: corresponding list of nn.DataFile() objects for probability_shift }
        """
        filename_no_extension = self.separate_extension(name)[0]
        outputFolder = nn.config.get(self.product_name, 'outputdirectory')
        outputSubFolder = os.path.join(outputFolder, filename_no_extension)

        return self.get_DataFile_probabilities_in_folder(outputSubFolder)
        

    def get_DataFile_probabilities_in_folder(self):
        raise NotImplementedError("There is no common implementation")


    def __get_num_evs(self, probability_dict):
        """ number of eigenvalues for each quantum model """
        num_evs = dict()
        for model, datafiles in probability_dict.items():
            if len(datafiles) == 0:   # if no k-points are calculated
                num_evs[model] = 0
            else:
                df = datafiles[0]
                num_evs[model] = sum(1 for var in df.variables if ('Psi^2' in var.name))   # this conditional counting is necessary because probability output may contain also eigenvalues and/or bandedges.
                logging.debug(f"Number of eigenvalues for {model}: {num_evs[model]}")
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
            num_evs                         for each quantum model, number of eigenstates

        NOTE:
            state index is base 0 (differ from nextnano++ output), state No is base 1 (identical to nextnano++ output)
        """
        # validate input
        if states_range_dict is not None and not isinstance(states_range_dict, dict):
            raise TypeError("Argument 'states_range_dict' must be a dict")
        if states_list_dict is not None and not isinstance(states_list_dict, dict):
            raise TypeError("Argument 'states_list_dict' must be a dict")
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
                            if model != 'kp8' and model not in self.model_names_valence:
                                raise ValueError(f"Quantum model '{model}' does not contain hole states.")
                            
                            # TODO: nn3 has two output files '_el' and '_hl' also in 8kp calculation
                            states_toBePlotted[model].append(self.find_highest_hole_state_atK0(outfolder, threshold=0.5))
                                
                        elif stateNo == 'lowestElectron':
                            if model != 'kp8' and model not in self.model_names_conduction:
                                raise ValueError(f"Quantum model '{model}' does not contain electron states.")

                            states_toBePlotted[model].append(self.find_lowest_electron_state_atK0(outfolder, threshold=0.5))
                            
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

                            states_toBePlotted[model] += [int(stateNo) - 1 for stateNo, occupation in zip(df.coords['no.'].value, df.variables['Occupation'].value) if occupation >= cutoff_occupation]
                        elif isinstance(stateNo, int):
                            if stateNo > num_evs[model]: 
                                raise ValueError("State index greater than number of eigenvalues calculated!")
                            states_toBePlotted[model].append(stateNo - 1)   
        logging.debug(f"states_toBePlotted (index base 0): {states_toBePlotted}")
        return states_toBePlotted, num_evs

    def get_transition_energy(self):
        raise NotImplementedError("There is no common implementation")
    
    def find_highest_hole_state_atK0(self):
        raise NotImplementedError("There is no common implementation")

    def find_lowest_electron_state_atK0(self):
        raise NotImplementedError("There is no common implementation")


    # -------------------------------------------------------
    # Data postprocessing
    # -------------------------------------------------------

    # We make it a static method because:
    # - this utility function doesn't access any properties of the class but makes sense that it belongs to the class
    # - we want to forbid method override in the inherited classes
    # - we want to make this method available without instantiation of an object.
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


    # We make it a static method because:
    # - this utility function doesn't access any properties of the class but makes sense that it belongs to the class
    # - we want to forbid method override in the inherited classes
    # - we want to make this method available without instantiation of an object.
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
            raise RuntimeError(f'No grid cells found that contain the point x = {wanted_value}')
        if cnt > 1:
            raise RuntimeError(f'Multiple grid cells found that contain the point x = {wanted_value}')
        return start_index, end_index



    def get_value_at_position(self, quantity_arr, position_arr, wantedPosition):
        """
        Get value at given position.
        If the position does not match any of array elements due to inconsistent gridding, interpolate array and return the value at wanted position.
        """
        if len(quantity_arr) != len(position_arr):
            raise ValueError('Array size does not match!')

        start_idx, end_idx = self.findCell(position_arr, wantedPosition)

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



    def get_maximum_points(quantity_arr, position_arr):
        if isinstance(quantity_arr, int) or isinstance(quantity_arr, float):
            warnings.warn(f"get_maximum_points(): Only one point exists in the array {quantity_arr}", category=RuntimeWarning)
            return position_arr[0], quantity_arr

        if len(quantity_arr) != len(position_arr):
            raise ValueError('Array size does not match!')
        ymax = np.amax(quantity_arr)
        if np.size(ymax) > 1:
            warnings.warn("Multiple maxima found. Taking the first...")
            ymax = ymax[0]
        xmaxIndex = np.where(quantity_arr == ymax)[0]
        xmax = position_arr[xmaxIndex.item(0)]             # type(xmaxIndex.item(0)) is 'int'

        return xmax, ymax



    def generate_colorscale(colormap, minValue, maxValue):
        """
        Generate a color scale with given colormap and range of values.
        """
        return plt.cm.ScalarMappable( cmap=colormap, norm=plt.Normalize(vmin=minValue, vmax=maxValue) )



    # We make it a static method because:
    # - this utility function doesn't access any properties of the class but makes sense that it belongs to the class
    # - we want to forbid method override in the inherited classes
    # - we want to make this method available without instantiation of an object.
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
            raise TypeError(f"Array must be numpy.ndarray. Type is {type(arr)}")

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


    def set_plot_labels(self, ax, x_label, y_label, title):
        """
        Set the labels and optimize their sizes (matplotlib default of font size is too small!)
        """
        ax.set_xlabel(x_label, fontsize=self.labelsize_default)  # r with superscript works
        ax.set_ylabel(y_label, fontsize=self.labelsize_default)
        ax.set_title(title, fontsize=self.labelsize_default)
        ax.tick_params(axis='x', labelsize=self.ticksize_default)
        ax.tick_params(axis='y', labelsize=self.ticksize_default)
        return ax


    def get_plot_title(self, originalTitle):
        """
        If the title is too long for display, omit the intermediate letters
        """
        title = self.separate_extension(originalTitle)[0]   # remove extension if present

        if len(title) > 25:
            beginning = title[:10]
            last  = title[-10:]
            title = beginning + ' ... ' + last

        return title



    def export_figs(self, 
            figFilename, 
            figFormat, 
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
            PDF = vector graphic
            PNG = high quality, lossless compression, large size (recommended)
            JPG = lower quality, lossy compression, small size (not recommended)
            SVG = supports animations and image editing for e.g. Adobe Illustrator

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
        if figFormat not in self.figFormat_list:
            raise ValueError(f"Non-supported figure format! It must be one of the following:\n{self.figFormat_list}")

        if fig is None and not figFormat == '.pdf':
            raise ValueError("Argument 'fig' must be specified to export non-PDF images!")

        if isinstance(fig, list) and len(fig) > 1 and not figFormat == '.pdf':
            raise NotImplementedError("Non-PDF formats cannot generate multiple pages.")

        # prepare output subfolder path
        if output_folder_path:
            outputSubfolder = os.path.join(output_folder_path, "nextnanopy")
        else:
            output_subfolder_name = self.separate_extension(output_subfolder_name)[0]   # chop off file extension if any
            outputSubfolder = os.path.join(nn.config.get(self.product_name, 'outputdirectory'), output_subfolder_name)

        mkdir_if_not_exist(outputSubfolder)
        export_fullpath = os.path.join(outputSubfolder, figFilename + figFormat)
        logging.info(f'Exporting figure to: \n{export_fullpath}\n')

        if figFormat == '.pdf':
            with backendPDF.PdfPages(export_fullpath, False) as pdf:
                for figure in range(1, plt.gcf().number + 1):
                    pdf.savefig(figure)
        else:
            fig.savefig(export_fullpath, dpi=200)


