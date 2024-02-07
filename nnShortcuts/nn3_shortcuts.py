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
from nnShortcuts.common import CommonShortcuts, NextnanopyScriptError, NextnanoInputFileError, NextnanoInputFileWarning


class nn3Shortcuts(CommonShortcuts):
    # nextnano solver
    product_name = 'nextnano3'
    
    model_names = ['cb1', 'vb1', 'vb2', 'vb3', 'kp6', 'kp8']  # in nextnano3
    model_names_conduction = ['cb1']
    model_names_valence    = ['vb1', 'vb2', 'vb3', 'kp6', 'kp8']



    def detect_quantum_model(self, filename):
        """
        Detect the quantum model from a file name.

        INPUT:
            filename    name of output data file

        RETURN:
            quantum model string
        """
        found = False

        for model in self.model_names:
            if model + '_' in filename[0:4]:
                found = True
                quantum_model = model
                break

        if not found:
            raise RuntimeError('Quantum model cannot be detected!')

        return quantum_model


    def getKPointsData1D_in_folder(self, folder_path):

        datafiles = self.get_DataFiles_in_folder('k_points', folder_path)

        # inplaneK_dict = {
        #     'Gamma': list(),
        #     'kp6': list(),
        #     'kp8': list()
        # }
        inplaneK_dict = {model_name: list() for model_name in self.model_names}
        # TODO: adjustments needed
        for df in datafiles:
            filename = os.path.split(df.fullpath)[1]
            quantum_model = self.detect_quantum_model(filename)
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


    def get_DataFile_probabilities_in_folder(self, folder_path, bias=None):
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
        datafiles = self.get_DataFiles_in_folder('_psi_squared', folder_path)  # TODO: is this finding the correct files?

        # probability_dict = {
        #     'Gamma': list(),
        #     'kp6': list(),
        #     'kp8': list()
        # }
        probability_dict = {model_name: list() for model_name in self.model_names}
        for df in datafiles:
            filename = os.path.split(df.fullpath)[1]
            quantum_model = self.detect_quantum_model(filename)
            probability_dict[quantum_model].append(df)

        # delete quantum model keys whose probabilities do not exist in output folder
        models_to_be_removed = [model for model, df in probability_dict.items() if len(df) == 0]

        for model in models_to_be_removed:
            probability_dict.pop(model)

        return probability_dict


    def get_DataFile_amplitudesK0_in_folder(self, folder_path):
        """
        Get single nextnanopy.DataFile of zone-center amplitude data in the folder of specified name.
        Shifted data is avoided since non-shifted one is used to calculate overlap and matrix elements.

        INPUT:
            folder_path      output folder path in which the datafile should be sought

        RETURN:
            dictionary { quantum model key: list of nn.DataFile() objects for amplitude data }
        """
        datafiles = self.get_DataFiles_in_folder('psi', folder_path, exclude_keywords='shift')   # return a list of nn.DataFile

        # amplitude_dict = {
        #     'Gamma': list(),
        #     'kp6': list(),
        #     'kp8': list()
        # }
        amplitude_dict = {model_name: list() for model_name in self.model_names}
        for df in datafiles:
            filename = os.path.split(df.fullpath)[1]
            quantum_model = self.detect_quantum_model(filename)

            if quantum_model == 'kp8' or quantum_model == 'kp6':
                if '_001_' not in filename: continue   # exclude non k|| = 0 amplitudes   # TODO: check if it's working
            amplitude_dict[quantum_model].append(df)

        # delete quantum model keys whose probabilities do not exist in output folder
        amplitude_dict_trimmed = {model: amplitude_dict[model] for model in amplitude_dict if len(amplitude_dict[model]) > 0}

        if len(amplitude_dict_trimmed) == 0:
            raise NextnanoInputFileError("Amplitudes are not output! Modify the input file.")

        return amplitude_dict_trimmed



    ############### plot probabilities ##################################


    ############### find ground states from kp8 result ############################
    def find_lowest_conduction_state_atK0(self, output_folder, threshold=0.5):
        """
        TODO: Since nn3 outputs e- and h-like eigenstates in separate output files, 
        1. this method should return 0
        2. methods that use this method should distinguish two output files kp8_psi_squared_el and kp8_psi_squared_hl. 

        From spinor composition data, determine the lowest conduction band state in an 8-band k.p simulation.
        This method should be able to detect it properly even when the effective bandgap is negative,
        i.e. when the lowest conduction band state is below the highest valence band state.

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
        state index (base 0) of the lowest conduction band state at in-plane k = 0

        """
        # get nn.DataFile object   # TODO: where is the output of spinor composition in nn3?
        try:
            datafile = self.get_DataFile_in_folder(['eigenvalues', '_info'], output_folder)   # spinor composition at in-plane k = 0
        except FileNotFoundError:
            warnings.warn("Spinor components output in CbHhLhSo basis is not found. Assuming decoupling of the conduction and valence bands...", category=NextnanoInputFileWarning)
            return int(0)

        # check if it is an 8-band k.p simulation result
        filename = os.path.split(datafile.fullpath)[1]
        if self.detect_quantum_model(filename) != 'kp8':
            raise NextnanopyScriptError("This method only applies to 8-band k.p model!")

        # find the lowest conduction band state
        num_evs = len(datafile.variables['cb1'].value)
        for stateIndex in range(num_evs):
            electronFraction = datafile.variables['cb1'].value[stateIndex] + datafile.variables['cb2'].value[stateIndex]
            if electronFraction > threshold:
                return stateIndex

        raise RuntimeError(f"No electron states found in: {output_folder}")


    def find_highest_valence_state_atK0(self, output_folder, threshold=0.5):
        """
        TODO: Since nn3 outputs e- and h-like eigenstates in separate output files, 
        1. this method should return 0
        2. methods that use this method should distinguish two output files kp8_psi_squared_el and kp8_psi_squared_hl. 

        From spinor composition data, determine the highest valence band state in an 8-band k.p simulation.
        This method should be able to detect it properly even when the effective bandgap is negative,
        i.e. when the highest valence band state is above the lowest conduction band state.

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
        state index (base 0) of the highest valence band state at in-plane k = 0

        """
        # get nn.DataFile object  # TODO: where is the output of spinor composition in nn3?
        try:
            datafile = self.get_DataFile_in_folder(['eigenvalues', '_info'], output_folder)   # spinor composition at in-plane k = 0
        except FileNotFoundError:
            warnings.warn("Spinor components output in CbHhLhSo basis is not found. Assuming decoupling of the conduction and valence bands...", category=NextnanoInputFileWarning)
            return int(0)

        # check if it is an 8-band k.p simulation result
        filename = os.path.split(datafile.fullpath)[1]
        if self.detect_quantum_model(filename) != 'kp8':
            raise NextnanopyScriptError("This method only applies to 8-band k.p model!")

        # find the highest valence band state
        num_evs = len(datafile.variables['cb1'].value)
        for stateIndex in reversed(range(num_evs)):
            electronFraction = datafile.variables['cb1'].value[stateIndex] + datafile.variables['cb2'].value[stateIndex]
            if electronFraction < threshold:
                return stateIndex

        raise RuntimeError(f"No hole states found in: {output_folder}")


    ################ optics analysis #############################################
    kp8_basis = ['cb1', 'cb2', 'hh1', 'hh2', 'lh1', 'lh2', 'so1', 'so2']
    kp6_basis = ['hh1', 'hh2', 'lh1', 'lh2', 'so1', 'so2']

    def calculate_overlap(self, output_folder, force_lightHole=False):
        """
        Calculate envelope overlap between the lowest electron and highest valence band states from wavefunction output data

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
        datafile_amplitude_at_k0 = self.get_DataFile_amplitudesK0_in_folder(output_folder)   # returns a dict of nn.DataFile

        if 'kp8' in datafile_amplitude_at_k0.keys():
            electron_state_is_multiband = True
            hole_state_is_multiband = True
            df_e = datafile_amplitude_at_k0['kp8'][0]
            df_h = datafile_amplitude_at_k0['kp8'][0]
            h_state_basis = self.kp8_basis
        elif 'kp6' in datafile_amplitude_at_k0.keys():
            electron_state_is_multiband = False
            hole_state_is_multiband = True
            df_e = datafile_amplitude_at_k0['Gamma'][0]
            df_h = datafile_amplitude_at_k0['kp6'][0]
            h_state_basis = self.kp6_basis
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
                E_HH = self.get_DataFile_in_folder(['energy_spectrum', '_HH'], output_folder).variables['Energy'].value[0]   # energy of the first eigenstate
                E_LH = self.get_DataFile_in_folder(['energy_spectrum', '_LH'], output_folder).variables['Energy'].value[0]   # energy of the first eigenstate
                if E_HH >= E_LH:
                    df_h = df_HH
                    h_state_basis = ['HH']
                else:
                    df_h = df_LH
                    h_state_basis = ['LH']
            
        x = df_e.coords['x'].value
        iLowestElectron = self.find_lowest_conduction_state_atK0(output_folder, threshold=0.5)
        iHighestHole    = self.find_highest_valence_state_atK0(output_folder, threshold=0.5)
        

        # extract amplitude of electron-like state
        if electron_state_is_multiband:
            amplitude_e = np.zeros((len(self.kp8_basis), len(x)), dtype=np.cdouble)
            for mu, band in enumerate(self.kp8_basis):
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
            for mu in range(len(self.kp8_basis)):
                for nu in range(len(self.kp8_basis)):
                    prod = np.multiply(np.conjugate(amplitude_h[nu,:]), amplitude_e[mu,:])   # multiply arguments element-wise
                    overlap += simps(prod, x)
        elif hole_state_is_multiband:
            for nu in range(len(self.kp6_basis)):
                prod = np.multiply(np.conjugate(amplitude_h[nu,:]), amplitude_e[:])   # multiply arguments element-wise
                overlap += simps(prod, x)
        else:
            prod = np.multiply(np.conjugate(amplitude_h[:]), amplitude_e[:])   # multiply arguments element-wise
            overlap += simps(prod, x)

        return overlap


    def get_transition_energy(self, output_folder, force_lightHole=False):
        """ 
        Get the transition energy = energy separation between the lowest electron and highest valence band states.
        Unit: eV
        """
        # TODO: make it compatible with single-band & kp6 models. See nnp implementation
        # NOTE: nn3 has two output files '_el' and '_hl' also in 8kp calculation.
        datafile_el = self.get_DataFile_in_folder(["eigenvalues", "el"], output_folder, exclude_keywords=["info", "pos"])
        datafile_hl = self.get_DataFile_in_folder(["eigenvalues", "hl"], output_folder, exclude_keywords=["info", "pos"])
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


    def get_hole_energy_difference(self, output_folder):
        """ 
        Get the hole energy difference = energy separation between the highest HH and highest LH states.
        Unit: eV
        """
        E_HH = self.get_DataFile_in_folder(['energy_spectrum', '_HH'], output_folder).variables['Energy'].value[0]   # energy of the first eigenstate
        E_LH = self.get_DataFile_in_folder(['energy_spectrum', '_LH'], output_folder).variables['Energy'].value[0]   # energy of the first eigenstate
            
        return E_HH - E_LH
