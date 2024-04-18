"""
Created: 2022/03/22
Updated: 2023/02/21

Useful shortcut functions for nextnano.NEGF postprocessing.
get_* methods return nn.DataFile() attribute (output data)
plot_* methods plot & save figures
animate_NEGF method generates animation

@author: takuma.sato@nextnano.com (inspired by scripts of David Stark)
"""

# Python includes
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging

# nextnanopy includes
import nextnanopy as nn
from nnShortcuts.common import CommonShortcuts, NextnanopyScriptError, NextnanoInputFileError, NextnanoInputFileWarning


class NEGFShortcuts(CommonShortcuts):
    # nextnano solver
    product_name = 'null'
    model_names = ['Gamma', 'kp8']

    SchrodingerRawSolutionFolder = "EnergyEigenstatesFull0V"
    wannierStarkFolder = "EnergyEigenstates"

    def __init__(self, is_xml, loglevel=logging.INFO):
        if is_xml:
            self.product_name = 'nextnano.NEGF'
        else:
            self.product_name = 'nextnano.NEGF++'
        super().__init__(loglevel)

    def get_IV(self, input_file_name):
        """
        Get I-V curve.
        OUTPUT: 2 nn.DataFile() attributes for current & voltage
        """
        datafile = self.get_DataFile('Current_vs_Voltage', input_file_name)
        voltage = datafile.coords['Potential per period']
        current = datafile.variables['Current density']
        return voltage, current

    def plot_IV(self, input_file_name):
        """
        Plot the I-V curve.
        The plot is saved as an png image file.
        """
        voltage, current = self.get_IV(input_file_name)

        fig, ax = plt.subplots()
        ax.plot(voltage.value, current.value, 'o-')
        ax.set_xlabel(voltage.label)
        ax.set_ylabel("Current density ($\mathrm{A}/\mathrm{cm}^{2}$)")
        # ax.set_title(input_file_name)

        # export to an image file
        outputFolder = nn.config.get(self.product_name, 'outputdirectory')
        filename_no_extension = CommonShortcuts.separate_extension(input_file_name)[0]
        outputSubfolder = os.path.join(outputFolder, filename_no_extension)
        self.export_figs("IV", "png", output_folder_path=outputSubfolder, fig=fig)



    def get_DataFiles_NEGFInit_in_folder(self, keywords, folder, search_raw_solution_folder = False, search_WannierStark_folder = False, exclude_keywords = None):
        """
        Get one or more nextnanopy.DataFile objects in the 'Init' folder.
        'Init' folder may contain both the raw Schrodinger solution and Wannier-Stark states.
        """
        if search_raw_solution_folder and search_WannierStark_folder:
            raise ValueError("Invalid input")
        
        if os.path.isdir(folder):
            subfolder = folder
        else:
            # unexpected, but alternative candidate path
            output_folder = nn.config.get(self.product_name, 'outputdirectory')
            filename_no_extension = os.path.split(folder)[1]   # remove paths if present
            subfolder = os.path.join(output_folder, filename_no_extension)
        d = nn.DataFolder(subfolder)

        # Get the fullpath of 'Init' subfolder
        for folder_name in d.folders.keys():
            if 'Init' in folder_name:
                # listOfFiles = d.go_to(folder_name).find(keyword, deep=True)
                init_folder = d.go_to(folder_name)
        
        if not init_folder: 
            raise RuntimeError(f"'Init' folder not found under\n{subfolder}")
        
        if search_raw_solution_folder:
            search_folder = os.path.join(init_folder.fullpath, NEGFShortcuts.SchrodingerRawSolutionFolder)
        elif search_WannierStark_folder:
            search_folder = os.path.join(init_folder.fullpath, NEGFShortcuts.wannierStarkFolder)
        else:
            search_folder = init_folder.fullpath

        return self.get_DataFiles_in_folder(keywords, search_folder, exclude_keywords)  # TODO: add options available

        
    def get_DataFile_amplitudesK0_in_folder(self, folder_path):
        """
        Get single nextnanopy.DataFile of zone-center amplitude data in the folder of specified name.
        Shifted data is avoided since non-shifted one is used to calculate overlap and matrix elements.

        INPUT:
            folder_path      output folder path in which the datafile should be sought

        RETURN:
            dictionary { quantum model key: list of nn.DataFile() objects for amplitude data }
        """
        datafiles = self.get_DataFiles_NEGFInit_in_folder('wavefunctions', folder_path, search_raw_solution_folder=True, exclude_keywords=['shift', 'spinor'])   # return a list of nn.DataFile

        # amplitude_dict = {
        #     'Gamma': list(),
        #     'kp6': list(),
        #     'kp8': list()
        # }
        amplitude_dict = {model_name: list() for model_name in self.model_names}
        for df in datafiles:
            filename = os.path.split(df.fullpath)[1]
            quantum_model = 'kp8'  # currently, NEGF has amplitude output on in kp8

            if quantum_model == 'kp8' or quantum_model == 'kp6':
                if '_k000_' not in filename: continue   # exclude non k|| = 0 amplitudes
            amplitude_dict[quantum_model].append(df)

        # delete quantum model keys whose probabilities do not exist in output folder
        amplitude_dict_trimmed = {model: amplitude_dict[model] for model in amplitude_dict if len(amplitude_dict[model]) > 0}

        if len(amplitude_dict_trimmed) == 0:
            raise NextnanoInputFileError(f"Amplitudes are not found at {folder_path}! Modify the input file.")

        return amplitude_dict_trimmed


    def get_DataFile_NEGF_atBias(self, keywords, name, bias):
        """
        Get single nextnanopy.DataFile of NEGF output data with the given string keyword(s) at the specified bias.

        Parameters
        ----------
        keywords : str or list of str
            Find output data file with the names containing single keyword or multiple keywords (AND search)
        name : str
            input file name (= output subfolder name). May contain extensions and/or fullpath.
        bias : float
            voltage drop per period

        Returns
        -------
        nextnanopy.DataFile object of the simulation data

        """
        output_folder = nn.config.get(self.product_name, 'outputdirectory')
        filename_no_extension = CommonShortcuts.separate_extension(name)[0]
        bias_subfolder = os.path.join(output_folder, filename_no_extension, str(bias) + 'mv')

        return self.get_DataFile_in_folder(keywords, bias_subfolder)


    # def get_convergenceInfo(self, bias):
    #     """
    #     Get convergence info at each bias. (Useful??)
    #     NOTE: nn.InputFile().execute(checkConvergence=True) may do the job more simply.
    #     OUTPUT: dict object
    #     {
    #       'Convergence factor':
    #       'Current convergence factor':
    #       'Number of iterations':
    #       'Normalisation lesser Green function':
    #       'Sum normalised spectral function':
    #     }

    #     """
    #     output_folder = nn.config.get(self.product_name, 'outputdirectory')
    #     filePath = os.path.join(output_folder, str(bias) + 'mv\\Convergence.txt')
    #     fp = open(filePath, 'r')
    #     lines = fp.readlines()
    #     conv = list()
    #     for line in lines:
    #         conv.append(line.strip().split('='))

    #     convergenceInfo = dict()
    #     for i in conv:
    #         convergenceInfo[i[0].strip()] = float(i[1])

    #     return convergenceInfo


    def get_conduction_bandedge(self, input_file_name, bias):
        """
        INPUT:
            input file name
            bias value

        RETURN: nn.DataFile() attributes
            datafile.coords['Position']
            datafile.variables['Conduction Band Edge']
        """
        try:
            datafile = self.get_DataFile_NEGF_atBias("EigenStates.dat", input_file_name, bias)
        except FileNotFoundError:
            try:
                datafile = self.get_DataFile_NEGF_atBias('Conduction_BandEdge.dat', input_file_name, bias)
            except FileNotFoundError:
                try:
                    datafile = self.get_DataFile_NEGF_atBias('ConductionBandEdge.dat', input_file_name, bias)
                    # print('Found ConductionBandEdge.dat')
                except FileNotFoundError:
                    try:
                        datafile = self.get_DataFile_NEGF_atBias('BandEdges.dat', input_file_name, bias)
                    except FileNotFoundError:
                        raise

        position = datafile.coords['Position']
        try:
            bandedge = datafile.variables['Conduction Band Edge']
        except KeyError:
            try:
                bandedge = datafile.variables['ConductionBandEdge']
            except KeyError:
                raise

        return position, bandedge
    

    def get_lightHole_bandedge(self, input_file_name, bias):
        """
        INPUT:
            input file name
            bias value

        RETURN: nn.DataFile() attributes
            datafile.coords['Position']
            datafile.variables['Light-Hole Band Edge']
        """
        try:
            datafile = self.get_DataFile_NEGF_atBias("EigenStates.dat", input_file_name, bias)
        except FileNotFoundError:
            try:
                datafile = self.get_DataFile_NEGF_atBias('BandEdges.dat', input_file_name, bias)
            except FileNotFoundError:
                raise

        position = datafile.coords['Position']
        try:
            bandedge = datafile.variables['Light-Hole Band Edge']
        except KeyError:
            try:
                bandedge = datafile.variables['LightHoleBandEdge']
            except KeyError:
                raise

        return position, bandedge


    def get_heavyHole_bandedge(self, input_file_name, bias):
        """
        INPUT:
            input file name
            bias value

        RETURN: nn.DataFile() attributes
            datafile.coords['Position']
            datafile.variables['Light-Hole Band Edge']
        """
        try:
            datafile = self.get_DataFile_NEGF_atBias("EigenStates.dat", input_file_name, bias)
        except FileNotFoundError:
            try:
                datafile = self.get_DataFile_NEGF_atBias('BandEdges.dat', input_file_name, bias)
            except FileNotFoundError:
                raise

        position = datafile.coords['Position']
        try:
            bandedge = datafile.variables['Heavy-Hole Band Edge']
        except KeyError:
            try:
                bandedge = datafile.variables['HeavyHoleBandEdge']
            except KeyError:
                raise

        return position, bandedge


    def get_splitOffHole_bandedge(self, input_file_name, bias):
        """
        INPUT:
            input file name
            bias value

        RETURN: nn.DataFile() attributes
            datafile.coords['Position']
            datafile.variables['Split-Off-Hole Band Edge']
        """
        try:
            datafile = self.get_DataFile_NEGF_atBias("EigenStates.dat", input_file_name, bias)
        except FileNotFoundError:
            try:
                datafile = self.get_DataFile_NEGF_atBias('BandEdges.dat', input_file_name, bias)
            except FileNotFoundError:
                raise

        position = datafile.coords['Position']
        try:
            bandedge = datafile.variables['Split-Off-Hole Band Edge']
        except KeyError:
            try:
                bandedge = datafile.variables['SplitOffHoleBandEdge']
            except KeyError:
                raise

        return position, bandedge
    

    def get_Fermi_levels(self, input_file_name, bias):
        """
        INPUT:
            input file name
            bias value

        RETURN: nn.DataFile() attributes
            z coordinates
            electron Fermi level
            hole Fermi level
        """
        datafile = self.get_DataFile_NEGF_atBias("FermiLevel.dat", input_file_name, bias=bias)
        position = datafile.coords['Position']
        try:
            return position, datafile.variables['Fermi level'], datafile.variables['Fermi level']
        except KeyError:
            return position, datafile.variables['Electron Fermi level'], datafile.variables['Hole Fermi level']
        

    def get_carrier_density_deviation(self, input_file_name, bias):
        datafile = self.get_DataFile_NEGF_atBias("CarrierDensityDeviation.dat", input_file_name, bias=bias)
        return datafile.coords['Position'], datafile.variables['Deviation from local neutrality']


    def get_WannierStarkStates_init(self, filename_no_extension):
        """
        RETURN: nn.DataFile() attribute
            datafile.coords['Position']
            datafile.variables['Conduction BandEdge']
            datafile.variables['Psi_*']
        """
        datafiles = self.get_DataFiles_NEGFInit_in_folder('EigenStates.dat', filename_no_extension, search_WannierStark_folder=True)
        for df in datafiles:
            if NEGFShortcuts.wannierStarkFolder in df.fullpath: 
                datafile = df

        position = datafile.coords['Position']
        conduction_bandedge = datafile.variables['ConductionBandEdge']

        Psi_squareds = []
        num_evs = len(datafile.variables) - 1
        for n in range(num_evs):
            for key in datafile.variables.keys():
                if f'Psi_{n+1}' in key: wanted_key = key   # NEGF output contains level and period indices, so variables['Psi_{n+1}'] doesn't work
            Psi_squareds.append(datafile.variables[wanted_key])

        return position, conduction_bandedge, Psi_squareds


    def get_WannierStarkStates_atBias(self, input_file, bias):
        """
        RETURN: nn.DataFile() attribute
            datafile.coords['Position']
            datafile.variables['Conduction BandEdge']
            datafile.variables['Psi_*']
        """
        datafile = self.get_DataFile_NEGF_atBias('EigenStates.dat', input_file, bias)

        position = datafile.coords['Position']
        conduction_bandedge = datafile.variables['ConductionBandEdge']

        Psi_squareds = []
        num_evs = len(datafile.variables) - 1
        print(datafile.variables['Psi_1 (lev.1 per.0)'].value)
        for n in range(num_evs):
            for key in datafile.variables.keys():
                if f'Psi_{n+1}' in key: wanted_key = key   # NEGF output contains level and period indices, so variables['Psi_{n+1}'] doesn't work
            Psi_squareds.append(datafile.variables[wanted_key])

        return position, conduction_bandedge, Psi_squareds


    def plot_WannierStarkStates_init(self,
            filename_no_extension, 
            start_position = -10000., 
            end_position   = 10000., 
            labelsize      = CommonShortcuts.labelsize_default, 
            ticksize       = CommonShortcuts.ticksize_default
            ):
        """
        Plot Wannier-Stark states on top of the conduction bandedge.
        The plot is saved as an png image file.

        Parameters
        ----------
            
        labelsize : int, optional
            font size of xlabel and ylabel
        ticksize : int, optional
            font size of xtics and ytics

        RETURN:
            matplotlib plot
        """
        position, CB, Psi_squareds = self.get_WannierStarkStates_init(filename_no_extension)

        # Store data arrays.
        # Cut off edges of the simulation region if needed.
        conduction_bandedge = CommonShortcuts.cutOff_edges1D(CB.value, position.value, start_position, end_position)
        WS_states           = [CommonShortcuts.cutOff_edges1D(Psi_squared.value, position.value, start_position, end_position) for Psi_squared in Psi_squareds]
        x                   = CommonShortcuts.cutOff_edges1D(position.value, position.value, start_position, end_position)

        WS_states = [CommonShortcuts.mask_part_of_array(WS_state) for WS_state in WS_states]   # hide flat tails

        fig, ax = plt.subplots()
        ax.set_xlabel(position.label, fontsize=labelsize)
        ax.set_ylabel("Energy (eV)", fontsize=labelsize)
        ax.set_title('Wannier-Stark states', fontsize=labelsize)
        ax.tick_params(axis='x', labelsize=ticksize)
        ax.tick_params(axis='y', labelsize=ticksize)
        ax.plot(x, conduction_bandedge, color=self.default_colors.bands['CB'], linewidth=0.7, label=CB.label)
        for WS_state in WS_states:
            ax.plot(x, WS_state, label='')   # TODO: label
        fig.tight_layout()

        # export to an image file
        outputFolder = nn.config.get(self.product_name, 'outputdirectory')
        outputSubfolder = os.path.join(outputFolder, filename_no_extension)
        self.export_figs("WannierStarkStates_init", "png", output_folder_path=outputSubfolder, fig=fig)

        return fig


    def get_gain_atBias(self, input_file, bias):
        datafile = self.get_DataFile_NEGF_atBias(['Gain', 'vs_Energy'], input_file, bias)
        return datafile.coords['Photon Energy'], datafile.variables['Gain']
    

    def get_SCGain_atBias(self, input_file, bias):
        datafile = self.get_DataFile_NEGF_atBias(['SemiClassical_', 'vs_Energy'], input_file, bias)
        return datafile.coords['Photon Energy'], datafile.variables['Gain']
    

    def get_SCGainDecomposed_atBias(self, input_file, bias):
        datafile = self.get_DataFile_NEGF_atBias(['SemiClassicalDecomposed', 'vs_Energy'], input_file, bias)
        return datafile.coords['Photon Energy'], datafile.variables['Gain']


    def plot_gain(self,
            input_file_name, 
            bias,
            start_position = -10000., 
            end_position   = 10000., 
            labelsize      = CommonShortcuts.labelsize_default, 
            ticksize       = CommonShortcuts.ticksize_default,
            title          = 'Gain spectrum',
            flipSign       = False
            ):
        """
        Plot NEGF gain (calculated from dynamical conductivity).
        The plot is saved as an png image file.

        Parameters
        ----------
            
        labelsize : int, optional
            font size of xlabel and ylabel
        ticksize : int, optional
            font size of xtics and ytics

        RETURN:
            matplotlib plot
        """
        EPhoton, gain = self.get_gain_atBias(input_file_name, bias)

        # Store data arrays.
        # Cut off edges of the simulation region if needed.
        gain_trimmed = CommonShortcuts.cutOff_edges1D(gain.value, EPhoton.value, start_position, end_position)
        EPhoton_trimmed = CommonShortcuts.cutOff_edges1D(EPhoton.value, EPhoton.value, start_position, end_position)

        if flipSign:
            gain_trimmed = - gain_trimmed

        fig, ax = plt.subplots()
        ax.set_xlabel(EPhoton.label, fontsize=labelsize)
        ax.set_ylabel("Gain (1/cm)", fontsize=labelsize)
        ax.set_title(title, fontsize=labelsize)
        ax.tick_params(axis='x', labelsize=ticksize)
        ax.tick_params(axis='y', labelsize=ticksize)
        ax.axhline(color='grey', linewidth=1.0)
        ax.plot(EPhoton_trimmed, gain_trimmed, color='orange', linewidth=1.0, label=EPhoton.label)
        fig.tight_layout()

        return fig
    

    def plot_SCGain(self,
            input_file_name, 
            bias,
            start_position = -10000., 
            end_position   = 10000., 
            labelsize      = CommonShortcuts.labelsize_default, 
            ticksize       = CommonShortcuts.ticksize_default,
            title          = 'Semiclassical gain spectrum',
            flip_sign       = False
            ):
        """
        Plot semiclassical gain (calculated from Fermi's golden rule).
        The plot is saved as an png image file.

        Parameters
        ----------
            
        labelsize : int, optional
            font size of xlabel and ylabel
        ticksize : int, optional
            font size of xtics and ytics

        RETURN:
            matplotlib plot
        """
        EPhoton, gain = self.get_SCGain_atBias(input_file_name, bias)
        # EPhoton, gain_decomposed = self.get_SCGainDecomposed_atBias(input_file_name, bias)
        print(gain)

        # Store data arrays.
        # Cut off edges of the simulation region if needed.
        gain_trimmed = CommonShortcuts.cutOff_edges1D(gain.value, EPhoton.value, start_position, end_position)
        # gain_decomposed_trimmed = CommonShortcuts.cutOff_edges1D(gain_decomposed.value, EPhoton.value, start_position, end_position)
        EPhoton_trimmed = CommonShortcuts.cutOff_edges1D(EPhoton.value, EPhoton.value, start_position, end_position)

        if flip_sign:
            gain_trimmed = - gain_trimmed
            # gain_decomposed_trimmed = - gain_decomposed_trimmed
            
        fig, ax = plt.subplots()
        ax.set_xlabel(EPhoton.label, fontsize=labelsize)
        ax.set_ylabel("Gain (1/cm)", fontsize=labelsize)
        ax.set_title(title, fontsize=labelsize)
        ax.tick_params(axis='x', labelsize=ticksize)
        ax.tick_params(axis='y', labelsize=ticksize)
        ax.axhline(color='grey', linewidth=1.0)
        ax.plot(EPhoton_trimmed, gain_trimmed, color='orange', linewidth=1.0, label=EPhoton.label)
        fig.tight_layout()

        return fig
    

    def get_2Ddata_atBias(self, input_file_name, bias, data='carrier'):
        """
        INPUT: one of the following strings: ['LDOS', 'carrier', 'current', 'current_with_dispersion']

        RETURN: nn.DataFile() attributes
            x = datafile.coords['x']
            y = datafile.coords['y']
            z = datafile.variables[variableKey]
        """
        if data == 'LDOS' or data == 'DOS':
            file = 'DOS_energy_resolved.vtr'
            variableKey = 'Density of states'
        elif data == 'carrier':
            file = 'ElectronDensity_energy_resolved.vtr'
            variableKey = 'Electron density'
        elif data == 'current':
            file = 'CurrentDensity_energy_resolved.vtr'
            variableKey = 'Current Density'
        elif data == 'current_with_dispersion':
            file = 'CurrentDensity_energy_resolved_WithDispersion.vtr'
            variableKey = 'Current Density'
        else:
            raise KeyError(f'Illegal data {data} requested!')

        datafile = self.get_DataFile_NEGF_atBias(file, input_file_name, bias)

        x = datafile.coords['x']
        y = datafile.coords['y']
        quantity = datafile.variables[variableKey]
        return x, y, quantity


    def draw_bandedges_on_2DPlot(self, ax, input_file_name, bias, labelsize, shadowBandgap):
        CB = None
        LH = None
        HH = None

        position, CB = self.get_conduction_bandedge(input_file_name, bias)
        try:
            position, LH = self.get_lightHole_bandedge(input_file_name, bias)
        except:
            pass
        try:
            position, HH = self.get_heavyHole_bandedge(input_file_name, bias)
        except:
            pass

        ax.set_xlabel(position.label, fontsize=labelsize)
        ax.plot(position.value, CB.value, color=self.default_colors.bands_dark_background['CB'], linewidth=0.7, label=CB.label)
        if LH is not None: ax.plot(position.value, LH.value, color=self.default_colors.bands_dark_background['LH'], linewidth=0.7, label=LH.label)
        if HH is not None: ax.plot(position.value, HH.value, color=self.default_colors.bands_dark_background['HH'], linewidth=0.7, label=HH.label)

        if shadowBandgap:
            # fill the gap
            if HH is not None:
                if LH is not None:
                    VBTop = np.maximum(HH.value, LH.value)
                else:
                    VBTop = HH.value
                ax.fill_between(position.value, VBTop, CB.value, color='grey')
            

    def draw_Fermi_levels_on_2DPlot(self, ax, input_file_name, bias, labelsize):
        """
        Returns
        -------
        E_FermiElectron : float
            electron Fermi energy
        E_FermiHole : float
            hole Fermi energy
        """
        color = self.default_colors.lines_on_colormap

        position, FermiElectron, FermiHole = self.get_Fermi_levels(input_file_name, bias)
        ax.plot(position.value, FermiElectron.value, color=color, linewidth=0.7, label=FermiElectron.label)
        ax.plot(position.value, FermiHole.value, color=color, linewidth=0.7, label=FermiHole.label)

        E_FermiElectron = FermiElectron.value[0]
        E_FermiHole = FermiHole.value[0]
        zmax = np.amax(position.value)
        isEquilibrium = (np.amin(FermiElectron.value) == np.amin(FermiHole.value)) and (np.amax(FermiElectron.value) == np.amax(FermiHole.value))
        if isEquilibrium:
            ax.annotate("$E_F$", color=color, fontsize=labelsize, xy=(0.9*zmax, E_FermiElectron), xytext=(0.9*zmax, E_FermiElectron + 0.05))
        else:
            ax.annotate("$E_F^e$", color=color, fontsize=labelsize, xy=(0.9*zmax, E_FermiElectron), xytext=(0.9*zmax, E_FermiElectron + 0.05))
            ax.annotate("$E_F^h$", color=color, fontsize=labelsize, xy=(0.9*zmax, E_FermiHole), xytext=(0.9*zmax, E_FermiHole + 0.05))
        return E_FermiElectron, E_FermiHole
        

    def draw_carrier_density_deviation_on_2DPlot(self, ax, input_file_name, bias, labelsize, E_FermiElectron):
        color = 'orange'

        position, carrier_dens_deviation = self.get_carrier_density_deviation(input_file_name, bias)
        max_deviation = np.amax(carrier_dens_deviation.value)
        min_deviation = np.amin(carrier_dens_deviation.value)
        min_energy, max_energy = ax.get_ylim()
        
        # rescale the density deviation so it fits within the energy range of ax
        scaling_factor = 0.3 * (max_energy - min_energy) / (max_deviation - min_deviation)
        print(f"Scaling deviation by {scaling_factor}")
        carrier_dens_deviation_scaled = carrier_dens_deviation.value * scaling_factor        
        ax.plot(position.value, E_FermiElectron + carrier_dens_deviation_scaled, color=color, linewidth=1.0, label=carrier_dens_deviation.label)

        zmax = np.amax(position.value)
        ax.annotate("$n_e - n_B$", color=color, fontsize=labelsize, xy=(0.1*zmax, E_FermiElectron), xytext=(0.1*zmax, E_FermiElectron + 0.2))


    def __get_inplane_dispersion(self, input_file_name, startIdx, stopIdx):
        """
        Parameters
        ----------
        input_file_name : str
            input file name (= output subfolder name). May contain extensions and/or fullpath.

        Returns
        -------
        tuple of numpy arrays
        """
        # load output data files
        filename_no_extension = CommonShortcuts.separate_extension(input_file_name)[0]
        datafiles_dispersion = self.get_DataFiles_NEGFInit_in_folder('InplaneDispersionReducedCorrected', filename_no_extension, search_raw_solution_folder=True)
        if len(datafiles_dispersion) == 0:
            raise RuntimeError("No output found for in-plane dispersion!")
        if len(datafiles_dispersion) > 1:
            raise RuntimeError("More than one output file found for in-plane dispersion!")
        df = datafiles_dispersion[0]

        # store data in arrays
        kPoints     = df.coords['inplane k'].value
        num_bands   = len(df.variables)
        logging.debug(f'num_bands = {num_bands}')
        num_kPoints = len(kPoints)

        # determine which states to plot   # TODO: bug: the last index isn't plotted if startIdx==0 and stopIdx==0!
        if startIdx == 0: startIdx = 1
        if stopIdx == 0: stopIdx = num_bands
        states_toBePlotted = np.arange(startIdx-1, stopIdx, 1)

        # store data in arrays
        dispersions = np.zeros((num_bands, num_kPoints), dtype=np.double)
        for index in states_toBePlotted:
            dispersions[index, ] = df.variables[f'Energy_{index+1}'].value

        return kPoints, dispersions, states_toBePlotted


    def plot_DOS(self,
            input_file_name, 
            bias, 
            labelsize=CommonShortcuts.labelsize_default,
            ticksize=CommonShortcuts.ticksize_default,
            zmin=None,
            zmax=None,
            attachDispersion=False, 
            shadowBandgap=True,
            showBias=True,
            showFermiLevel=True,
            showDensityDeviation=True,
            lattice_temperature=None
            ):
        """
        Overlay bandedge with local density of states. Loads the following output data:
        DOS_energy_resolved.vtr

        The plot is saved as an png image file.

        Parameters
        ----------
        lattice_temperature : float
            If not None, the energy kBT is indicated inside the dispersion plot.
        """
        x, y, quantity = self.get_2Ddata_atBias(input_file_name, bias, 'LDOS')

        logging.info("Plotting DOS...")
        unit = r'$\mathrm{nm}^{-1} \mathrm{eV}^{-1}$'
        label = 'Density of states (' + unit + ')'

        if attachDispersion:
            # Create subplots with shared y-axis and remove spacing
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 3]})
            plt.subplots_adjust(wspace=0)

            self.__draw_2D_color_plot(fig, ax2, x.value, y.value, quantity.value, label, bias, labelsize, ticksize, zmin, zmax, showBias, False)
            self.draw_bandedges_on_2DPlot(ax2, input_file_name, bias, labelsize, shadowBandgap)
            if showFermiLevel:
                E_FermiElectron, E_FermiHole = self.draw_Fermi_levels_on_2DPlot(ax2, input_file_name, bias, labelsize)
                if showDensityDeviation:
                    self.draw_carrier_density_deviation_on_2DPlot(ax2, input_file_name, bias, labelsize, E_FermiElectron)

            kPoints, dispersions, states_toBePlotted = self.__get_inplane_dispersion(input_file_name, 0, 0)  # TODO: implement user-defined state index range (see nnpShortcuts.plot_dispersion)
            if showBias:
                title = 'Dispersion'
            else:
                title = ''
            CommonShortcuts.draw_inplane_dispersion(ax1, kPoints, dispersions, states_toBePlotted, True, True, labelsize, title=title, lattice_temperature=lattice_temperature)  # dispersions[iState, ik]
        else:
            fig, ax = plt.subplots()
            self.__draw_2D_color_plot(fig, ax, x.value, y.value, quantity.value, label, bias, labelsize, ticksize, zmin, zmax, showBias, True)
            self.draw_bandedges_on_2DPlot(ax, input_file_name, bias, labelsize, shadowBandgap)
            if showFermiLevel:
                E_FermiElectron, E_FermiHole = self.draw_Fermi_levels_on_2DPlot(ax, input_file_name, bias, labelsize)
                if showDensityDeviation:
                    self.draw_carrier_density_deviation_on_2DPlot(ax, input_file_name, bias, labelsize, E_FermiElectron)

        fig.tight_layout()

        # export to an image file
        outputFolder = nn.config.get(self.product_name, 'outputdirectory')
        filename_no_extension = CommonShortcuts.separate_extension(input_file_name)[0]
        outputSubfolder = os.path.join(outputFolder, filename_no_extension)
        self.export_figs("DOS", "png", output_folder_path=outputSubfolder, fig=fig)

        return fig


    def plot_carrier_density(self,
            input_file_name, 
            bias, 
            labelsize=CommonShortcuts.labelsize_default, 
            ticksize=CommonShortcuts.ticksize_default,
            zmin=None,
            zmax=None,
            attachDispersion=False,
            shadowBandgap=True,
            showBias=True,
            showFermiLevel=True,
            showDensityDeviation=True,
            lattice_temperature=None
            ):
        """
        Overlay bandedge with energy-resolved carrier density. Loads the following output data:
        CarrierDensity_energy_resolved.vtr or ElectronDensity_energy_resolved.vtr
        
        The plot is saved as an png image file.

        Parameters
        ----------
        lattice_temperature : float
            If not None, the energy kBT is indicated inside the dispersion plot.
        """
        x, y, quantity = self.get_2Ddata_atBias(input_file_name, bias, 'carrier')

        logging.info("Plotting electron density...")
        unit = r'$\mathrm{cm}^{-3} \mathrm{eV}^{-1}$'
        label = 'Electron density (' + unit + ')'

        if attachDispersion:
            # Create subplots with shared y-axis and remove spacing
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 3]})
            plt.subplots_adjust(wspace=0)

            # 2D color plot
            self.__draw_2D_color_plot(fig, ax2, x.value, y.value, quantity.value, label, bias, labelsize, ticksize, zmin, zmax, showBias, False)
            self.draw_bandedges_on_2DPlot(ax2, input_file_name, bias, labelsize, shadowBandgap)
            if showFermiLevel:
                E_FermiElectron, E_FermiHole = self.draw_Fermi_levels_on_2DPlot(ax2, input_file_name, bias, labelsize)
                if showDensityDeviation:
                    self.draw_carrier_density_deviation_on_2DPlot(ax2, input_file_name, bias, labelsize, E_FermiElectron)
            
            # dispersion plot
            kPoints, dispersions, states_toBePlotted = self.__get_inplane_dispersion(input_file_name, 0, 0)  # TODO: implement user-defined state index range (see nnpShortcuts.plot_dispersion)
            if showBias:
                title = 'Dispersion'
            else:
                title = ''
            CommonShortcuts.draw_inplane_dispersion(ax1, kPoints, dispersions, states_toBePlotted, True, True, labelsize, title=title, lattice_temperature=lattice_temperature)  # dispersions[iState, ik]
        else:
            fig, ax = plt.subplots()

            # 2D color plot
            self.__draw_2D_color_plot(fig, ax, x.value, y.value, quantity.value, label, bias, labelsize, ticksize, zmin, zmax, showBias, True)
            self.draw_bandedges_on_2DPlot(ax, input_file_name, bias, labelsize, shadowBandgap)
            if showFermiLevel:
                E_FermiElectron, E_FermiHole = self.draw_Fermi_levels_on_2DPlot(ax, input_file_name, bias, labelsize)
                if showDensityDeviation:
                    self.draw_carrier_density_deviation_on_2DPlot(ax, input_file_name, bias, labelsize, E_FermiElectron)
            
        fig.tight_layout()

        # export to an image file
        outputFolder = nn.config.get(self.product_name, 'outputdirectory')
        filename_no_extension = CommonShortcuts.separate_extension(input_file_name)[0]
        outputSubfolder = os.path.join(outputFolder, filename_no_extension)
        self.export_figs("CarrierDensity", "png", output_folder_path=outputSubfolder, fig=fig)

        return fig


    def plot_current_density(self,
            input_file_name, 
            bias, 
            labelsize=CommonShortcuts.labelsize_default, 
            ticksize=CommonShortcuts.ticksize_default,
            zmin=None,
            zmax=None,
            shadowBandgap=False,
            showBias=True
            ):
        """
        Overlay bandedge with energy-resolved current density. Loads the following output data:
        CurrentDensity_energy_resolved.vtr

        The plot is saved as an png image file.
        """
        x, y, quantity = self.get_2Ddata_atBias(input_file_name, bias, 'current')

        logging.info("Plotting current density...")
        unit = r'$\mathrm{A}$ $\mathrm{cm}^{-2} \mathrm{eV}^{-1}$'
        label = 'Current density (' + unit + ')'

        fig, ax = plt.subplots()
        self.__draw_2D_color_plot(fig, ax, x.value, y.value, quantity.value, label, bias, labelsize, ticksize, zmin, zmax, showBias, True)
        self.draw_bandedges_on_2DPlot(ax, input_file_name, bias, labelsize, shadowBandgap)
        fig.tight_layout()

        # export to an image file
        outputFolder = nn.config.get(self.product_name, 'outputdirectory')
        filename_no_extension = CommonShortcuts.separate_extension(input_file_name)[0]
        outputSubfolder = os.path.join(outputFolder, filename_no_extension)
        self.export_figs("CurrentDensity", "png", output_folder_path=outputSubfolder, fig=fig)

        return fig


    def __draw_2D_color_plot(self, fig, ax, X, Y, Z, label, bias, labelsize, ticksize, zmin, zmax, showBias, set_ylabel):
        pcolor = ax.pcolormesh(X, Y, Z.T, vmin=zmin, vmax=zmax, cmap=self.default_colors.colormap)
        cbar = fig.colorbar(pcolor)
        cbar.set_label(label, fontsize=labelsize)
        cbar.ax.tick_params(labelsize=ticksize * 0.9)

        if set_ylabel:
            ax.set_ylabel("Energy (eV)", fontsize=labelsize)
        ax.set_xlim(np.amin(X), np.amax(X))
        ax.set_ylim(np.amin(Y), np.amax(Y))
        if showBias:
            ax.set_title(f'bias={bias}mV', fontsize=labelsize)
        else:
            ax.set_title('', fontsize=labelsize)
        ax.tick_params(axis='x', labelsize=ticksize)
        ax.tick_params(axis='y', labelsize=ticksize)


    # TODO: Which quantity should be overlayed to Gain?
    # TODO: support for other xaxes, for self-consistent results
    # def get_gain(input_file_name, xaxis='Energy'):
    #     """
    #     Get 2D plot of gain with specified x-axis. Loads one of the following output data:
    #     - Gain_vs_EField*.dat (1D)
    #     - Gain_vs_Voltage*.dat (1D)
    #     - Gain_Map_EnergyVoltage.fld
    #     - Gain_SelfConsistent_vs_Energy.dat (2D)
    #     - Gain_SelfConsistent_vs_Frequency.dat (2D)
    #     - Gain_SelfConsistent_vs_Wavelength.dat (2D)
    #     """
    #     # datafile = self.get_DataFile(f'Gain_SelfConsistent_vs_{xaxis}.dat', input_file_name)
    #     datafile = self.get_DataFile('Gain_vs_Voltage', input_file_name)
    #     voltage = datafile.variables['Potential per period']
    #     gain = datafile.variables['Maximum gain']
    #     return voltage, gain

    # def plot_gain():


    def get_biases(self, name):
        """
        Return the list of bias values that exist in the simulation output folder specified by name

        Parameters
        ----------
        name : str
            input file name (= output subfolder name). May contain extensions and/or fullpath.

        Returns
        -------
        biases : list
            bias values calculated in the simulation.

        """
        output_folder = nn.config.get(self.product_name, 'outputdirectory')
        filename_no_extension = CommonShortcuts.separate_extension(name)[0]
        datafolder = nn.DataFolder(os.path.join(output_folder, filename_no_extension))

        biases = [int(folder_name.replace('mV', '')) for folder_name in datafolder.folders.keys() if ('mV' in folder_name) and ('Init' not in folder_name)]
        biases.sort()   # ascending order
        return biases


    def animate_NEGF(self, input_file_name, leftFig='DOS', rightFig='carrier'):
        """
        Generate gif animation from simulation results at different bias points. Loads two of the following output data:
        - energy-resolved local density of states
        - energy-resolved carrier density
        - energy-resolved current density
        - emitted power
        INPUT:
            two strings among ['DOS', 'carrier', 'current', 'power']

        """
        quantity_names = ['DOS', 'carrier', 'current', 'power']   # TODO: should be global variables in this file
        unit = r'$\mathrm{nm}^{-1} \mathrm{eV}^{-1}$'   # TODO: make a list of units for 2D data
        label = 'Density of states (' + unit + ')'   # TODO: make a list of labels for 2D data

        if leftFig not in quantity_names:
            raise KeyError(f"Entry must be {quantity_names}!")
        if rightFig not in quantity_names:
            raise KeyError(f"Entry must be {quantity_names}!")

        input_file_name = CommonShortcuts.separate_extension(input_file_name)[0]

        array_of_biases = np.array(self.get_biases(input_file_name))

        # get 2D data at the largest bias
        x_last, y_last, quantity_last = self.get_2Ddata_atBias(input_file_name, array_of_biases[-1], leftFig)

        # define a map from (xIndex, yIndex, biasIndex) to scalar value
        F = np.zeros((len(x_last.value), len(y_last.value), len(array_of_biases)))

        # store data to F
        for i, bias in enumerate(array_of_biases):
            position, CB = self.get_conduction_bandedge(input_file_name, bias)
            x, y, quantity = self.get_2Ddata_atBias(input_file_name, bias, leftFig)
            F[:, :, i] = quantity.value

        fig, ax = plt.subplots()
        ax.set_xlabel('x label')
        ax.set_ylabel('y label')
        ax.set_title('title')
        ax.set_xlim(np.amin(x_last.value), np.amax(x_last.value))
        ax.set_ylim(np.amin(y_last.value), np.amax(y_last.value))

        # Plot colormap for the initial bias.
        # F[:-1, :-1, 0] gives the values on the x-y plane at initial bias.
        cax = ax.pcolormesh(x.value, y.value, F[:-1, :-1, 0], vmin=-1, vmax=1, cmap=self.default_colors.colormap)
        cbar = fig.colorbar(cax)
        cbar.set_label(label)


        def animate_2D_plots(bias):
            # update plot title
            ax.set_title('{}, bias={:.1f} mV'.format(input_file_name, array_of_biases[i]))
            # update 2D color plot
            cax.set_array(F[:-1, :-1, i].flatten())
            # update conduction bandedge plot
            ax.plot(x.value, CB.value, color=self.default_colors.lines_on_colormap, linewidth=0.7, label=CB.label)


        # ax.legend(loc='upper left')

        # generate GIF animation
        from matplotlib import animation

        cwd = os.getcwd()
        # os.chdir(output_folder_path)
        logging.info(f'Exporting GIF animation to: {os.getcwd()}\n')
        anim = animation.FuncAnimation(fig, animate_2D_plots, frames=len(array_of_biases)-1, interval=700)
        anim.save(f'{input_file_name}.gif', writer='pillow')
        # os.chdir(cwd)



    def get_LIV(self, name):
        """
        Load power-current-voltage data from simulation output.

        Parameters
        ----------
        name : str
            input file name (= output subfolder name). May contain extensions and/or fullpath.

        Units (identical to nextnano.NEGF output)
        -----------------------------------------
        current density [A/cm^2]
        potential drop per period [mV]
        output power density [W/cm^3]

        Returns
        -------
        current_density_LV : np.array
            DESCRIPTION.
        potential_drop_per_period : np.array
            DESCRIPTION.
        output_power_density : np.array
            DESCRIPTION.

        """
        # find the output file
        outputFolder = nn.config.get(self.product_name, 'outputdirectory')
        filename_no_extension = CommonShortcuts.separate_extension(name)[0]
        outputSubfolder = os.path.join(outputFolder, filename_no_extension)
        df = self.get_DataFile_in_folder('L-I-V.dat', outputSubfolder)
        
        data = np.loadtxt(df.fullpath, skiprows=1)

        current_density_LV = np.array(data[:, 0])
        potential_drop_per_period = np.array(data[:, 1])
        output_power_density = np.array(data[:, 2])

        return current_density_LV, potential_drop_per_period, output_power_density


    def plot_Light_Current_Voltage_characteristics(self,
            names, 
            period_length, 
            num_periods, 
            area, 
            front_mirror_loss, 
            total_cavity_loss, 
            labels, 
            labelsize=CommonShortcuts.labelsize_default, 
            ticksize=CommonShortcuts.ticksize_default, 
            Imin=None, Imax=None, 
            Vmin=None, Vmax=None, 
            Pmin=None, Pmax=None
            ):
        """
        Plot the voltage-current and optical output power-current characteristics in one figure.

        Parameters
        ----------
        names : list of str
            Specifies input files

        period_length : float
            length of one QCL/ICL period in [nm]

        num_periods : int
            Number of periods in the QCL/ICL structure.

        area : float
            area perpendicular to the growth direction in [m^2]

        front_mirror_loss : float
            front mirror loss

        total_cavity_loss : float
            total cavity loss

        labels : list of str
            plot legends for each simulation. Should be in the same order as 'names'.

        labelsize : int, optional
            font size of xlabel and ylabel

        ticksize : int, optional
            font size of xtics and ytics

        Imin, Imax : float
            plot range of current density

        Vmin, Vmax : float
            plot range of voltage

        Pmin, Pmax : float
            plot range of power

        Units in plot
        -------------
        current [A]
        current density [kA/cm^2]
        voltage across the entire superlattice [V]
        output power [mW]

        Units in nextnano.NEGF output
        -----------------------------
        current density [A/cm^2]
        potential drop per period [mV]
        output power density [W/cm^3]

        Returns
        -------
        fig : matplotlib.figure.Figure object

        """
        # validate arguments
        if len(names) != len(labels): 
            raise NextnanopyScriptError(f"Number of input files ({len(names)}) do not match that of plot labels ({len(labels)})")

        # volume in [cm^3]
        area_in_cm2 = area * pow(self.scale1ToCenti, 2)
        volume = (period_length / self.scale1ToNano * self.scale1ToCenti) * num_periods * area_in_cm2

        def forward_conversion(I):
            """ convert current (A) to current density (kA/cm^2) """
            return I / area_in_cm2 * self.scale1ToKilo

        def backward_conversion(density):
            """ convert current density (kA/cm^2) to current (A) """
            return density * area_in_cm2 / self.scale1ToKilo


        # list of data for sweeping temperature
        current_densities_IV = list()
        voltages             = list()
        current_densities_LV = list()
        output_powers = list()

        for i, name in enumerate(names):
            # I-V data, units adjusted
            datafile_IV = self.get_DataFile('Current_vs_Voltage.dat', name)
            density = datafile_IV.variables['Current density'].value * self.scale1ToKilo
            V = datafile_IV.coords['Potential per period'].value * num_periods / self.scale1ToMilli
            current_densities_IV.append(density)
            voltages.append(V)

            # L-V data, units adjusted
            current_density_LV, potential_drop, power_density = self.get_LIV(name)
            density = current_density_LV * self.scale1ToKilo
            P = (power_density * volume * self.scale1ToMilli) * front_mirror_loss / total_cavity_loss   # external output power
            current_densities_LV.append(density)
            output_powers.append(P)


        logging.info("\nPlotting L-I-V curves...")
        from matplotlib.ticker import MultipleLocator

        fig, ax1 = plt.subplots()

        linetypes = ['solid', 'dashed', 'dotted', 'dashdot']

        # I-V curves
        color = self.default_colors.current_voltage
        cnt = 0
        for current_density, V in zip(current_densities_IV, voltages):
            I = backward_conversion(current_density)
            ax1.plot(I, V, color=color, ls=linetypes[cnt], label=labels[cnt])
            cnt += 1
        ax1.set_xlabel('Current ($\mathrm{A}$)', fontsize=labelsize)
        ax1.set_ylabel('Voltage ($\mathrm{V}$)', color=color, fontsize=labelsize)
        ax1.tick_params(axis='y', labelcolor=color, labelsize=ticksize)
        ax1.set_xlim(Imin, Imax)
        ax1.set_ylim(Vmin, Vmax)
        plt.xticks([0.0, 0.5, 1.0, 1.5])
        plt.yticks([0, 5, 10, 15])
        ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax1.yaxis.set_minor_locator(MultipleLocator(1))
        ax1.legend()

        ax3 = ax1.secondary_xaxis('top', functions=(forward_conversion, backward_conversion))
        ax3.set_xlabel('Current density ($\mathrm{kA}/\mathrm{cm}^2$)', fontsize=labelsize)


        # Optical power - voltage curves
        color = self.default_colors.light_voltage
        ax2 = ax1.twinx()   # shared x axis
        cnt = 0
        for density, P in zip(current_densities_LV, output_powers):
            I = backward_conversion(density)
            ax2.plot(I, P, '.', color=color, ls=linetypes[cnt], label=labels[cnt])
            cnt += 1
        ax2.set_ylabel('Optical power ($\mathrm{mW}$)', color=color, fontsize=labelsize)
        ax2.tick_params(axis='y', labelcolor=color, labelsize=ticksize)
        ax2.set_ylim(Pmin, Pmax)
        plt.yticks([0, 50, 100, 150])
        ax2.yaxis.set_minor_locator(MultipleLocator(10))

        fig.tight_layout()
        plt.show()

        return fig


    ############### plot wavefunctions ##################################
    def plot_probabilities(self,
            input_file,
            bias                = None,
            states_range_dict   = None,
            states_list_dict    = None,
            start_position      = -10000.,
            end_position        = 10000.,
            hide_tails          = False,
            only_k0             = True,
            show_spinor         = False,
            show_state_index    = False,
            want_valence_band   = True,
            color_by_fraction_of = 'conduction_band',
            plot_title          = '',
            labelsize           = CommonShortcuts.labelsize_default,
            ticksize            = CommonShortcuts.ticksize_default,
            savePDF             = False,
            savePNG             = False,
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
            nextnano++ input file.
        bias : real, optional
            If not None, that bias is used to search for the energy eigenstates output folder. 
            If None, output is sought in the Init folder.
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
        want_valence_band : bool, optional
            If True, plot the valence band edges. The default is True.
        color_by_fraction_of : str, optional
            If 8-band k.p simulation, colour the probabilities by the spinor fraction of the specified band. The default is 'conduction_band'.
        plot_title : str, optional
            title of the probability plot. The default is ''.
        labelsize : int, optional
            font size of xlabel, ylabel and colorbar label
        ticksize : int, optional
            font size of xtics, ytics and colorbar tics
        savePDF : str, optional
            save the plot in the PDF format. The default is False.
        savePNG : str, optional
            save the plot in the PNG format. The default is False.

        Returns
        -------
        None.

        """
        if color_by_fraction_of not in ['conduction_band', 'heavy_hole']:
            raise ValueError(f"color_by_fraction_of '{color_by_fraction_of}' is not supported")

        from matplotlib import colors
        from matplotlib.gridspec import GridSpec

        # load output data files
        datafiles_probability_dict = self.get_DataFile_probabilities_with_name(input_file.fullpath, bias=bias)


        for model, datafiles in datafiles_probability_dict.items():
            if isinstance(datafiles, list):
                if len(datafiles) == 0: continue
                datafile_probability = datafiles[0]
                # print(type(datafile_probability))
            elif isinstance(datafiles, nn.DataFile):
                datafile_probability = datafiles
            else:
                raise RuntimeError("Data type of 'datafile_probability' " + type(datafile_probability) + " is unknown!")
            
            x_probability  = datafile_probability.coords['Position'].value
        if not datafile_probability:
            raise NextnanoInputFileError('Probabilities are not output! Modify the input file.')

        # store data in arrays (independent of quantum models)
        kIndex = 0  # TODO: currently we only support only_k0 output
        x             = datafile_probability.coords['Position'].value
        CBBandedge    = datafile_probability.variables['ConductionBandEdge'].value

        if want_valence_band:
            LHBandedge    = datafile_probability.variables['LightHoleBandEdge'].value
            HHBandedge    = datafile_probability.variables['HeavyHoleBandEdge'].value
            SOBandedge    = datafile_probability.variables['SplitOffHoleBandEdge'].value

        states_toBePlotted, num_evs = self.get_states_to_be_plotted(datafiles_probability_dict, states_range_dict=states_range_dict, states_list_dict=states_list_dict)


        # visualize the in-plane k point_maxts at which Schroedinger eq. has been solved
        if only_k0:
            num_kPoints = dict()
            for model in states_toBePlotted:
                num_kPoints[model] = 1
        # else:
            # TODO: adjust for NEGF
            # inplaneK_dict = getKPointsData1D(input_file)
            # plot_inplaneK(inplaneK_dict)
            # num_kPoints = get_num_kPoints(inplaneK_dict)


        # dictionary containing quantum model keys and 2-dimensional list for each key that stores psi^2 for all (eigenstate, kIndex)
        psiSquared = dict.fromkeys(datafiles_probability_dict.keys())
        for model in states_toBePlotted:
            psiSquared[model] = [ [ 0 for kIndex in range(num_kPoints[model]) ] for stateIndex in range(num_evs[model]) ]  # stateIndex in states_toBePlotted[model] would give a list of the same size

        nBandEdgeOutputs = 5

        for model, dfs in datafiles_probability_dict.items():
            if not isinstance(dfs, list):
                datafile = dfs
                dfs = list()
                dfs.append(datafile)
                
            if len(dfs) == 0: continue
            
            for cnt, stateIndex in enumerate(states_toBePlotted[model]):
                for kIndex in range(num_kPoints[model]):
                    # psiSquared_oldgrid = dfs[kIndex].variables[f"Psi^2_{stateIndex+1} (lev.{stateIndex+1} per.0)"].value  # TODO: generalize. nPeriod is not always 1
                    lis = list(dfs[kIndex].variables.keys())  # workaround: get the keys
                    if stateIndex+1 > len(dfs[kIndex].variables) - nBandEdgeOutputs:
                        raise NextnanopyScriptError("Specified state range is out of the range of calculated states.")
                    
                    psiSquared_oldgrid = dfs[kIndex].variables[lis[nBandEdgeOutputs + stateIndex]].value # first five data are bandedges
                    psiSquared[model][cnt][kIndex] = CommonShortcuts.convert_grid(psiSquared_oldgrid, x_probability, x)   # grid interpolation needed because of 'output_bandedges{ averaged=no }'


        # chop off edges of the simulation region
        CBBandedge = CommonShortcuts.cutOff_edges1D(CBBandedge, x, start_position, end_position)
        if want_valence_band:
            HHBandedge = CommonShortcuts.cutOff_edges1D(HHBandedge, x, start_position, end_position)
            LHBandedge = CommonShortcuts.cutOff_edges1D(LHBandedge, x, start_position, end_position)
            SOBandedge = CommonShortcuts.cutOff_edges1D(SOBandedge, x, start_position, end_position)
            # VBTop = CommonShortcuts.cutOff_edges1D(VBTop, x, start_position, end_position)


        for model in states_toBePlotted:
            for cnt, stateIndex in enumerate(states_toBePlotted[model]):
                for kIndex in range(num_kPoints[model]):
                    psiSquared[model][cnt][kIndex] = CommonShortcuts.cutOff_edges1D(psiSquared[model][cnt][kIndex], x, start_position, end_position)   # chop off edges of the simulation region

        x = CommonShortcuts.cutOff_edges1D(x, x, start_position, end_position)
        # simLength = x[-1]-x[0]   # (nm)


        # mask psiSquared data where it is flat
        if hide_tails:
            for model in states_toBePlotted:
                for cnt, stateIndex in enumerate(states_toBePlotted[model]):
                    for kIndex in range(num_kPoints[model]):
                        psiSquared[model][cnt][kIndex] = CommonShortcuts.mask_part_of_array(psiSquared[model][cnt][kIndex], method='flat', tolerance=1e-2)


        if 'kp6' in datafiles_probability_dict.keys() or 'kp8' in datafiles_probability_dict.keys():
            # output data of spinor composition at all in-plane k
            datafiles_spinor = {
                'kp6': list(),
                'kp8': list()
            }
            datafiles = self.get_DataFiles(['wavefunctions_k000_spinor_composition_AngMom'], input_file.fullpath)
            # datafiles = [df for cnt in range(len(datafiles)) for df in datafiles if str(cnt).zfill(5) + '_CbHhLhSo' in os.path.split(df.fullpath)[1]]   # sort spinor composition datafiles in ascending kIndex  # TODO: C++ doesn't have multiple in-plane k output
            for df in datafiles:
                datafiles_spinor['kp8'].append(df)
            del datafiles
            
            # dictionary containing quantum model keys and 1-dimensional np.ndarrays for each key that stores spinor composition for all (eigenstate, kIndex)
            compositions = dict()

            for model, state_indices in states_toBePlotted.items():
                if model not in ['kp6', 'kp8']: continue

                compositions[model] = np.zeros((num_evs[model], num_kPoints[model], 4))   # compositions[quantum model][eigenvalue index][k index][spinor index]
                print(datafiles_spinor[model][0])  # TODO: nextnanopy negf parser or InputFile class has a bug, or, my output from NEGF++ is not optimal
                for stateIndex in state_indices:
                    for kIndex in range(num_kPoints[model]):
                        # store spinor composition data
                        if model == 'kp8':
                            compositions[model][stateIndex, kIndex, 0] = datafiles_spinor[model][kIndex].variables[0].value[stateIndex] + datafiles_spinor[model][kIndex].variables[4].value[stateIndex]
                        # TODO: test if spinor compositions are read correctly
                        compositions[model][stateIndex, kIndex, 1] = datafiles_spinor[model][kIndex].variables[1].value[stateIndex] + datafiles_spinor[model][kIndex].variables[5].value[stateIndex]
                        compositions[model][stateIndex, kIndex, 2] = datafiles_spinor[model][kIndex].variables[2].value[stateIndex] + datafiles_spinor[model][kIndex].variables[6].value[stateIndex]
                        compositions[model][stateIndex, kIndex, 3] = datafiles_spinor[model][kIndex].variables[3].value[stateIndex] + datafiles_spinor[model][kIndex].variables[7].value[stateIndex]

        # define plot title
        title = CommonShortcuts.get_plot_title(plot_title)

        def draw_bandedges(ax, model, want_valence_band):
            self.set_plot_labels(ax, 'Position (nm)', 'Energy (eV)', title)
            if model == 'Gamma' or model == 'kp8':
                ax.plot(x, CBBandedge, label='conduction band', linewidth=0.6, color=self.default_colors.bands['CB'])
            if want_valence_band:
                if model == 'HH' or model == 'kp6' or model == 'kp8':
                    ax.plot(x, HHBandedge, label='heavy hole', linewidth=0.6, color=self.default_colors.bands['HH'])
                if model == 'LH' or model == 'kp6' or model == 'kp8':
                    ax.plot(x, LHBandedge, label='light hole', linewidth=0.6, color=self.default_colors.bands['LH'])
                # if model == 'SO' or model == 'kp6' or model == 'kp8':
                #     ax.plot(x, SOBandedge, label='split-off hole', linewidth=0.6, color=self.default_colors.bands['SO'])
                # if model == 'LH' or model == 'kp6' or model == 'kp8':
                #     ax.plot(x, VBTop, label='VB top without strain', linewidth=0.6, color=self.default_colors.bands['LH'])

        def draw_probabilities(self, ax, state_indices, model, kIndex, show_state_index, color_by_fraction_of):
            if model != 'kp8' and color_by_fraction_of:
                warnings.warn(f"Option 'color_by_fraction_of' is only effective in 8kp simulations, but {model} results are being used")
            if model == 'kp8' and not color_by_fraction_of:
                color_by_fraction_of = 'conduction_band'  # default
            skip_annotation = False
            for cnt, stateIndex in enumerate(state_indices):
                if model == 'kp8':
                    # color according to spinor compositions
                    if color_by_fraction_of == 'conduction_band':
                        plot_color = scalarmappable.to_rgba(compositions['kp8'][stateIndex, kIndex, 0])
                    elif color_by_fraction_of == 'heavy_hole':
                        plot_color = scalarmappable.to_rgba(compositions['kp8'][stateIndex, kIndex, 1])
                else:
                    # color according to the quantum model that yielded the solution
                    plot_color = self.default_colors.bands[model]
                ax.plot(x, psiSquared[model][cnt][kIndex], color=plot_color)

                if show_state_index:
                    xmax, ymax = CommonShortcuts.get_maximum_points(psiSquared[model][cnt][kIndex], x)
                    if skip_annotation:   # if annotation was skipped in the previous iteration, annotate
                        # ax.annotate(f'n={stateIndex},{stateIndex+1}', xy=(xmax, ymax), xytext=(xmax-0.05*simLength, ymax+0.07))
                        ax.annotate(f'{stateIndex},{stateIndex+1}', xy=(xmax, ymax), xytext=(xmax, ymax+0.07))
                        skip_annotation = False   # wavefunction degeneracy is atmost 2
                    elif cnt < len(state_indices)-1:  # if not the last state
                        xmax_next, ymax_next = CommonShortcuts.get_maximum_points(psiSquared[model][cnt+1][kIndex], x)
                        if abs(xmax_next - xmax) < 1.0 and abs(ymax_next - ymax) < 1e-1:
                            skip_annotation = True
                        else:
                            skip_annotation = False
                            # ax.annotate(f'n={stateIndex+1}', xy=(xmax, ymax), xytext=(xmax-0.05*simLength, ymax+0.07))
                            ax.annotate(f'{stateIndex+1}', xy=(xmax, ymax), xytext=(xmax, ymax+0.07))
                    else:
                        # ax.annotate(f'n={stateIndex+1}', xy=(xmax, ymax), xytext=(xmax-0.05*simLength, ymax+0.07))
                        ax.annotate(f'{stateIndex+1}', xy=(xmax, ymax), xytext=(xmax, ymax+0.07))
            # ax.legend(loc='lower left')
            # ax.legend(loc='upper left')
            ax.legend()

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
                draw_bandedges(ax_probability, model, want_valence_band)


                if model == 'kp8':
                    # define colorbar representing electron fraction
                    divnorm = colors.TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0)
                    scalarmappable = plt.cm.ScalarMappable(cmap='seismic', norm=divnorm)
                    cbar = fig.colorbar(scalarmappable)
                    cbar.set_label("Conduction-band fraction", fontsize=labelsize)
                    cbar.ax.tick_params(labelsize=ticksize)

                draw_probabilities(self, ax_probability, state_indices, model, kIndex, show_state_index, color_by_fraction_of)

                if show_spinor and (model == 'kp6' or model == 'kp8'):
                    draw_spinor_pie_charts(grid_spinor, state_indices, model, stateIndex, kIndex, show_state_index)
                else:
                    fig.tight_layout()


        #-------------------------------------------
        # Plots --- save all the figures to one PDF
        #-------------------------------------------
        if savePDF:
            export_filename = f'{CommonShortcuts.separate_extension(input_file.fullpath)[0]}_probabilities'
            self.export_figs(export_filename, 'pdf')
        if savePNG:
            export_filename = f'{CommonShortcuts.separate_extension(input_file.fullpath)[0]}_probabilities'
            self.export_figs(export_filename, 'png', fig=fig)   # NOTE: presumably only the last fig instance is exported

        # --- display in the GUI
        plt.show()

        return fig


    def plot_RRSWavefunctions(self,
            input_file,
            start_position      = -10000.,
            end_position        = 10000.,
            hide_tails          = False,
            show_state_index    = False,
            want_valence_band   = True,
            color_by_fraction_of = 'conduction_band',
            plot_title          = '',
            labelsize           = CommonShortcuts.labelsize_default,
            ticksize            = CommonShortcuts.ticksize_default,
            savePDF             = False,
            savePNG             = False,
            ):
        """
        Plot the Reduced Real Space modes on top of bandedges.
        
        The Reduced Real Space modes are coloured according to
        electron fraction (for kp8 model).

        Parameters
        ----------
        input_file : nextnanopy.InputFile object
            nextnano++ input file.
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
        want_valence_band : bool, optional
            If True, plot the valence band edges. The default is True.
        color_by_fraction_of : str, optional
            If 8-band k.p simulation, colour the probabilities by the spinor fraction of the specified band. The default is 'conduction_band'.
        plot_title : str, optional
            title of the probability plot. The default is ''.
        labelsize : int, optional
            font size of xlabel, ylabel and colorbar label
        ticksize : int, optional
            font size of xtics, ytics and colorbar tics
        savePDF : str, optional
            save the plot in the PDF format. The default is False.
        savePNG : str, optional
            save the plot in the PNG format. The default is False.

        Returns
        -------
        None.

        """
        if color_by_fraction_of not in ['conduction_band', 'heavy_hole']:
            raise ValueError(f"color_by_fraction_of '{color_by_fraction_of}' is not supported")

        from matplotlib import colors
        
        # load output data files
        datafile_RRSModes = self.get_DataFile(['ReducedRealSpaceModes.dat'], input_file.fullpath)

        x_RRSModes  = datafile_RRSModes.coords['Position'].value
        
        # store data in arrays (independent of quantum models)
        x             = datafile_RRSModes.coords['Position'].value
        CBBandedge    = datafile_RRSModes.variables['ConductionBandEdge'].value
        
        if want_valence_band:
            LHBandedge    = datafile_RRSModes.variables['LightHoleBandEdge'].value
            HHBandedge    = datafile_RRSModes.variables['HeavyHoleBandEdge'].value
            SOBandedge    = datafile_RRSModes.variables['SplitOffHoleBandEdge'].value
            # VBTop         = datafile_RRSModes.variables['ValenceBandTopWithoutStrain'].value
            
        

        # dictionary containing quantum model keys and 2-dimensional list for each key that stores psi^2 for all (eigenstate, kIndex)
        nStates = len(datafile_RRSModes.variables) - 5
        psiSquared = [ 0 for stateIndex in range(nStates) ]
        print("nStates: ", nStates)
        for stateIndex in range(nStates):
            # psiSquared_oldgrid = datafile_RRSModes.variables[f"Psi_{stateIndex+1} (lev.{stateIndex+1} per.0)"].value  # TODO: generalize. nPeriod is not always 1
            lis = list(datafile_RRSModes.variables.keys())  # workaround: get the keys
            psiSquared_oldgrid = datafile_RRSModes.variables[lis[5 + stateIndex]].value # first five data are bandedges
            psiSquared[stateIndex] = CommonShortcuts.convert_grid(psiSquared_oldgrid, x_RRSModes, x)   # grid interpolation needed because of 'output_bandedges{ averaged=no }'


        # chop off edges of the simulation region
        CBBandedge = CommonShortcuts.cutOff_edges1D(CBBandedge, x, start_position, end_position)
        if want_valence_band:
            HHBandedge = CommonShortcuts.cutOff_edges1D(HHBandedge, x, start_position, end_position)
            LHBandedge = CommonShortcuts.cutOff_edges1D(LHBandedge, x, start_position, end_position)
            SOBandedge = CommonShortcuts.cutOff_edges1D(SOBandedge, x, start_position, end_position)
            # VBTop = CommonShortcuts.cutOff_edges1D(VBTop, x, start_position, end_position)

        for stateIndex in range(nStates):
            psiSquared[stateIndex] = CommonShortcuts.cutOff_edges1D(psiSquared[stateIndex], x, start_position, end_position)   # chop off edges of the simulation region

        x = CommonShortcuts.cutOff_edges1D(x, x, start_position, end_position)
        # simLength = x[-1]-x[0]   # (nm)


        # mask psiSquared data where it is flat
        if hide_tails:
            for stateIndex in range(nStates):
                psiSquared[stateIndex] = CommonShortcuts.mask_part_of_array(psiSquared[stateIndex], method='flat', tolerance=1e-2)

        # define plot title
        title = CommonShortcuts.get_plot_title(plot_title)

        def draw_bandedges(ax, model, want_valence_band):
            self.set_plot_labels(ax, 'Position (nm)', 'Energy (eV)', title)
            if model == 'Gamma' or model == 'kp8':
                ax.plot(x, CBBandedge, label='conduction band', linewidth=0.6, color=self.default_colors.bands['CB'])
            if want_valence_band:
                if model == 'HH' or model == 'kp6' or model == 'kp8':
                    ax.plot(x, HHBandedge, label='heavy hole', linewidth=0.6, color=self.default_colors.bands['HH'])
                if model == 'LH' or model == 'kp6' or model == 'kp8':
                    ax.plot(x, LHBandedge, label='light hole', linewidth=0.6, color=self.default_colors.bands['LH'])
                # if model == 'SO' or model == 'kp6' or model == 'kp8':
                #     ax.plot(x, SOBandedge, label='split-off hole', linewidth=0.6, color=self.default_colors.bands['SO'])
                # if model == 'LH' or model == 'kp6' or model == 'kp8':
                #     ax.plot(x, VBTop, label='VB top without strain', linewidth=0.6, color=self.default_colors.bands['LH'])

        def draw_probabilities(self, ax, state_indices, model, show_state_index, color_by_fraction_of):
            if model != 'kp8' and color_by_fraction_of:
                warnings.warn(f"Option 'color_by_fraction_of' is only effective in 8kp simulations, but {model} results are being used")
            if model == 'kp8' and not color_by_fraction_of:
                color_by_fraction_of = 'conduction_band'  # default
            skip_annotation = False
            for cnt, stateIndex in enumerate(state_indices):
                ax.plot(x, psiSquared[stateIndex])

                if show_state_index:
                    xmax, ymax = CommonShortcuts.get_maximum_points(psiSquared[stateIndex], x)
                    if skip_annotation:   # if annotation was skipped in the previous iteration, annotate
                        # ax.annotate(f'n={stateIndex},{stateIndex+1}', xy=(xmax, ymax), xytext=(xmax-0.05*simLength, ymax+0.07))
                        ax.annotate(f'{stateIndex},{stateIndex+1}', xy=(xmax, ymax), xytext=(xmax, ymax+0.07))
                        skip_annotation = False   # wavefunction degeneracy is atmost 2
                    elif cnt < len(state_indices)-1:  # if not the last state
                        xmax_next, ymax_next = CommonShortcuts.get_maximum_points(psiSquared[stateIndex+1], x)
                        if abs(xmax_next - xmax) < 1.0 and abs(ymax_next - ymax) < 1e-1:
                            skip_annotation = True
                        else:
                            skip_annotation = False
                            # ax.annotate(f'n={stateIndex+1}', xy=(xmax, ymax), xytext=(xmax-0.05*simLength, ymax+0.07))
                            ax.annotate(f'{stateIndex+1}', xy=(xmax, ymax), xytext=(xmax, ymax+0.07))
                    else:
                        # ax.annotate(f'n={stateIndex+1}', xy=(xmax, ymax), xytext=(xmax-0.05*simLength, ymax+0.07))
                        ax.annotate(f'{stateIndex+1}', xy=(xmax, ymax), xytext=(xmax, ymax+0.07))
            # ax.legend(loc='lower left')
            ax.legend(loc='upper left')


        # instantiate matplotlib subplot objects for bandedge & probability distribution & spinor pie charts
        fig, ax_probability = plt.subplots()
        model = 'kp8'
        draw_bandedges(ax_probability, model, want_valence_band)
        # ax_probability.set_title(f'{title} (quantum model: {model})', color=self.default_colors.bands[model])


        # if model == 'kp8':
        #     # define colorbar representing electron fraction
        #     divnorm = colors.TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0)
        #     scalarmappable = plt.cm.ScalarMappable(cmap='seismic', norm=divnorm)
        #     cbar = fig.colorbar(scalarmappable)
        #     cbar.set_label("Conduction-band fraction", fontsize=labelsize)
        #     cbar.ax.tick_params(labelsize=ticksize)

        draw_probabilities(self, ax_probability, np.arange(nStates), model, show_state_index, color_by_fraction_of)

        fig.tight_layout()


        #-------------------------------------------
        # Plots --- save all the figures to one PDF
        #-------------------------------------------
        if savePDF:
            export_filename = f'{CommonShortcuts.separate_extension(input_file.fullpath)[0]}_RRSModes'
            self.export_figs(export_filename, 'pdf')
        if savePNG:
            export_filename = f'{CommonShortcuts.separate_extension(input_file.fullpath)[0]}_RRSModes'
            self.export_figs(export_filename, 'png', fig=fig)   # NOTE: presumably only the last fig instance is exported

        # --- display in the GUI
        plt.show()


    def get_DataFile_probabilities_in_folder(self, folder_path, bias=None):
        """
        Get single nextnanopy.DataFile of probability_shift data in the specified folder.

        Parameters
        ----------
        folder_path : str
            output folder path in which the datafile should be sought
        bias : real, optional
            If not None, that bias is used to search for the energy eigenstates output folder. 
            If None, output is sought in the Init folder.
        
        Returns
        -------
        dictionary { quantum model key: corresponding list of nn.DataFile() objects for probability_shift }

        """
        if bias is None:
            # search for the raw solution in Init folder
            datafiles = self.get_DataFiles_NEGFInit_in_folder("EigenStates.dat", folder_path, search_raw_solution_folder=True)
            datafiles = [df for df in datafiles if NEGFShortcuts.SchrodingerRawSolutionFolder in df.fullpath]
            if len(datafiles) > 1: raise RuntimeError("Multiple data files found with keyword 'EigenStates.dat' in 'Init' folder!")
            probability_dict = {'kp8': list(datafiles)}  # TODO: generalize to cover 1,2,3-band cases
        else:
            # search for biased output
            datafile = self.get_DataFile_NEGF_atBias("EigenStates.dat", folder_path, bias=bias)
            logging.info("Taking EigenStates from biased folder")
            assert(isinstance(datafile, nn.DataFile))
            probability_dict = {'kp8': datafile}  # TODO: generalize to cover 1,2,3-band cases

        assert(isinstance(probability_dict, dict))
        return probability_dict



    ############### find ground states from kp8 result ############################
    def find_lowest_conduction_state_atK0(self, output_folder, threshold=0.5):
        """
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
        # get nn.DataFile object
        try:
            datafile = self.get_DataFile_in_folder(['wavefunctions_k000_spinor_composition_AngMom'], output_folder)   # spinor composition at in-plane k = 0
        except FileNotFoundError:
            raise

        # find the lowest conduction band state
        # nextnanopy.DataFile parses the labels in NEGF++ output file wrongly. But datafile.variables contain the data 'cb1 hh1 lh1 so1 cb2 hh2 lh2 so2 sum' for index 0-8
        num_evs = len(datafile.variables[0].value)
        # print(f"datafile coords {datafile.coords}")
        # print(f"datafile variables {datafile.variables}")
        for stateIndex in range(num_evs):
            electronFraction = datafile.variables[0].value[stateIndex] + datafile.variables[4].value[stateIndex]
            if electronFraction > threshold:
                logging.info(f"Lowest electron index = {stateIndex} because electronFraction = {electronFraction}")
                return stateIndex

        raise RuntimeError(f"No electron states found in: {output_folder}")


    def find_highest_valence_state_atK0(self, output_folder, threshold=0.5):
        """
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
        # get nn.DataFile object
        try:
            datafile = self.get_DataFile_in_folder(['wavefunctions_k000_spinor_composition_AngMom'], output_folder)   # spinor composition at in-plane k = 0
        except FileNotFoundError:
            raise

        # find the highest valence band state
        # nextnanopy.DataFile parses the labels in NEGF++ output file wrongly. But datafile.variables contain the data 'cb1 hh1 lh1 so1 cb2 hh2 lh2 so2 sum' for index 0-8
        # print(datafile.coords)
        # print(f"find_highest_valence_state: datafile.variables\n{datafile.variables}")
        num_evs = len(datafile.variables[0].value)
        for stateIndex in reversed(range(num_evs)):
            electronFraction = datafile.variables[0].value[stateIndex] + datafile.variables[4].value[stateIndex]
            if electronFraction < threshold:
                logging.info(f"Highest hole index = {stateIndex} because electronFraction = {electronFraction}")
                return stateIndex

        raise RuntimeError(f"No hole states found in: {output_folder}")


    def find_highest_HH_state_atK0(self, output_folder, threshold=0.5, want_second_highest=False):
        """
            From spinor composition data, determine the highest heavy-hole state in an 8-band k.p simulation.
            This method should be able to detect it properly even when the effective bandgap is negative.
            
            Parameters
            ----------
            output_folder : str
                output folder path
            threshold : real, optional
                If electron fraction in the spinor composition is less than this value, the state is assumed to be a hole state.
                The default is 0.5.
            want_second_highest : bool
                If True, return the index of the second highest heavy-hole state.

            Returns
            -------
            state index (base 0) of the highest heavy-hole state at in-plane k = 0

        """
        # get nn.DataFile object
        try:
            datafile = self.get_DataFile_in_folder(['wavefunctions_k000_spinor_composition_AngMom'], output_folder)   # spinor composition at in-plane k = 0
        except FileNotFoundError:
            raise

        # find the highest heavy-hole state
        # nextnanopy.DataFile parses the labels in NEGF++ output file wrongly. But datafile.variables contain the data 'cb1 hh1 lh1 so1 cb2 hh2 lh2 so2 sum' for index 0-8
        firstStateIndex = None
        secondStateIndex = None
        num_evs = len(datafile.variables[0].value)
        for stateIndex in reversed(range(num_evs)):
            HHFraction = datafile.variables[1].value[stateIndex] + datafile.variables[5].value[stateIndex]
            if HHFraction > threshold:
                logging.info(f"Highest heavy-hole index = {stateIndex} (heavy-hole contribution = {HHFraction})")
                if firstStateIndex is None:
                    firstStateIndex = stateIndex
                elif want_second_highest and (secondStateIndex is None):
                    secondStateIndex = stateIndex
                else:  # states have been found
                    if want_second_highest:
                        return firstStateIndex, secondStateIndex
                    else:
                        return firstStateIndex

        if firstStateIndex is None:
            raise RuntimeError(f"No heavy-hole states found in: {output_folder}")
        elif want_second_highest and (secondStateIndex is None):
            raise RuntimeError(f"Second heavy-hole state wasn't found in: {output_folder}")
    

    def find_highest_LH_state_atK0(self, output_folder, threshold=0.5):
        """
            From spinor composition data, determine the highest light-hole state in an 8-band k.p simulation.
            This method should be able to detect it properly even when the effective bandgap is negative.
            
            Parameters
            ----------
            output_folder : str
                output folder path
            threshold : real, optional
                If electron fraction in the spinor composition is less than this value, the state is assumed to be a hole state.
                The default is 0.5.

            Returns
            -------
            state index (base 0) of the highest light-hole state at in-plane k = 0

        """
        # get nn.DataFile object
        try:
            datafile = self.get_DataFile_in_folder(['wavefunctions_k000_spinor_composition_AngMom'], output_folder)   # spinor composition at in-plane k = 0
        except FileNotFoundError:
            raise

        # find the highest light-hole state
        # nextnanopy.DataFile parses the labels in NEGF++ output file wrongly. But datafile.variables contain the data 'cb1 hh1 lh1 so1 cb2 hh2 lh2 so2 sum' for index 0-8
        num_evs = len(datafile.variables[0].value)
        # print(datafile.coords)
        # print(datafile.variables)
        for stateIndex in reversed(range(num_evs)):
            LHFraction = datafile.variables[2].value[stateIndex] + datafile.variables[6].value[stateIndex]
            if LHFraction > threshold:
                logging.info(f"Highest light-hole index = {stateIndex} (light-hole contribution = {LHFraction})")
                return stateIndex

        raise RuntimeError(f"No light-hole states found in: {output_folder}")
    

    ################ optics analysis #############################################
    kp8_basis = ['cb1', 'cb2', 'hh1', 'hh2', 'lh1', 'lh2', 'so1', 'so2']
    
    def calculate_overlap(self, output_folder, force_lightHole=False):
        """
        Calculate envelope overlap (complex absolute value |z|) between the lowest electron and highest valence band states from wavefunction output data

        Parameters
        ----------
        output_folder : str
            Absolute path of output folder that contains the amplitude data
        force_lightHole : bool, optional
            If True, always use the overlap of Gamma and light-hole states. Effective only for single-band models.
            Default is False
            TODO: To be implemented

        Returns
        -------
        overlap : np.ndarray with dtype=cdouble
            Spatial overlap of envelope functions

        Requires
        --------
        SciPy package installation

        """
        from scipy.integrate import simps

        iLowestElectron = self.find_lowest_conduction_state_atK0(output_folder, threshold=0.5)
        iHighestHole    = self.find_highest_valence_state_atK0(output_folder, threshold=0.5)

        # load amplitude data
        datafile_amplitude_at_k0 = self.get_DataFile_amplitudesK0_in_folder(output_folder)   # returns a dict of nn.DataFile
        
        # extract amplitude of electron- and hole-like states
        x = datafile_amplitude_at_k0['kp8'][0].coords['Position'].value  # real space coords are common for all amplitude outputs
        amplitude_e = np.zeros((len(self.kp8_basis), len(x)), dtype=np.cdouble)  # psi of lowest CB state
        amplitude_h = np.zeros((len(self.kp8_basis), len(x)), dtype=np.cdouble)  # psi of highest VB state
        for iBand, band in enumerate(self.kp8_basis):
            for df in datafile_amplitude_at_k0['kp8']:
                if band in os.path.split(df.fullpath)[1]:  # if the band name is in the output filename
                    amplitude_e[iBand,:].real = df.variables[f'Psi_{iLowestElectron+1}_real'].value
                    amplitude_e[iBand,:].imag = df.variables[f'Psi_{iLowestElectron+1}_imag'].value
                    amplitude_h[iBand,:].real = df.variables[f'Psi_{iHighestHole+1}_real'].value
                    amplitude_h[iBand,:].imag = df.variables[f'Psi_{iHighestHole+1}_imag'].value

        overlap = 0.

        # calculate overlap
        for mu in range(len(self.kp8_basis)):
            for nu in range(len(self.kp8_basis)):
                prod = np.multiply(np.conjugate(amplitude_h[nu,:]), amplitude_e[mu,:])   # multiply arguments element-wise (NOT inner product of spinor vector!)
                overlap += simps(prod, x)
        # for mu in range(len(self.kp8_basis)):
        #     prod = np.multiply(np.conjugate(amplitude_e[mu,:]), amplitude_e[mu,:])   # inner product of spinor vector
        #     overlap += simps(prod, x)
        return overlap


    ################ Energy getters #############################################
    def get_transition_energy(self, output_folder):
        """
        Get the transition energy = energy separation between the lowest electron and highest valence band states.
        Unit: eV
        """
        E_electron = self.__get_lowest_electron_energy(output_folder)
        E_hole = self.__get_highest_hole_energy(output_folder)

        return E_electron - E_hole
    
    
    def get_HH1_LH1_energy_difference(self, output_folder):
        """
        Get the energy separation between the highest HH and highest LH states.
        Unit: eV
        """
        E_HH = self.__get_highest_HH_energy(output_folder, False)
        E_LH = self.__get_highest_LH_energy(output_folder)

        return E_HH - E_LH 
    

    def get_HH1_HH2_energy_difference(self, output_folder):
        """
        Get the energy separation between the highest HH and second highest HH states.
        Unit: eV
        """
        E_HH1, E_HH2 = self.__get_highest_HH_energy(output_folder, True)

        return E_HH1 - E_HH2 
    

    def __get_lowest_electron_energy(self, output_folder):
        """
        Unit: eV
        """
        datafile = self.get_DataFile_in_folder(["EnergySpectrum"], output_folder)
        iLowestElectron = self.find_lowest_conduction_state_atK0(output_folder, threshold=0.5)
        return datafile.variables[0].value[iLowestElectron]
    

    def __get_highest_hole_energy(self, output_folder):
        """
        Unit: eV
        """
        datafile = self.get_DataFile_in_folder(["EnergySpectrum"], output_folder)
        iHighestHole = self.find_highest_valence_state_atK0(output_folder, threshold=0.5)
        return datafile.variables[0].value[iHighestHole]


    def __get_highest_HH_energy(self, output_folder, want_second_highest):
        """
        Unit: eV
        """
        datafile = self.get_DataFile_in_folder(["EnergySpectrum"], output_folder)
        if want_second_highest:
            iFirstHighestHH, iSecondHighestHH = self.find_highest_HH_state_atK0(output_folder, threshold=0.5, want_second_highest=want_second_highest)
            return datafile.variables[0].value[iFirstHighestHH], datafile.variables[0].value[iSecondHighestHH]
        else:
            iHighestHH = self.find_highest_HH_state_atK0(output_folder, threshold=0.5, want_second_highest=want_second_highest)
            return datafile.variables[0].value[iHighestHH]


    def __get_highest_LH_energy(self, output_folder):
        """
        Unit: eV
        """
        datafile = self.get_DataFile_in_folder(["EnergySpectrum"], output_folder)
        iHighestLH = self.find_highest_LH_state_atK0(output_folder, threshold=0.5)
        return datafile.variables[0].value[iHighestLH]

