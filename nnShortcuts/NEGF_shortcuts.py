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

# nextnanopy includes
import nextnanopy as nn
from nnShortcuts.common import CommonShortcuts


class NEGFShortcuts(CommonShortcuts):
    # nextnano solver
    software = 'nextnano.NEGF'

    def get_IV(self, input_file_name):
        """
        Get I-V curve.
        OUTPUT: 2 nn.DataFile() attributes for current & voltage
        """
        datafile = self.getDataFile('Current_vs_Voltage', input_file_name)
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
        outputFolder = nn.config.get(self.software, 'outputdirectory')
        filename_no_extension = super().separateFileExtension(input_file_name)[0]
        outputSubfolder = os.path.join(outputFolder, filename_no_extension)
        self.export_figs("IV", "png", output_folder_path=outputSubfolder, fig=fig)



    def getDataFile_NEGF_init(self, keywords, name):
        output_folder = nn.config.get(self.software, 'outputdirectory')
        filename_no_extension = super().separateFileExtension(name)[0]
        subfolder = os.path.join(output_folder, filename_no_extension)
        d = nn.DataFolder(subfolder)

        # Get the fullpath of 'Init' subfolder
        for folder_name in d.folders.keys():
            if 'Init' in folder_name:
                # listOfFiles = d.go_to(folder_name).find(keyword, deep=True)
                init_folder_path = d.go_to(folder_name).fullpath

        return self.getDataFile_in_folder(keywords, init_folder_path)  # TODO: add options available



    def getDataFile_NEGF_atBias(self, keywords, name, bias):
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
        output_folder = nn.config.get(self.software, 'outputdirectory')
        filename_no_extension = super().separateFileExtension(name)[0]
        bias_subfolder = os.path.join(output_folder, filename_no_extension, str(bias) + 'mv')

        return self.getDataFile_in_folder(keywords, bias_subfolder)


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
    #     output_folder = nn.config.get(self.software, 'outputdirectory')
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
            nn.InputFile() object
            bias value

        RETURN: nn.DataFile() attributes
            datafile.coords['Position']
            datafile.variables['Conduction_BandEdge']
        """
        # datafile = getDataFile_NEGF_atBias('WannierStark_states.dat', input_file_name, bias)
        try:
            datafile = self.getDataFile_NEGF_atBias('Conduction_BandEdge.dat', input_file_name, bias)
        except FileNotFoundError:
            try:
                datafile = self.getDataFile_NEGF_atBias('ConductionBandEdge.dat', input_file_name, bias)
            except FileNotFoundError:
                raise

        position = datafile.coords['Position']
        # conduction_bandedge = datafile.variables['Conduction_BandEdge']
        conduction_bandedge = datafile.variables['Conduction Band Edge']
        return position, conduction_bandedge


    def get_WannierStarkStates_init(self, name):
        """
        RETURN: nn.DataFile() attribute
            datafile.coords['Position']
            datafile.variables['Conduction BandEdge']
            datafile.variables['Psi_*']
        """
        datafile = self.getDataFile_NEGF_init('WannierStark_states.dat', name)

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
        datafile = self.getDataFile_NEGF_atBias('WannierStark_states.dat', input_file, bias)

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
            name, 
            start_position = -10000., 
            end_position   = 10000., 
            labelsize      = None, 
            ticksize       = None
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
        if labelsize is None: labelsize = self.labelsize_default
        if ticksize is None: ticksize = self.ticksize_default

        position, CB, Psi_squareds = self.get_WannierStarkStates_init(name)

        # Store data arrays.
        # Cut off edges of the simulation region if needed.
        conduction_bandedge = super().cutOff_edges1D(CB.value, position.value, start_position, end_position)
        WS_states           = [super().cutOff_edges1D(Psi_squared.value, position.value, start_position, end_position) for Psi_squared in Psi_squareds]
        x                   = super().cutOff_edges1D(position.value, position.value, start_position, end_position)

        WS_states = [super().mask_part_of_array(WS_state) for WS_state in WS_states]   # hide flat tails

        fig, ax = plt.subplots()
        ax.set_xlabel(position.label, fontsize=labelsize)
        ax.set_ylabel("Energy (eV)", fontsize=labelsize)
        ax.set_title('Wannier-Stark states', fontsize=labelsize)
        ax.tick_params(axis='x', labelsize=ticksize)
        ax.tick_params(axis='y', labelsize=ticksize)
        ax.plot(x, conduction_bandedge, color='red', linewidth=0.7, label=CB.label)
        for WS_state in WS_states:
            ax.plot(x, WS_state, label='')   # TODO: label
        fig.tight_layout()

        # export to an image file
        outputFolder = nn.config.get(self.software, 'outputdirectory')
        filename_no_extension = super().separateFileExtension(name)[0]
        outputSubfolder = os.path.join(outputFolder, filename_no_extension)
        self.export_figs("WannierStarkStates_init", "png", output_folder_path=outputSubfolder, fig=fig)

        return fig


    def get_2Ddata_atBias(self, input_file_name, bias, data='carrier'):
        """
        INPUT: one of the following strings: ['LDOS', 'carrier', 'current']

        RETURN: nn.DataFile() attributes
            x = datafile.coords['x']
            y = datafile.coords['y']
            z = datafile.variables[variableKey]
        """
        if data == 'LDOS' or data == 'DOS':
            file = 'DOS_energy_resolved.vtr'
            variableKey = 'Density of states'
        elif data == 'carrier':
            file = 'CarrierDensity_energy_resolved.vtr'
            variableKey = 'Carrier density'
        elif data == 'current':
            file = 'CurrentDensity_energy_resolved.vtr'
            variableKey = 'Current Density'
        else:
            raise KeyError(f'Illegal data {data} requested!')

        datafile = self.getDataFile_NEGF_atBias(file, input_file_name, bias)

        x = datafile.coords['x']
        y = datafile.coords['y']
        quantity = datafile.variables[variableKey]
        return x, y, quantity


    def plot_DOS(self,
            input_file_name, 
            bias, 
            labelsize=None,
            ticksize=None
            ):
        """
        Overlay bandedge with local density of states. Loads the following output data:
        DOS_energy_resolved.vtr

        The plot is saved as an png image file.
        """
        if labelsize is None: labelsize = self.labelsize_default
        if ticksize is None: ticksize = self.ticksize_default

        position, CB = self.get_conduction_bandedge(input_file_name, bias)
        x, y, quantity = self.get_2Ddata_atBias(input_file_name, bias, 'LDOS')

        print("Plotting DOS...")
        unit = r'$\mathrm{nm}^{-1} \mathrm{eV}^{-1}$'
        label = 'Density of states (' + unit + ')'

        fig, ax = plt.subplots()
        pcolor = ax.pcolormesh(x.value, y.value, quantity.value.T)
        cbar = fig.colorbar(pcolor)
        cbar.set_label(label, fontsize=labelsize)
        cbar.ax.tick_params(labelsize=ticksize * 0.9)

        ax.set_xlabel(position.label, fontsize=labelsize)
        ax.set_ylabel("Energy (eV)", fontsize=labelsize)
        ax.set_ylim(np.amin(y.value), np.amax(y.value))
        ax.set_title(f'bias={bias}mV', fontsize=labelsize)
        ax.tick_params(axis='x', labelsize=ticksize)
        ax.tick_params(axis='y', labelsize=ticksize)

        ax.plot(position.value, CB.value, color='white', linewidth=0.7, label=CB.label)
        fig.tight_layout()

        # export to an image file
        outputFolder = nn.config.get(self.software, 'outputdirectory')
        filename_no_extension = super().separateFileExtension(input_file_name)[0]
        outputSubfolder = os.path.join(outputFolder, filename_no_extension)
        self.export_figs("DOS", "png", output_folder_path=outputSubfolder, fig=fig)

        return fig


    def plot_carrier_density(self,
            input_file_name, 
            bias, 
            labelsize=None, 
            ticksize=None
            ):
        """
        Overlay bandedge with energy-resolved carrier density. Loads the following output data:
        CarrierDensity_energy_resolved.vtr

        The plot is saved as an png image file.
        """
        if labelsize is None: labelsize = self.labelsize_default
        if ticksize is None: ticksize = self.ticksize_default

        position, CB = self.get_conduction_bandedge(input_file_name, bias)
        x, y, quantity = self.get_2Ddata_atBias(input_file_name, bias, 'carrier')

        print("Plotting carrier density...")
        unit = r'$\mathrm{cm}^{-3} \mathrm{eV}^{-1}$'
        label = 'Carrier density (' + unit + ')'

        fig, ax = plt.subplots()
        pcolor = ax.pcolormesh(x.value, y.value, quantity.value.T)
        cbar = fig.colorbar(pcolor)
        cbar.set_label(label, fontsize=labelsize)
        cbar.ax.tick_params(labelsize=ticksize * 0.9)

        ax.set_xlabel(position.label, fontsize=labelsize)
        ax.set_ylabel("Energy (eV)", fontsize=labelsize)
        ax.set_ylim(np.amin(y.value), np.amax(y.value))
        ax.set_title(f'bias={bias}mV', fontsize=labelsize)
        ax.tick_params(axis='x', labelsize=ticksize)
        ax.tick_params(axis='y', labelsize=ticksize)

        ax.plot(position.value, CB.value, color='white', linewidth=0.7, label=CB.label)
        fig.tight_layout()

        # export to an image file
        outputFolder = nn.config.get(self.software, 'outputdirectory')
        filename_no_extension = super().separateFileExtension(input_file_name)[0]
        outputSubfolder = os.path.join(outputFolder, filename_no_extension)
        self.export_figs("CarrierDensity", "png", output_folder_path=outputSubfolder, fig=fig)

        return fig


    def plot_current_density(self,
            input_file_name, 
            bias, 
            labelsize=None, 
            ticksize=None
            ):
        """
        Overlay bandedge with energy-resolved current density. Loads the following output data:
        CurrentDensity_energy_resolved.vtr

        The plot is saved as an png image file.
        """
        if labelsize is None: labelsize = self.labelsize_default
        if ticksize is None: ticksize = self.ticksize_default

        position, CB = self.get_conduction_bandedge(input_file_name, bias)
        x, y, quantity = self.get_2Ddata_atBias(input_file_name, bias, 'current')

        print("Plotting current density...")
        unit = r'$\mathrm{A}$ $\mathrm{cm}^{-2} \mathrm{eV}^{-1}$'
        label = 'Current density (' + unit + ')'

        fig, ax = plt.subplots()
        pcolor = ax.pcolormesh(x.value, y.value, quantity.value.T)
        cbar = fig.colorbar(pcolor)
        cbar.set_label(label, fontsize=labelsize)
        cbar.ax.tick_params(labelsize=ticksize * 0.9)

        ax.set_xlabel(position.label, fontsize=labelsize)
        ax.set_ylabel("Energy (eV)", fontsize=labelsize)
        ax.set_ylim(np.amin(y.value), np.amax(y.value))
        ax.set_title(f'bias={bias}mV', fontsize=labelsize)
        ax.tick_params(axis='x', labelsize=ticksize)
        ax.tick_params(axis='y', labelsize=ticksize)

        ax.plot(position.value, CB.value, color='white', linewidth=0.7, label=CB.label)
        fig.tight_layout()

        # export to an image file
        outputFolder = nn.config.get(self.software, 'outputdirectory')
        filename_no_extension = super().separateFileExtension(input_file_name)[0]
        outputSubfolder = os.path.join(outputFolder, filename_no_extension)
        self.export_figs("CurrentDensity", "png", output_folder_path=outputSubfolder, fig=fig)

        return fig


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
    #     # datafile = self.getDataFile(f'Gain_SelfConsistent_vs_{xaxis}.dat', input_file_name)
    #     datafile = self.getDataFile('Gain_vs_Voltage', input_file_name)
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
        output_folder = nn.config.get(self.software, 'outputdirectory')
        filename_no_extension = super().separateFileExtension(name)[0]
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

        input_file_name = super().separateFileExtension(input_file_name)[0]

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
        cax = ax.pcolormesh(x.value, y.value, F[:-1, :-1, 0], vmin=-1, vmax=1, cmap='viridis')
        cbar = fig.colorbar(cax)
        cbar.set_label(label)


        def animate_2D_plots(bias):
            # update plot title
            ax.set_title('{}, bias={:.1f} mV'.format(input_file_name, array_of_biases[i]))
            # update 2D color plot
            cax.set_array(F[:-1, :-1, i].flatten())
            # update conduction bandedge plot
            ax.plot(x.value, CB.value, color='white', linewidth=0.7, label=CB.label)


        # ax.legend(loc='upper left')

        # generate GIF animation
        from matplotlib import animation

        cwd = os.getcwd()
        # os.chdir(output_folder_path)
        print(f'Exporting GIF animation to: {os.getcwd()}\n')
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
        outputFolder = nn.config.get(self.software, 'outputdirectory')
        filename_no_extension = super().separateFileExtension(name)[0]
        outputSubfolder = os.path.join(outputFolder, filename_no_extension)
        df = self.getDataFile_in_folder('L-I-V.dat', outputSubfolder)
        
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
            labelsize=None, 
            ticksize=None, 
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
        if labelsize is None: labelsize = self.labelsize_default
        if ticksize is None: ticksize = self.ticksize_default

        # validate arguments
        if len(names) != len(labels): 
            raise self.NextnanopyScriptError(f"Number of input files ({len(names)}) do not match that of plot labels ({len(labels)})")

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
            datafile_IV = self.getDataFile('Current_vs_Voltage.dat', name)
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


        print("\nPlotting L-I-V curves...")
        from matplotlib.ticker import MultipleLocator

        fig, ax1 = plt.subplots()

        linetypes = ['solid', 'dashed', 'dotted', 'dashdot']

        # I-V curves
        color = 'tab:blue'
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
        color = 'tab:red'
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


    ############### plot probabilities ##################################
    def plot_probabilities(self,
            input_file,
            states_range_dict   = None,
            states_list_dict    = None,
            start_position      = -10000.,
            end_position        = 10000.,
            hide_tails          = False,
            only_k0             = True,
            show_spinor         = False,
            show_state_index    = False,
            plot_title          = '',
            labelsize           = None,
            ticksize            = None,
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
        if labelsize is None: labelsize = self.labelsize_default
        if ticksize is None: ticksize = self.ticksize_default

        from matplotlib import colors
        from matplotlib.gridspec import GridSpec

        # load output data files
        datafile_bandedge = self.getDataFile('BandEdges', input_file.fullpath)
        datafiles_probability_dict = self.getDataFile_probabilities_with_name(input_file.fullpath)

        for model, datafiles in datafiles_probability_dict.items():
            if len(datafiles) == 0: continue

            datafile_probability = datafiles[0]
            x_probability  = datafile_probability.coords['Position'].value
        if not datafile_probability:
            raise self.NextnanoInputFileError('Probabilities are not output! Modify the input file.')


        # store data in arrays (independent of quantum models)
        x             = datafile_bandedge.coords['Position'].value
        CBBandedge    = datafile_bandedge.variables['Conduction band edge with strain'].value
        LHBandedge    = datafile_bandedge.variables['LH band edge with strain'].value
        # HHBandedge    = datafile_bandedge.variables['HH'].value  # TODO: C++ code needs improvement
        SOBandedge    = datafile_bandedge.variables['SO band edge with strain'].value

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

        for model, dfs in datafiles_probability_dict.items():
            if len(dfs) == 0: continue

            for cnt, stateIndex in enumerate(states_toBePlotted[model]):
                for kIndex in range(num_kPoints[model]):
                    psiSquared_oldgrid = dfs[kIndex].variables[f"Psi^2_{stateIndex+1} (lev.{stateIndex+1} per.0)"].value  # TODO: generalize. nPeriod is not always 1
                    psiSquared[model][cnt][kIndex] = super().convert_grid(psiSquared_oldgrid, x_probability, x)   # grid interpolation needed because of 'output_bandedges{ averaged=no }'


        # chop off edges of the simulation region
        CBBandedge = super().cutOff_edges1D(CBBandedge, x, start_position, end_position)
        # HHBandedge = super().cutOff_edges1D(HHBandedge, x, start_position, end_position)  # TODO: C++ code needs improvement
        LHBandedge = super().cutOff_edges1D(LHBandedge, x, start_position, end_position)
        SOBandedge = super().cutOff_edges1D(SOBandedge, x, start_position, end_position)


        for model in states_toBePlotted:
            for cnt, stateIndex in enumerate(states_toBePlotted[model]):
                for kIndex in range(num_kPoints[model]):
                    psiSquared[model][cnt][kIndex] = super().cutOff_edges1D(psiSquared[model][cnt][kIndex], x, start_position, end_position)   # chop off edges of the simulation region

        x = super().cutOff_edges1D(x, x, start_position, end_position)
        simLength = x[-1]-x[0]   # (nm)


        # mask psiSquared data where it is flat
        if hide_tails:
            for model in states_toBePlotted:
                for cnt, stateIndex in enumerate(states_toBePlotted[model]):
                    for kIndex in range(num_kPoints[model]):
                        psiSquared[model][cnt][kIndex] = super().mask_part_of_array(psiSquared[model][cnt][kIndex], method='flat', tolerance=1e-3)


        if 'kp6' in datafiles_probability_dict.keys() or 'kp8' in datafiles_probability_dict.keys():
            # output data of spinor composition at all in-plane k
            datafiles_spinor = {
                'kp6': list(),
                'kp8': list()
            }
            datafiles = self.getDataFiles(['spinor_composition'], input_file.fullpath)
            # datafiles = [df for cnt in range(len(datafiles)) for df in datafiles if str(cnt).zfill(5) + '_CbHhLhSo' in os.path.split(df.fullpath)[1]]   # sort spinor composition datafiles in ascending kIndex  # TODO: C++ doesn't have multiple in-plane k output
            for df in datafiles:
                quantum_model = 'kp8'
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
                if model not in ['kp6', 'kp8']: continue

                compositions[model] = np.zeros((num_evs[model], num_kPoints[model], 4))   # compositions[quantum model][eigenvalue index][k index][spinor index]

                for stateIndex in state_indices:
                    for kIndex in range(num_kPoints[model]):
                        # store spinor composition data
                        if model == 'kp8':
                            compositions[model][stateIndex, kIndex, 0] = datafiles_spinor[model][kIndex].variables['s1'].value[stateIndex] + datafiles_spinor[model][kIndex].variables['s2'].value[stateIndex]

                        compositions[model][stateIndex, kIndex, 1] = datafiles_spinor[model][kIndex].variables['x1'].value[stateIndex] + datafiles_spinor[model][kIndex].variables['x2'].value[stateIndex]
                        compositions[model][stateIndex, kIndex, 2] = datafiles_spinor[model][kIndex].variables['y1'].value[stateIndex] + datafiles_spinor[model][kIndex].variables['y2'].value[stateIndex]
                        compositions[model][stateIndex, kIndex, 3] = datafiles_spinor[model][kIndex].variables['z1'].value[stateIndex] + datafiles_spinor[model][kIndex].variables['z2'].value[stateIndex]

        # define plot title
        title = self.getPlotTitle(plot_title)

        def draw_bandedges(ax, model):
            self.set_plot_labels(ax, 'Position (nm)', 'Energy (eV)', title)
            if model == 'Gamma' or model == 'kp8':
                ax.plot(x, CBBandedge, label='CB', linewidth=0.6, color=self.band_colors['CB'])
            # TODO: C++ code needs improvement
            # if model == 'HH' or model == 'kp6' or model == 'kp8':
            #     ax.plot(x, HHBandedge, label='HH', linewidth=0.6, color=self.band_colors['HH'])
            if model == 'LH' or model == 'kp6' or model == 'kp8':
                ax.plot(x, LHBandedge, label='LH', linewidth=0.6, color=self.band_colors['LH'])
            if model == 'SO' or model == 'kp6' or model == 'kp8':
                ax.plot(x, SOBandedge, label='SO', linewidth=0.6, color=self.band_colors['SO'])

        def draw_probabilities(ax, state_indices, model, kIndex, show_state_index):
            skip_annotation = False
            for cnt, stateIndex in enumerate(state_indices):
                if model == 'kp8':
                    # color according to electron fraction of the state
                    plot_color = scalarmappable.to_rgba(compositions['kp8'][stateIndex, kIndex, 0])
                else:
                    # color according to the quantum model that yielded the solution
                    plot_color = self.band_colors[model]
                ax.plot(x, psiSquared[model][cnt][kIndex], color=plot_color)

                if show_state_index:
                    xmax, ymax = self.get_maximum_points(psiSquared[model][cnt][kIndex], x)
                    if skip_annotation:   # if annotation was skipped in the previous iteration, annotate
                        # ax.annotate(f'n={stateIndex},{stateIndex+1}', xy=(xmax, ymax), xytext=(xmax-0.05*simLength, ymax+0.07))
                        ax.annotate(f'{stateIndex},{stateIndex+1}', xy=(xmax, ymax), xytext=(xmax, ymax+0.07))
                        skip_annotation = False   # wavefunction degeneracy is atmost 2
                    elif cnt < len(state_indices)-1:  # if not the last state
                        xmax_next, ymax_next = self.get_maximum_points(psiSquared[model][cnt+1][kIndex], x)
                        if abs(xmax_next - xmax) < 1.0 and abs(ymax_next - ymax) < 1e-1:
                            skip_annotation = True
                        else:
                            skip_annotation = False
                            # ax.annotate(f'n={stateIndex+1}', xy=(xmax, ymax), xytext=(xmax-0.05*simLength, ymax+0.07))
                            ax.annotate(f'{stateIndex+1}', xy=(xmax, ymax), xytext=(xmax, ymax+0.07))
                    else:
                        # ax.annotate(f'n={stateIndex+1}', xy=(xmax, ymax), xytext=(xmax-0.05*simLength, ymax+0.07))
                        ax.annotate(f'{stateIndex+1}', xy=(xmax, ymax), xytext=(xmax, ymax+0.07))
            ax.legend()

        def draw_spinor_pie_charts(gs_spinor, state_indices, model, stateIndex, kIndex, show_state_index):
            num_rows, num_columns = self.getRowColumnForDisplay(len(state_indices))  # determine arrangement of spinor composition plots
            list_of_colors = [self.band_colors[model] for model in ['CB', 'HH', 'LH', 'SO']]
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
                    num_rows, num_columns = self.getRowColumnForDisplay(len(state_indices))

                    fig = plt.figure(figsize=plt.figaspect(0.4))
                    grid_probability = GridSpec(1, 1, figure=fig, left=0.10, right=0.48, bottom=0.15, wspace=0.05)
                    grid_spinor = GridSpec(num_rows, num_columns, figure=fig, left=0.55, right=0.98, bottom=0.15, hspace=0.2)
                    ax_probability = fig.add_subplot(grid_probability[0,0])
                else:
                    fig, ax_probability = plt.subplots()
                if only_k0:
                    ax_probability.set_title(f'{title} (quantum model: {model})', color=self.band_colors[model])
                else:
                    ax_probability.set_title(f'{title} (quantum model: {model}), k index: {kIndex}', color=self.band_colors[model])
                draw_bandedges(ax_probability, model)


                if model == 'kp8':
                    # define colorbar representing electron fraction
                    divnorm = colors.TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0)
                    scalarmappable = plt.cm.ScalarMappable(cmap='seismic', norm=divnorm)
                    cbar = fig.colorbar(scalarmappable)
                    cbar.set_label("Electron fraction", fontsize=labelsize)
                    cbar.ax.tick_params(labelsize=ticksize)

                draw_probabilities(ax_probability, state_indices, model, kIndex, show_state_index)

                if show_spinor and (model == 'kp6' or model == 'kp8'):
                    draw_spinor_pie_charts(grid_spinor, state_indices, model, stateIndex, kIndex, show_state_index)
                else:
                    fig.tight_layout()


        # not needed for NEGF
        # if both electrons and holes have been calculated separately (but not by kp8),
        # plot bandedge and probabilities at k|| = 0
        # calculated_e_models = [model for model in states_toBePlotted if model in model_names_conduction]
        # calculated_h_models = [model for model in states_toBePlotted if model in model_names_valence]

        # models = [model in model_names_conduction for model in states_toBePlotted]
        # models.append([model in model_names_conduction for model in states_toBePlotted])

        # if len(calculated_e_models) >= 1 and len(calculated_h_models) >= 1:
        #     fig, ax_combi = plt.subplots()
        #     ax_combi.set_title(f"{title} ({calculated_e_models}+{calculated_h_models}), zone-center")
        #     draw_bandedges(ax_combi, 'Gamma')
        #     draw_bandedges(ax_combi, 'HH')
        #     draw_bandedges(ax_combi, 'LH')
        #     draw_bandedges(ax_combi, 'SO')

        #     for model in calculated_e_models + calculated_h_models:
        #         draw_probabilities(ax_combi, states_toBePlotted[model], model, 0, show_state_index)
        #     fig.tight_layout()


        #-------------------------------------------
        # Plots --- save all the figures to one PDF
        #-------------------------------------------
        if savePDF:
            export_filename = f'{super().separateFileExtension(input_file.fullpath)[0]}_probabilities'
            self.export_figs(export_filename, 'pdf')
        if savePNG:
            export_filename = f'{super().separateFileExtension(input_file.fullpath)[0]}_probabilities'
            self.export_figs(export_filename, 'png', fig=fig)   # NOTE: presumably only the last fig instance is exported

        # --- display in the GUI
        plt.show()


    def getDataFile_probabilities_in_folder(self, folder_path):
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
        datafiles = self.getDataFiles_in_folder("probabilities_shifted", folder_path)

        if len(datafiles) > 1: raise RuntimeError("Multiple data files found with keyword 'Probabilities_selected'!")
        probability_dict = {'kp8': list(datafiles)} # currently, wavefunctions are output only in 8-band models
        return probability_dict



    ############### find ground states from kp8 result ############################
    def find_lowest_electron_state_atK0(self, output_folder, threshold=0.5):
        """
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
        # get nn.DataFile object
        try:
            datafile = self.getDataFile_in_folder(['spinor'], output_folder)   # spinor composition at in-plane k = 0
        except FileNotFoundError:
            warnings.warn("Spinor components output in CbHhLhSo basis is not found. Assuming decoupling of the conduction and valence bands...", category=self.NextnanoInputFileWarning)
            return int(0)

        # find the lowest electron state
        num_evs = len(datafile.variables['s1'].value)
        for stateIndex in range(num_evs):
            electronFraction = datafile.variables['s1'].value[stateIndex] + datafile.variables['s2'].value[stateIndex]
            if electronFraction > threshold:
                return stateIndex

        raise RuntimeError(f"No electron states found in: {output_folder}")


    def find_highest_hole_state_atK0(self, output_folder, threshold=0.5):
        """
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
        # get nn.DataFile object
        try:
            datafile = self.getDataFile_in_folder(['spinor'], output_folder)   # spinor composition at in-plane k = 0
        except FileNotFoundError:
            warnings.warn("Spinor components output in CbHhLhSo basis is not found. Assuming decoupling of the conduction and valence bands...", category=self.NextnanoInputFileWarning)
            return int(0)

        # find the highest hole state
        num_evs = len(datafile.variables['s1'].value)
        for stateIndex in reversed(range(num_evs)):
            electronFraction = datafile.variables['s1'].value[stateIndex] + datafile.variables['s2'].value[stateIndex]
            if electronFraction < threshold:
                return stateIndex

        raise RuntimeError(f"No hole states found in: {output_folder}")


    def get_transition_energy():
        pass # TODO: urgent