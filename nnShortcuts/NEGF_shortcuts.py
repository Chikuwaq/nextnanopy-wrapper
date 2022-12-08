"""
Created: 2022/03/22
Updated: 2022/09/02

Useful shortcut functions for nextnano.NEGF postprocessing.
get_* methods return nn.DataFile() attribute (output data)
plot_* methods plot & save figures
animate_NEGF method generates animation

@author: takuma.sato@nextnano.com (inspired by scripts of David Stark)
"""

# Python includes
import os
import matplotlib.pyplot as plt
import numpy as np

# nextnanopy includes
import nextnanopy as nn
import common


software = 'nextnano.NEGF'


def get_IV(input_file_name):
    """
    Get I-V curve.
    OUTPUT: 2 nn.DataFile() attributes for current & voltage
    """
    datafile = common.getDataFile('Current_vs_Voltage', input_file_name, software)
    voltage = datafile.coords['Potential per period']
    current = datafile.variables['Current density']
    return voltage, current

def plot_IV(input_file_name):
    """
    Plot the I-V curve.
    The plot is saved as an png image file.
    """
    voltage, current = get_IV(input_file_name)

    fig, ax = plt.subplots()
    ax.plot(voltage.value, current.value, 'o-')
    ax.set_xlabel(voltage.label)
    ax.set_ylabel("Current density ($\mathrm{A}/\mathrm{cm}^{2}$)")
    # ax.set_title(input_file_name)

    # export to an image file
    outputFolder = nn.config.get(software, 'outputdirectory')
    filename_no_extension = common.separateFileExtension(input_file_name)[0]
    outputSubfolder = os.path.join(outputFolder, filename_no_extension)
    common.export_figs("IV", "png", software, output_folder_path=outputSubfolder, fig=fig)



def getDataFile_NEGF_init(keywords, name):
    output_folder = nn.config.get(software, 'outputdirectory')
    filename_no_extension = common.separateFileExtension(name)[0]
    subfolder = os.path.join(output_folder, filename_no_extension)
    d = nn.DataFolder(subfolder)

    # Get the fullpath of 'Init' subfolder
    for folder_name in d.folders.keys():
        if 'Init' in folder_name:
            # listOfFiles = d.go_to(folder_name).find(keyword, deep=True)
            init_folder_path = d.go_to(folder_name).fullpath

    return common.getDataFile_in_folder(keywords, init_folder_path, software)  # TODO: add options available



def getDataFile_NEGF_atBias(keywords, name, bias):
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
    output_folder = nn.config.get(software, 'outputdirectory')
    filename_no_extension = common.separateFileExtension(name)[0]
    bias_subfolder = os.path.join(output_folder, filename_no_extension, str(bias) + 'mv')

    return common.getDataFile_in_folder(keywords, bias_subfolder, software)


# def get_convergenceInfo(bias):
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
#     output_folder = nn.config.get(software, 'outputdirectory')
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


def get_conduction_bandedge(input_file_name, bias):
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
        datafile = getDataFile_NEGF_atBias('Conduction_BandEdge.dat', input_file_name, bias)
    except FileNotFoundError:
        try:
            datafile = getDataFile_NEGF_atBias('ConductionBandEdge.dat', input_file_name, bias)
        except FileNotFoundError:
            raise

    position = datafile.coords['Position']
    # conduction_bandedge = datafile.variables['Conduction_BandEdge']
    conduction_bandedge = datafile.variables['Conduction Band Edge']
    return position, conduction_bandedge


def get_WannierStarkStates_init(name):
    """
    RETURN: nn.DataFile() attribute
        datafile.coords['Position']
        datafile.variables['Conduction BandEdge']
        datafile.variables['Psi_*']
    """
    datafile = getDataFile_NEGF_init('WannierStark_states.dat', name)

    position = datafile.coords['Position']
    conduction_bandedge = datafile.variables['ConductionBandEdge']

    Psi_squareds = []
    num_evs = len(datafile.variables) - 1
    for n in range(num_evs):
        for key in datafile.variables.keys():
            if f'Psi_{n+1}' in key: wanted_key = key   # NEGF output contains level and period indices, so variables['Psi_{n+1}'] doesn't work
        Psi_squareds.append(datafile.variables[wanted_key])

    return position, conduction_bandedge, Psi_squareds


def get_WannierStarkStates_atBias(input_file, bias):
    """
    RETURN: nn.DataFile() attribute
        datafile.coords['Position']
        datafile.variables['Conduction BandEdge']
        datafile.variables['Psi_*']
    """
    datafile = getDataFile_NEGF_atBias('WannierStark_states.dat', input_file, bias)

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


def plot_WannierStarkStates_init(
        name, 
        start_position=None, end_position=None, 
        labelsize=common.labelsize_default, 
        ticksize=common.ticksize_default
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
    position, CB, Psi_squareds = get_WannierStarkStates_init(name)

    # store data arrays
    if start_position is None and end_position is None:
        conduction_bandedge = CB.value
        WS_states = [Psi_squared.value for Psi_squared in Psi_squareds]
        x = position.value
    else: # cut off edges of the simulation region
        conduction_bandedge = common.cutOff_edges1D(CB.value, position.value, start_position, end_position)
        WS_states = [common.cutOff_edges1D(Psi_squared.value, position.value, start_position, end_position) for Psi_squared in Psi_squareds]
        x = common.cutOff_edges1D(position.value, position.value, start_position, end_position)

    WS_states = [common.mask_part_of_array(WS_state) for WS_state in WS_states]   # hide flat tails

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
    outputFolder = nn.config.get(software, 'outputdirectory')
    filename_no_extension = common.separateFileExtension(name)[0]
    outputSubfolder = os.path.join(outputFolder, filename_no_extension)
    common.export_figs("WannierStarkStates_init", "png", software, output_folder_path=outputSubfolder, fig=fig)

    return fig


def get_2Ddata_atBias(input_file_name, bias, data='carrier'):
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
        raise KeyError('Illegal data requested!')

    datafile = getDataFile_NEGF_atBias(file, input_file_name, bias)

    x = datafile.coords['x']
    y = datafile.coords['y']
    quantity = datafile.variables[variableKey]
    return x, y, quantity


def plot_DOS(
        input_file_name, 
        bias, 
        labelsize=common.labelsize_default, 
        ticksize=common.ticksize_default
        ):
    """
    Overlay bandedge with local density of states. Loads the following output data:
    DOS_energy_resolved.vtr

    The plot is saved as an png image file.
    """
    position, CB = get_conduction_bandedge(input_file_name, bias)
    x, y, quantity = get_2Ddata_atBias(input_file_name, bias, 'LDOS')

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
    outputFolder = nn.config.get(software, 'outputdirectory')
    filename_no_extension = common.separateFileExtension(input_file_name)[0]
    outputSubfolder = os.path.join(outputFolder, filename_no_extension)
    common.export_figs("DOS", "png", software, output_folder_path=outputSubfolder, fig=fig)

    return fig


def plot_carrier_density(input_file_name, bias, labelsize=common.labelsize_default, ticksize=common.ticksize_default):
    """
    Overlay bandedge with energy-resolved carrier density. Loads the following output data:
    CarrierDensity_energy_resolved.vtr

    The plot is saved as an png image file.
    """
    position, CB = get_conduction_bandedge(input_file_name, bias)
    x, y, quantity = get_2Ddata_atBias(input_file_name, bias, 'carrier')

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
    outputFolder = nn.config.get(software, 'outputdirectory')
    filename_no_extension = common.separateFileExtension(input_file_name)[0]
    outputSubfolder = os.path.join(outputFolder, filename_no_extension)
    common.export_figs("CarrierDensity", "png", software, output_folder_path=outputSubfolder, fig=fig)

    return fig


def plot_current_density(
        input_file_name, 
        bias, 
        labelsize=common.labelsize_default, 
        ticksize=common.ticksize_default
        ):
    """
    Overlay bandedge with energy-resolved current density. Loads the following output data:
    CurrentDensity_energy_resolved.vtr

    The plot is saved as an png image file.
    """
    position, CB = get_conduction_bandedge(input_file_name, bias)
    x, y, quantity = get_2Ddata_atBias(input_file_name, bias, 'current')

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
    outputFolder = nn.config.get(software, 'outputdirectory')
    filename_no_extension = common.separateFileExtension(input_file_name)[0]
    outputSubfolder = os.path.join(outputFolder, filename_no_extension)
    common.export_figs("CurrentDensity", "png", software, output_folder_path=outputSubfolder, fig=fig)

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
#     # datafile = common.getDataFile(f'Gain_SelfConsistent_vs_{xaxis}.dat', input_file_name, software)
#     datafile = common.getDataFile('Gain_vs_Voltage', input_file_name, software)
#     voltage = datafile.variables['Potential per period']
#     gain = datafile.variables['Maximum gain']
#     return voltage, gain

# def plot_gain():


def get_biases(name):
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
    output_folder = nn.config.get(software, 'outputdirectory')
    filename_no_extension = common.separateFileExtension(name)[0]
    datafolder = nn.DataFolder(os.path.join(output_folder, filename_no_extension))

    biases = [int(folder_name.replace('mV', '')) for folder_name in datafolder.folders.keys() if ('mV' in folder_name) and ('Init' not in folder_name)]
    biases.sort()   # ascending order
    return biases


def animate_NEGF(input_file_name, leftFig='DOS', rightFig='carrier'):
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

    import numpy as np
    import matplotlib.pyplot as plt

    input_file_name = common.separateFileExtension(input_file_name)[0]

    array_of_biases = np.array(get_biases(input_file_name))

    # get 2D data at the largest bias
    x_last, y_last, quantity_last = get_2Ddata_atBias(input_file_name, array_of_biases[-1], leftFig)

    # define a map from (xIndex, yIndex, biasIndex) to scalar value
    F = np.zeros((len(x_last.value), len(y_last.value), len(array_of_biases)))

    # store data to F
    for i, bias in enumerate(array_of_biases):
        position, CB = get_conduction_bandedge(input_file_name, bias)
        x, y, quantity = get_2Ddata_atBias(input_file_name, bias, leftFig)
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



def get_LIV(name):
    """
    Import power-current-voltage data from simulation output.

    Parameters
    ----------
    name : TYPE
        DESCRIPTION.

    Units (identical to nextnano.NEGF output)
    ---------------------------------------
    current density [A/cm^2]
    potential drop per period [mV]
    output power density [W/cm^3]

    Returns
    -------
    current_density_LV : TYPE
        DESCRIPTION.
    output_power_density : TYPE
        DESCRIPTION.

    """

    keyword = 'L-I-V.dat'

    # find the output file
    outputFolder = nn.config.get(software, 'outputdirectory')
    filename_no_extension = common.separateFileExtension(name)[0]
    outputSubfolder = os.path.join(outputFolder, filename_no_extension)
    print(f'Searching for output data with keyword {keyword}...')
    listOfFiles = nn.DataFolder(outputSubfolder).find(keyword, deep=True)
    if len(listOfFiles) != 1: raise RuntimeError(f"Multiple or no L-I-V data found in the directory {outputSubfolder}")

    data = np.loadtxt(listOfFiles[0], skiprows=1)

    current_density_LV = np.array(data[:, 0])
    potential_drop_per_period = np.array(data[:, 1])
    output_power_density = np.array(data[:, 2])

    return current_density_LV, potential_drop_per_period, output_power_density


def plot_Light_Current_Voltage_characteristics(
        names, 
        period_length, 
        num_periods, 
        area, 
        front_mirror_loss, 
        total_cavity_loss, 
        labels, 
        labelsize=common.labelsize_default, 
        ticksize=common.ticksize_default, 
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
    if len(names) != len(labels): raise common.NextnanopyScriptError(f"Number of input files ({len(names)}) do not match that of plot labels ({len(labels)})")

    # volume in [cm^3]
    area_in_cm2 = area * pow(common.scale1ToCenti, 2)
    volume = (period_length / common.scale1ToNano * common.scale1ToCenti) * num_periods * area_in_cm2

    def forward_conversion(I):
        """ convert current (A) to current density (kA/cm^2) """
        return I / area_in_cm2 * common.scale1ToKilo

    def backward_conversion(density):
        """ convert current density (kA/cm^2) to current (A) """
        return density * area_in_cm2 / common.scale1ToKilo


    # list of data for sweeping temperature
    current_densities_IV = list()
    voltages             = list()
    current_densities_LV = list()
    output_powers = list()

    for i, name in enumerate(names):
        # I-V data, units adjusted
        datafile_IV = common.getDataFile('Current_vs_Voltage.dat', name, software)
        density = datafile_IV.variables['Current density'].value * common.scale1ToKilo
        V = datafile_IV.coords['Potential per period'].value * num_periods / common.scale1ToMilli
        current_densities_IV.append(density)
        voltages.append(V)

        # L-V data, units adjusted
        current_density_LV, potential_drop, power_density = get_LIV(name)
        density = current_density_LV * common.scale1ToKilo
        P = (power_density * volume * common.scale1ToMilli) * front_mirror_loss / total_cavity_loss   # external output power
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



