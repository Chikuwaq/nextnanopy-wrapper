"""
Created on 2022/05/21

The SweepHelper class facilitates postprocessing of nextnano simulations 
when single or multiple input variables are swept.

This object-oriented user interface internally & automatically invokes nnShortcuts.

@author: takuma.sato@nextnano.com
"""

# Python includes
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import warnings
import logging

# nextnanopy includes
import nextnanopy as nn
import nnShortcuts.common as common
import nnShortcuts.nnp_shortcuts as nnp



class SweepHelper:
    """
        This class bridges the input and output of the nextnanopy.Sweep() simulations to facilitate postprocessing of multiple simulation data obtained by sweeping variable(s) in the input file.

        The initialization of the class will detect the software to use and construct a table of sweep information which is useful for postprocessing.

        Running sweep simulation may take long time. If the output data already exists for the identical input file and sweep values, this class allows postprocessing without running simulations.
        WARNING: Plots do not always guarantee that the data is up-to-date, e.g. when you modify the input file but do not change the file name and sweep range.

        Notes
        -----
            - You can sweep as many variables as you like in one go, unless the file paths exceed the limit of the system.
            - The sweep data can be exported to a CSV or an Excel file by:
              SweepHelper.data.to_csv()
              SweepHelper.data.to_excel()
              For the available options, see pandas.DataFrame.

        Attributes
        ----------
        sweep_space : dict of numpy.ndarray - { 'sweep variable': array of values }
            axes and coordinates of the sweep parameter space

        software : str
            nextnano software used

        master_input_file : nextnanopy.InputFile object
            master input file in which one or more variables are swept

        output_folder_path : str
            parent output folder path of sweep simulations

        sweep_obj : nextnanopy.Sweep object
            instantiated based on self.sweep_space and self.master_input_file

        data : pandas.DataFrame object
            table of sweep data with the following columns:
                sweep_coords : tuple
                    sweep space coordinates of each simulation. (value of 1st sweep variable, value of 2nd sweep variable, ...)
                output_subfolder : str
                    output subfolder path of each simulation
                overlap : complex
                    envelope overlap between the highest hole-like and lowest electron-like states
                transition_energy : real
                    energy difference between the highest hole-like and lowest electron-like states
                hole_energy_difference : real
                    energy difference between the highest heavy-hole and highest light-hole states
            Rows are ordered in the same manner as self.sweep_obj.input_files.

        states_to_be_plotted : dict of numpy.ndarray - { 'quantum model': array of values }
            for each quantum model, array of eigenstate indices to be plotted in the figure (default: all states in the output data)


        Methods
        -------
        execute_sweep(self, convergenceCheck=True, show_log=False)
            Run nextnano simulations.

        plot_dispersions(self, sweep_variable)
            Plot multiple dispersions from sweep output.

        generate_gif(self, sweep_variable)
            Generate GIF animation from multiple dispersions obtained from sweep.

        plot_overlap_squared(self, x_axis, y_axis, x_label='', y_label='', plot_title='', figFilename=None)
            Plot the overlap colormap as a function of two selected sweep axes.

        plot_transition_energies(self, x_axis, y_axis, x_label='', y_label='', plot_title='', figFilename=None)
            Plot the transition energies as a function of two selected sweep axes.

        plot_inplaneK()
            Plot the in-plane k points at which the Schroedinger equation has been solved.

    """

    def __init__(self, sweep_ranges, master_input_file, eigenstate_range=None, round_decimal=8, loglevel=logging.INFO):
        """
        Parameters
        ----------
        sweep_ranges : dict of tuple - { 'sweep variable': tuple([min, max], number of points) }
            the range and number of simulations for each sweep variable key.

        master_input_file : nextnanopy.InputFile object
            master input file in which one or more variables are swept

        eigenstate_range : dict of list - { 'quantum model': [min, max] }, optional
            for each quantum model, eigenstates in this range will be plotted. (default: plot all states in the output data)

        round_decimal : int, optional
            maximum number of decimals for swept variables. Default is 8 (consistent to nextnanopy)

        loglevel : logging level, optional
            determines to which extent internal process should be printed to console. 
            Available options are DEBUG/INFO/WARNING/ERROR/CRITICAL. See logging module for details.

        """
        # validate arguments
        if not isinstance(sweep_ranges, dict): raise TypeError("__init__(): argument 'sweep_ranges' must be a dict")
        # if not isinstance(master_input_file, nn.InputFile): raise TypeError("__init__(): argument 'master_input_file' must be a nextnanopy.InputFile object")   # TODO: object type has been modified in nextnanopy
        if eigenstate_range is not None:
            if not isinstance(eigenstate_range, dict): raise TypeError("__init__(): argument 'eigenstate_range' must be a dict")
            for model, plot_range in eigenstate_range.items():
                if model not in nnp.model_names: raise KeyError(f"__init__(): Illegal quantum model '{model}'")
                if len(plot_range) != 2: raise ValueError("__init__(): argument 'eigenstate_range' must be of the form 'quantum model': [min, max]")

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

        # generate self.sweep_space
        self.sweep_space = dict()
        for var in sweep_ranges:
            bounds, num_points = sweep_ranges[var]
            self.sweep_space[var] = np.around(np.linspace(bounds[0], bounds[1], num_points), round_decimal)   # avoid lengthy filenames

        # detect software
        self.software, extension = common.detect_software_new(master_input_file)
        if self.software != 'nextnano++': raise NotImplementedError("class SweepHelper currently supports only nextnano++ simulations.")

        # store master input file object
        self.master_input_file = master_input_file

        # store parent output folder path of sweep simulations
        input_file_name = common.separateFileExtension(self.master_input_file.fullpath)[0]
        self.output_folder_path = common.getSweepOutputFolderPath(input_file_name, self.software, *self.sweep_space.keys())

        # instantiate nn.Sweep object
        self.sweep_obj = nn.Sweep(self.sweep_space, self.master_input_file.fullpath)
        self.sweep_obj.save_sweep(round_decimal=round_decimal)  # ensure the same decimals for self.sweep_space and input file names

        logging.debug("\nSweep space axes:")
        logging.debug(f"{ [ key for key in self.sweep_space.keys() ] }")

        # instantiate pandas.DataFrame to store sweep data
        input_paths = [input_file.fullpath for input_file in self.sweep_obj.input_files]
        self.data = pd.DataFrame({
            'sweep_coords' : list(itertools.product(*self.sweep_space.values())),  # create cartesian coordinates in the sweep space. Consistent to nextnanopy implementation.
            'output_subfolder' : [common.get_output_subfolder_path(self.output_folder_path, input_path) for input_path in input_paths],
            'overlap' : None,
            'transition_energy' : None,
            'hole_energy_difference' : None
            })
        logging.info(f"Initialized data table:\n{self.data}")
        assert len(self.data) == len(self.sweep_obj.input_files)


        # initialize eigenstate indices to be plotted
        if eigenstate_range is not None:
            self.states_to_be_plotted = dict()
            for model, plot_range in eigenstate_range.items():
                self.states_to_be_plotted[model] = np.arange(plot_range[0]-1, plot_range[1], 1)
        else:   # default
            if self.__output_subfolders_exist():   # if output data exists, set to all states in the output data
                datafiles_probability = nnp.getDataFile_probabilities_in_folder(self.data.loc[0, 'output_subfolder'])
                self.states_to_be_plotted, num_evs = nnp.get_states_to_be_plotted(datafiles_probability)   # states_range_dict=None -> all states are plotted
            else:
                self.states_to_be_plotted = None   # self.states_to_be_plotted will be set up after sweep execution. See execute_sweep()


    # def __repr__(self):
    #     return ""

    def __str__(self):
        """ this method is executed when print(SweepHelper object) is invoked """
        print("\n[SweepHelper]")
        print("\tMaster input file: ", self.master_input_file.fullpath)
        print("\tSolver: ", self.software)
        print("\tSweep space grids: ")
        for var, values in self.sweep_space.items():
            print(f"\t\t{var} = ", values)
        print("\tOutput folder: ", self.output_folder_path)
        print("\tOutput data exists: ", self.__output_subfolders_exist())
        return ""


    ### auxillary postprocessing methods ####################################
    def __output_subfolders_exist(self):
        """
        Check if all output subfolders with the same input file name and sweep values exist
        (does not garantee that the results are up-to-date!)

        Returns
        -------
        bool

        """
        return all(os.path.isdir(s) for s in self.data['output_subfolder'])


    def __validate_sweep_variables(self, sweep_var):
        if sweep_var not in self.sweep_space:
            if sweep_var not in self.master_input_file.variables:
                raise KeyError(f"Variable {sweep_var} is not in the input file!")
            else:
                raise KeyError(f"Variable {sweep_var} has not been swept.")
        return

    def __slice_data_for_colormap(self, key, x_axis, y_axis, datatype=np.double):
        """
        Extract the data along two specified sweep parameters

        Parameters
        ----------
        key : str
            key in self.data object, specifying the data to be sliced
        x_axis : str
            sweep variable to be taken for x-axis.
        y_axis : str
            sweep variable to be taken for y-axis.
        datatype : numpy.dtype, optional
            Data type of the sliced data will be set to this type.

        Returns
        -------
        x_values : 1D numpy array
            Values of the first sweep variable
        y_values : 1D numpy array
            Values of the second sweep variable
        res : 2D numpy array
            Sliced data

        """
        # validate arguments
        if key not in self.data.keys():
            raise ValueError(f"{key} is not calculated in this sweep!")
        self.__validate_sweep_variables(x_axis)
        self.__validate_sweep_variables(y_axis)

        x_values = self.sweep_space[x_axis]
        y_values = self.sweep_space[y_axis]

        # identify index of plot axes
        for i, var in enumerate(self.sweep_space.keys()):
            if var == x_axis:
                x_axis_variable_index = i
            if var == y_axis:
                y_axis_variable_index = i

        sweep_space_reduced = self.__extract_2D_plane_from_sweep_space(x_axis, y_axis)

        # returns bool whether the point in the sweep space belongs to the plot region
        def isIn(coords):
            return all(coords[i] in sweep_space_reduced[var] for i, var in enumerate(self.sweep_space.keys()))

        # pick up sub-array of overlap data accordingly
        # res = [[0 for i in range(len(y_values))] for i in range(len(x_values))]
        res = np.zeros((len(y_values), len(x_values)), dtype=datatype)  # matplotlib pcolormesh assumes (num of y values)=(num of rows) and (num of x values)=(num of columns) for C axis data
        for coords, quantity in zip(self.data['sweep_coords'], self.data[key]):
            if not isIn(coords): continue

            xIndex = np.where(x_values == coords[x_axis_variable_index])   # find index for which arr == value
            yIndex = np.where(y_values == coords[y_axis_variable_index])
            res[yIndex, xIndex] = quantity  # matplotlib pcolormesh assumes (num of y values)=(num of rows) and (num of x values)=(num of columns) for C axis data

        return x_values, y_values, res


    def __extract_1D_line_from_sweep_space(self, sweep_var):
        """
        Extract 1D line from multidimensional (d >= 1) sweep space.
        """
        sweep_space_reduced = self.sweep_space

        # ask the values for other axes
        logging.info(f"Taking '{sweep_var}' for plot axis.")
        for var, array in self.sweep_space.items():
            if var == sweep_var: continue

            print("\nRemaining sweep dimension: ", var)
            print("Simulation has been performed at: ")
            for i, val in enumerate(array):
                print(f"index {i}: {val}")
            while True:
                choice = input("Specify value for the plot with index: ")
                if choice == 'q':
                    raise RuntimeError('Nextnanopy terminated.') from None
                try:
                    iChoice = int(choice)
                except ValueError:
                    print("Invalid input. (Type 'q' to quit)")
                    continue
                if iChoice not in range(len(array)):
                    print("Invalid input. (Type 'q' to quit)")
                else:
                    break
            sweep_space_reduced[var] = [array[iChoice]]   # only one element, but has to be an Iterable for the use below
        return sweep_space_reduced


    def __extract_2D_plane_from_sweep_space(self, sweep_var1, sweep_var2):
        """
        Extract 2D plane from multidimensional (d >= 2) sweep space.
        """
        sweep_space_reduced = self.sweep_space

        # ask the values for other axes
        logging.info(f"Taking '{sweep_var1}' and '{sweep_var2}' for plot axes.")
        for var, array in self.sweep_space.items():
            if var == sweep_var1 or var == sweep_var2: continue

            print("\nRemaining sweep dimension: ", var)
            print("Simulation has been performed at: ")
            for i, val in enumerate(array):
                print(f"index {i}: {val}")
            while True:
                choice = input("Specify value for the plot with index: ")
                if choice == 'q':
                    raise RuntimeError('Nextnanopy terminated.') from None
                try:
                    iChoice = int(choice)
                except ValueError:
                    print("Invalid input. (Type 'q' to quit)")
                    continue
                if iChoice not in range(len(array)):
                    print("Invalid input. (Type 'q' to quit)")
                else:
                    break
            sweep_space_reduced[var] = [array[int(choice)]]   # only one element, but has to be an Iterable for the use below
        logging.debug("Extracted sweep_space", sweep_space_reduced)
        return sweep_space_reduced


    def __setup_2D_color_plot(self, ax, x_axis, y_axis, x_label, y_label, plot_title, x_values, y_values):
        """ 
        Set labels, ticks and titles of a 2D colormap plot 
        """
        if x_label is None: x_label = x_axis
        if y_label is None: y_label = y_axis

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(plot_title)
        plt.xticks(x_values)
        plt.yticks(y_values)

        # trim x- and y-axis tick labels, while keeping all the ticks present
        nSimulations_x = len(self.sweep_space[x_axis])
        nSimulations_y = len(self.sweep_space[y_axis])
        max_num_xticks = 6   # TBD
        max_num_yticks = 20  # TBD
        if nSimulations_x > max_num_xticks:
            xticklabel_interval = int(nSimulations_x / max_num_xticks)
            for i, xtick in enumerate(ax.xaxis.get_ticklabels()):  
                if i % xticklabel_interval != 0: xtick.set_visible(False)
        if nSimulations_y > max_num_yticks:
            yticklabel_interval = int(nSimulations_y / max_num_yticks)
            for i, ytick in enumerate(ax.yaxis.get_ticklabels()):
                if i % yticklabel_interval != 0: ytick.set_visible(False)
        


    ### Sweep execution #####################################################
    def execute_sweep(self, convergenceCheck=True, show_log=False, parallel_limit=1):
        """
        Run simulations.

        Parameters
        ----------
        convergenceCheck : bool, optional
            The default is True.
        show_log : bool, optional
            The default is False.
        parallel_limit : int, optional
            Maximum number of parallel execution

        """
        # warn the user if many serial simulations are requested
        num_of_simulations = self.data['sweep_coords'].size
        if parallel_limit == 1:
            if (num_of_simulations > 100 and self.software == 'nextnano++') or (num_of_simulations > 10 and self.software == 'nextnano.NEGF'):
                while (True):
                    choice = input(f"WARNING: {num_of_simulations} simulations requested without parallelization. Are you sure you want to run all of them one-by-one? [y/n]")
                    if choice == 'y': break
                    elif choice == 'n': raise RuntimeError('Nextnanopy terminated.')

        logging.info(f"Running {num_of_simulations} simulations with max. {parallel_limit} parallelization ...")

        # execute sweep simulations
        self.sweep_obj.execute_sweep(
                delete_input_files = False,   # Do not delete input files so that SweepHelper.execute_sweep() can be invoked independently of __init__.
                overwrite          = True,    # avoid enumeration of output folders for secure output data access. 
                convergenceCheck   = convergenceCheck, 
                show_log           = show_log, 
                parallel_limit     = parallel_limit
                )   

        # If not given at the class instantiation, determine how many eigenstates to plot (states_to_be_plotted attribute)
        if self.states_to_be_plotted is None:   # by default, plot all states in the output data
            datafiles_probability = nnp.getDataFile_probabilities_in_folder(self.data.loc[0, 'output_subfolder'])
            self.states_to_be_plotted, num_evs = nnp.get_states_to_be_plotted(datafiles_probability)   # states_range_dict=None -> all states are plotted


    ### Dispersion ###########################################################
    def plot_dispersions(self, sweep_variable, savePDF=False):
        """
        Plot multiple dispersions from sweep output.

        Parameters
        ----------
        sweep_variable : str
            sweep variable name

        Returns
        -------
        None.

        """
        if not self.__output_subfolders_exist():
            raise RuntimeError("Simulation data does not exist for this sweep!")

        # validate argument
        self.__validate_sweep_variables(sweep_variable)

        sweep_space_reduced = self.__extract_1D_line_from_sweep_space(sweep_variable)

        # returns bool whether the point in the sweep space belongs to the plot region
        def isIn(coords):
            return all(coords[i] in sweep_space_reduced[var] for i, var in enumerate(sweep_space_reduced.keys()))

        # plot dispersions
        for model in self.states_to_be_plotted.keys():
            for coords, outfolder in zip(self.data['sweep_coords'], self.data['output_subfolder']):
                if isIn(coords):
                    nnp.plot_dispersion(outfolder, np.amin(self.states_to_be_plotted[model]), np.amax(self.states_to_be_plotted[model]), savePDF=savePDF)
        return


    def generate_gif(self, sweep_variable):
        """
        Generate GIF animation from multiple dispersions obtained from sweep.

        Parameters
        ----------
        sweep_variable : str
            sweep variable name

        Returns
        -------
        None.

        Note
        ----
        Currently, only 1D sweep is supported!!

        """
        if not self.__output_subfolders_exist():
            raise RuntimeError("Simulation data does not exist for this sweep!")

        # validate argument
        self.__validate_sweep_variables(sweep_variable)

        # generate GIF
        nnp.generate_gif(self.master_input_file, sweep_variable, self.sweep_space[sweep_variable], self.states_to_be_plotted)


    ### Optics analysis #######################################################
    def plot_overlap_squared(self, x_axis, y_axis, x_label=None, y_label=None, force_lightHole=False, plot_title='', figFilename=None, colormap='Greys'):
        """
        Plot the overlap colormap as a function of two selected sweep axes.

        Parameters
        ----------
        x_axis : str
            sweep variable for x-axis
        y_axis : str
            sweep variable for y-axis
        x_label : str, optional
            custom x-axis label
        y_label : str, optional
            custom y-axis label
        plot_title : str, optional
            title of the plot
        figFilename : str, optional
            output file name
        colormap : str, optional
            colormap used for the color bar

        Returns
        -------
        fig : matplotlib.figure.Figure object

        """
        if not self.__output_subfolders_exist():
            raise RuntimeError("Simulation data does not exist for this sweep!")

        # validate input
        self.__validate_sweep_variables(x_axis)
        self.__validate_sweep_variables(y_axis)      

        # Compute overlaps and store them in self.data
        logging.info("Calculating overlap...")
        self.data['overlap'] = self.data['output_subfolder'].apply(nnp.calculate_overlap, force_lightHole=force_lightHole)
        # self.data['overlap_squared'] = self.data['overlap'].apply(common.absolute_squared)   # BUG: somehow the results become complex128, not float --> cannot be plotted
        
        # x- and y-axis coordinates and 2D array-like of overlap data
        x_values, y_values, overlap = self.__slice_data_for_colormap('overlap', x_axis, y_axis, datatype=np.cdouble)   # complex double = two double-precision floats
        overlap_squared = np.abs(overlap)**2
        
        assert np.amin(overlap_squared) >= 0


        # instantiate 2D color plot
        fig, ax = plt.subplots()
        if not plot_title: plot_title = "Envelope overlap"
        self.__setup_2D_color_plot(ax, x_axis, y_axis, x_label, y_label, plot_title, x_values, y_values)

        # color setting
        from matplotlib import colors
        divnorm = colors.Normalize(vmin=0.)   # set the colorscale minimum to 0
        pcolor = ax.pcolormesh(x_values, y_values, overlap_squared, cmap=colormap, norm=divnorm, shading='auto')
        # pcolor = ax.pcolormesh(x_values, y_values, overlap_squared, cmap=colormap, shading='auto')

        cbar = fig.colorbar(pcolor)
        cbar.set_label("Envelope overlap")
        fig.tight_layout()
        plt.show()

        if figFilename is None or figFilename == "":
            name = os.path.split(self.output_folder_path)[1]
            figFilename = name + "_overlap"
        common.export_figs(figFilename, "png", self.software, output_folder_path=self.output_folder_path, fig=fig)


        # write info to a file
        max_val, indices = common.find_maximum(overlap_squared)  
        y_index, x_index = indices
        filepath = os.path.join(self.output_folder_path, os.path.join("nextnanopy", "info.txt"))
        logging.info(f"Writing info to:\n{filepath}")
        f = open(filepath, "w")  # w = write = overwrite existing content
        f.write(f"Overlap squared maximum {max_val} at:\n")
        f.write(f"{x_axis} = {x_values[x_index]}\n")
        f.write(f"{y_axis} = {y_values[y_index]}\n")
        f.close()
        
        return fig


    def plot_transition_energies(self, x_axis, y_axis, x_label=None, y_label=None, force_lightHole=False, plot_title='', figFilename=None, colormap=None, set_center_to_zero=False, unit='meV'):
        """
        Plot the transition energy (lowest electron eigenenergy - highest hole eigenenergy) colormap as a function of two selected sweep axes.

        Parameters
        ----------
        x_axis : str
            sweep variable for x-axis
        y_axis : str
            sweep variable for y-axis
        x_label : str, optional
            custom x-axis label
        y_label : str, optional
            custom y-axis label
        plot_title : str, optional
            title of the plot
        figFilename : str, optional
            output file name
        colormap : str, optional
            colormap used for the color bar
        set_center_to_zero : bool, optional
            If you are interested in the sign of transition energy (e.g. insulator-semimetal-topological insulator phase transition), set to True.
            Default is False
        unit : str, optional
            unit of transition energy. Currently supports 'meV' 'micron' 'um' 'nm'
        Returns
        -------
        fig : matplotlib.figure.Figure object

        """
        if not self.__output_subfolders_exist():
            raise RuntimeError("Simulation data does not exist for this sweep!")

        # validate input
        self.__validate_sweep_variables(x_axis)
        self.__validate_sweep_variables(y_axis)
        if unit not in ['meV', 'micron', 'um', 'nm']:
            raise ValueError(f"Energy unit {unit} is not supported.")

        # Get transition energies and store them in self.data
        self.data['transition_energy'] = self.data['output_subfolder'].apply(nnp.get_transition_energy, force_lightHole=force_lightHole)

        # x- and y-axis coordinates and 2D array-like of overlap data
        x_values, y_values, transition_energies = self.__slice_data_for_colormap('transition_energy', x_axis, y_axis, datatype=np.double)

        # Align unit
        if unit == 'meV':
            transition_energies_scaled = transition_energies * common.scale1ToMilli
        elif unit == 'micron' or unit == 'um':
            transition_energies_scaled = common.electronvolt_to_micron(transition_energies)
        elif unit == 'nm':
            transition_energies_scaled = common.electronvolt_to_micron(transition_energies) * 1e3

        
        # instantiate 2D color plot
        fig, ax = plt.subplots()
        if not plot_title: plot_title = "Transition energies"
        self.__setup_2D_color_plot(ax, x_axis, y_axis, x_label, y_label, plot_title, x_values, y_values)

        # color setting
        if colormap is None:  
            # default colors
            if set_center_to_zero: 
                colormap = 'seismic'
            else:
                colormap = 'viridis'
        if set_center_to_zero:
            from matplotlib import colors
            divnorm = colors.TwoSlopeNorm(vcenter=0.)
            pcolor = ax.pcolormesh(x_values, y_values, transition_energies_scaled, cmap=colormap, norm=divnorm, shading='auto')
        else:
            pcolor = ax.pcolormesh(x_values, y_values, transition_energies_scaled, cmap=colormap, shading='auto')
        
        cbar = fig.colorbar(pcolor)
        if unit == 'meV':
            cbar.set_label("Transition energy ($\mathrm{meV}$)")
        elif unit == 'micron' or unit == 'um':
            cbar.set_label("Wavelength ($\mu\mathrm{m}$)")
        elif unit == 'nm':
            cbar.set_label("Wavelength ($\mathrm{nm}$)")
        fig.tight_layout()
        plt.show()

        if figFilename is None or figFilename == "":
            name = os.path.split(self.output_folder_path)[1]
            figFilename = name + "_transitionEnergies"
        common.export_figs(figFilename, "png", self.software, output_folder_path=self.output_folder_path, fig=fig)

        return fig


    ### highest hole state ###################################################
    def plot_hole_energy_difference(self, x_axis, y_axis, x_label=None, y_label=None, plot_title='', figFilename=None, colormap=None, set_center_to_zero=True):
        """
        Plot the hole energy difference (HH - LH) colormap as a function of two selected sweep axes.

        Parameters
        ----------
        x_axis : str
            sweep variable for x-axis
        y_axis : str
            sweep variable for y-axis
        x_label : str, optional
            custom x-axis label
        y_label : str, optional
            custom y-axis label
        plot_title : str, optional
            title of the plot
        figFilename : str, optional
            output file name
        colormap : str, optional
            colormap used for the color bar
        set_center_to_zero : bool, optional
            If you are interested in the sign of transition energy (e.g. insulator-semimetal-topological insulator phase transition), set to True.
            Default is True

        Returns
        -------
        fig : matplotlib.figure.Figure object

        """
        if not self.__output_subfolders_exist():
            raise RuntimeError("Simulation data does not exist for this sweep!")

        # validate input
        self.__validate_sweep_variables(x_axis)
        self.__validate_sweep_variables(y_axis)

        # Get transition energies and store them in self.data
        self.data['hole_energy_difference'] = self.data['output_subfolder'].apply(nnp.get_hole_energy_difference)

        # x- and y-axis coordinates and 2D array-like of overlap data
        x_values, y_values, transition_energies = self.__slice_data_for_colormap('hole_energy_difference', x_axis, y_axis, datatype=np.double)


        # instantiate 2D color plot
        fig, ax = plt.subplots()
        if not plot_title: plot_title = "Energy difference HH - LH"
        self.__setup_2D_color_plot(ax, x_axis, y_axis, x_label, y_label, plot_title, x_values, y_values)

        # color setting
        if colormap is None:  
            # default colors
            if set_center_to_zero: 
                colormap = 'seismic'
            else:
                colormap = 'viridis'
        if set_center_to_zero:
            from matplotlib import colors
            divnorm = colors.TwoSlopeNorm(vcenter=0.)
            pcolor = ax.pcolormesh(x_values, y_values, transition_energies*common.scale1ToMilli, cmap=colormap, norm=divnorm, shading='auto')
        else:
            pcolor = ax.pcolormesh(x_values, y_values, transition_energies*common.scale1ToMilli, cmap=colormap, shading='auto')
        
        cbar = fig.colorbar(pcolor)
        cbar.set_label("Hole energy difference HH-LH ($\mathrm{meV}$)")
        fig.tight_layout()
        plt.show()

        if figFilename is None or figFilename == "":
            name = os.path.split(self.output_folder_path)[1]
            figFilename = name + "_holeEnergyDifference"
        common.export_figs(figFilename, "png", self.software, output_folder_path=self.output_folder_path, fig=fig)

        return fig


    ### in-plane k ###########################################################
    def plot_inplaneK(self):

        inplane_k = nnp.getKPointsData1D_in_folder(self.data.loc[0, 'output_subfolder'])   # assuming k points are identical to all the sweeps
        return nnp.plot_inplaneK(inplane_k)
