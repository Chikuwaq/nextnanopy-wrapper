"""
Created on 2022/05/21

The SweepManager class facilitates postprocessing of nextnano simulations
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
import copy
import shutil
import subprocess
import time
import platform


# nextnanopy includes
import nextnanopy as nn

# my includes
from nnShortcuts.common import CommonShortcuts
from boostsweep.slurm_data import SlurmData
from boostsweep.sweep_space import SweepSpace
from nnShortcuts.default_colors import DefaultColors


class SweepManager:
    """
        This class bridges the input and output of sweep simulation by nextnanopy simulations to facilitate postprocessing of multiple simulation data obtained by sweeping variable(s) in the input file.

        The initialization of the class will detect the software to use and construct a table of sweep information which is useful for postprocessing.

        Running sweep simulation may take long time. If the output data already exists for the identical input file and sweep values, this class allows postprocessing without running simulations.
        WARNING: Plots do not always guarantee that the data is up-to-date, e.g. when you modify the input file but do not change the file name and sweep range.

        Advantages over nextnanopy.Sweep class
        --------------------------------------
            - You can sweep more variables you like in one go. When the temporary file name would get too long, use unique IDs until the end of simulation. Output folder names can be recovered by the recover_original_filenames() method.
            - The output folder paths are calculated even if you do not run the simulations. Because of this, you do not need to keep Sweep class object alive for postprocessing of sweep results.
            - Supports job submission to Slurm workload manager. See submit_sweep_to_slurm().

        Attributes
        ----------
        sweep_space : dict of numpy.ndarray - { 'sweep variable': array of values }
            Axes and coordinates of the sweep parameter space.
            Used for 1D/2D slicing of the whole sweep outputs.
            NOTE: When the SweepManager object has been instantiated with a list of individual sweep coords, sweep_space.get_values() may not be useful.

        shortcuts : nextnanopy-wrapper shortcut object
            shortcut object for the nextnano solver used for the master_input_file (automatically detected)

        master_input_file : dict of nextnanopy.InputFile object
            master input file in which one or more variables are swept.
            'original'  = original file name
            'short' = abbreviated file name

        isFilenameAbbreviated : bool
            indicates if the input file and output folder names are abbreviated

        output_folder_path : dict of str
            parent output folder path of sweep simulations
            'original'  = original file name
            'short' = abbreviated file name

        inputs : pandas.DataFrame object
            table of input data for each sweep coordinates with the following columns:
                sweep_coords : tuple
                    sweep space coordinates of each simulation. (value of 1st sweep variable, value of 2nd sweep variable, ...)
                obj : nextnanopy.InputFile object
                    Shallow copies of self.master_input_file with modified variable values
                fullpaths_original : str
                    fullpath to temporary input files with original file name
                fullpaths_short : str
                    fullpath to temporary input files with abbreviated file name
            NOTE: We split the temporary input file paths from nextnanopy's Sweep class object to avoid unnecessary calls of save_sweep(), which costs time.

        outputs : pandas.DataFrame object
            table of output data for each sweep coordinates with the following columns:
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
                absorption_at_transition_energy_TE : real
                    amplitude of optical absorption of x polarization at the transition energy (1/cm)
                absorption_at_transition_energy_TM : real
                    amplitude of optical absorption of z polarization at the transition energy (1/cm)

        states_to_be_plotted : dict of numpy.ndarray - { 'quantum model': array of values }
            for each quantum model, array of eigenstate indices to be plotted in the figure (default: all states in the output data)

        slurm_data : SlurmData object
            information used for Slurm jobs

        default_colors : DefaultColors object
    """
    default_colors = DefaultColors()
    n_characters_uuid = 5

    def __init__(self, sweep_ranges, master_input_file, eigenstate_range=None, round_decimal=8, abbreviate_if_too_long=True, loglevel=logging.INFO):
        """
        Parameters
        ----------
        sweep_ranges : dict of tuple - { 'sweep variable': tuple([min, max], number of points) }
                       OR
                       dict of list  - { 'sweep variable': list(value1, value2, ...) }
                       OR
                       list of tuple - [
                                         ('sweep variable 1', 'sweep variable 2', ...),  # str specifying variable names
                                         (value1-1, value2-1, ...),                      # individual sweep space coordinates
                                         (value1-2, value2-2, ...),
                                         ...
                                        ]
                       OR
                       None          - Run the master input file only.
            specifies the values of each sweep variable.

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
        # initialize members
        self.master_input_file = dict()
        self.sweep_space = dict()
        self.output_folder_path = dict()

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
        self.sweep_space = SweepSpace.create_from_sweep_ranges(sweep_ranges, round_decimal)

        # prepare shortcuts for the nextnano solver used
        self.shortcuts = CommonShortcuts.get_shortcut(master_input_file)
        if self.shortcuts.product_name not in ['nextnano3', 'nextnano++', 'nextnano.NEGF', 'nextnano.NEGF++']:
            raise NotImplementedError("class SweepManager currently supports only nextnano++ and nextnano.NEGF++ simulations.")

        if eigenstate_range is not None:
            if not isinstance(eigenstate_range, dict): raise TypeError(f"__init__(): argument 'eigenstate_range' must be a dict, but is {type(eigenstate_range)}")
            for model, plot_range in eigenstate_range.items():
                if model not in self.shortcuts.model_names: raise KeyError(f"__init__(): Quantum model '{model}' is not supported")
                if len(plot_range) != 2: raise ValueError("__init__(): argument 'eigenstate_range' must be of the form 'quantum model': [min, max]")

        self.round_decimal = round_decimal  # used by __create_input_file_fullpaths()

        # store its related data
        var_names = self.sweep_space.get_variable_names()
        if var_names is None:
            self.output_folder_path['original'] = self.shortcuts.compose_sweep_output_folder_path(master_input_file.fullpath)
        else:
            self.output_folder_path['original']  = self.shortcuts.compose_sweep_output_folder_path(master_input_file.fullpath, *var_names)

        # store master input file object with the original file name
        master_input_file.config.set(self.shortcuts.product_name, 'outputdirectory', self.output_folder_path['original'])
        self.master_input_file['original'] = copy.deepcopy(master_input_file)


        # fill sweep input and output data tables
        self.inputs = pd.DataFrame(columns=[
            'sweep_coords',
            'obj',
            'fullpaths_original',
            'fullpaths_short'
        ])
        self.outputs = pd.DataFrame(columns=[
            'sweep_coords',
            'output_subfolder_original',
            'output_subfolder_short',
            'overlap',
            'transition_energy_eV',
            'transition_energy_meV',
            'transition_energy_micron',
            'transition_energy_nm',
            'HH1-LH1',
            'HH1-HH2',
            'absorption_at_transition_energy_TE',
            'absorption_at_transition_energy_TM',
            'ave_current'
        ])

        if isinstance(sweep_ranges, dict):
            # create cartesian coordinates in the sweep space. Consistent to nextnanopy implementation.
            self.inputs['sweep_coords'] = list(itertools.product(*self.sweep_space.get_values()))
        elif isinstance(sweep_ranges, list):
            self.inputs['sweep_coords'] = sweep_ranges[1:]
        elif sweep_ranges is None:
            self.inputs['sweep_coords'] = None

        self.outputs['sweep_coords'] = self.inputs['sweep_coords']
        self.inputs['fullpaths_original'] = self.__create_input_file_fullpaths(self.master_input_file['original'])
        self.outputs['output_subfolder_original'] = SweepManager.__compose_subfolder_paths(0, self.inputs['fullpaths_original'], self.output_folder_path['original'])

        # prepare files and folders with abbreviated names if needed
        if var_names is None:
            outfolder = self.shortcuts.compose_sweep_output_folder_path(self.master_input_file['original'].fullpath)
            outpath = outfolder
        else:
            outfolder = self.shortcuts.compose_sweep_output_folder_path(self.master_input_file['original'].fullpath, *self.sweep_space.get_variable_names())
            initSweepCoords = {key: arr[0] for key, arr in self.sweep_space.get_items()}
            subfolder = self.shortcuts.compose_sweep_output_subfolder_name(self.master_input_file['original'].fullpath, initSweepCoords)
            outpath = os.path.join(outfolder, subfolder)
        max_path_length = 260
        if platform.system() == 'Linux':
            max_path_length = 4095
        elif platform.system() == 'Darwin':
            max_path_length = 1024
        if (len(outpath) + 160 <= max_path_length) or not abbreviate_if_too_long:
            self.isFilenameAbbreviated = False
        elif self.output_subfolders_exist_with_originalname():
            # simulation outputs of this sweep exist already. The user might want to access those outputs without executing sweep simulation.
            self.isFilenameAbbreviated = False
        else:
            self.isFilenameAbbreviated = True

            import uuid
            logging.info(f"Because the output path is too long ({len(outpath)}), creating a temporary input file with shorter name...")
            dir = os.path.dirname(master_input_file.fullpath)
            ext = os.path.splitext(master_input_file.fullpath)[1]
            id = str(uuid.uuid4())
            filename = 'tmp' + id[:SweepManager.n_characters_uuid] + ext  # using a part of the Universally Unique Identifier
            temp_path = os.path.join(dir, filename)
            master_input_file.save(temp_path, overwrite=True, automkdir=True)

        if var_names is None:
            self.output_folder_path['short']  = self.shortcuts.compose_sweep_output_folder_path(master_input_file.fullpath)
        else:
            self.output_folder_path['short']  = self.shortcuts.compose_sweep_output_folder_path(master_input_file.fullpath, *self.sweep_space.get_variable_names())

        master_input_file.config.set(self.shortcuts.product_name, 'outputdirectory', self.output_folder_path['short'])
        self.master_input_file['short'] = master_input_file

        self.inputs['fullpaths_short'] = self.__create_input_file_fullpaths(self.master_input_file['short'])
        self.outputs['output_subfolder_short'] = SweepManager.__compose_subfolder_paths(0, self.inputs['fullpaths_short'], self.output_folder_path['short'])


        # for convenience in postprocessing/visualizing CSV/Excel output
        if var_names is not None:
            def extract_coord(tupl, index=0):
                return tupl[index]
            for i, coord_key in enumerate(var_names):
                self.outputs[coord_key] = self.outputs['sweep_coords'].apply(extract_coord, index=i)

        logging.info(f"Initialized output data table:\n{self.outputs}")
        assert len(self.outputs) == self.inputs['fullpaths_short'].size


        # initialize eigenstate indices to be plotted
        if eigenstate_range is not None:
            self.states_to_be_plotted = dict()
            for model, plot_range in eigenstate_range.items():
                self.states_to_be_plotted[model] = np.arange(plot_range[0]-1, plot_range[1], 1)
        else:   # default
            self.states_to_be_plotted = None   # if this remains None, it will be set up after sweep execution. See execute_sweep().
            # if self.__output_subfolders_exist():   # if output data exists, set to all states in the output data   # TODO: output subfolder not found when input file is NEGF...?
            #     try:
            #         datafiles_probability = self.shortcuts.get_DataFile_probabilities_in_folder(self.data.loc[0, 'output_subfolder_original'])
            #         self.states_to_be_plotted, num_evs = self.shortcuts.get_states_to_be_plotted(datafiles_probability)   # states_range_dict=None -> all states are plotted
            #     except FileNotFoundError as e:
            #         pass

        self.slurm_data = SlurmData(self.output_folder_path['short'])

    @staticmethod
    def __compose_subfolder_paths(n_characters_to_remove : int, input_fullpaths : pd.DataFrame, output_folder_path : str):
        subfolder_paths = list()
        for input_path in input_fullpaths:
            sweep_input_name = os.path.split(input_path)[1]   # remove directory
            sweep_var_string = sweep_input_name[n_characters_to_remove:]  # remove master input file name to shorten the paths
            subfolder_paths.append(CommonShortcuts.get_output_subfolder_path(output_folder_path, sweep_var_string))
        return subfolder_paths

    def __str__(self):
        """ this method is executed when print(SweepManager object) is invoked """
        print("\n[SweepManager]")
        print("\tMaster input file: ", self.master_input_file['original'].fullpath)
        print("\tSolver: ", self.shortcuts.product_name)
        print("\tSweep space grids: ")
        for var, values in self.sweep_space.get_items():
            print(f"\t\t{var} = ", values)
        print("\tOutput folder: ", self.output_folder_path['short'])
        print("\tOutput data exists: ", self.__output_subfolders_exist())
        return ""


    def __create_input_file_fullpaths(self, master_input_file):
        """
        Sweep.save_sweep() creates temporary input files with these names.
        However, we do not use Sweep.save_sweep() in __init__ for code speed when execution of sweep is not desired (when simulation outputs already exist).

        Returns
        -------
        input file fullpaths : pandas.Series object
        """
        input_file_fullpaths = ['' for _ in range(self.get_num_simulations())]

        var_names = self.sweep_space.get_variable_names()
        if var_names is None:
            input_file_fullpaths[0] = master_input_file.fullpath
        else:
            filename_path, filename_extension = os.path.splitext(master_input_file.fullpath)
            folder = os.path.split(filename_path)[0]
            for i, combination in enumerate(self.outputs['sweep_coords']):
                # filename_end = '__'  # code following nextnanopy > inputs.py > Sweep.create_input_files()
                filename_end = ''
                for var_name, var_value in zip(var_names, combination):
                    var_value_string = SweepManager.format_number(var_value, self.round_decimal)
                    filename_end += '{}_{}_'.format(var_name, var_value_string)
                # input_file_fullpaths[i] = filename_path + filename_end + filename_extension  # code following nextnanopy > inputs.py > Sweep.create_input_files()
                input_file_fullpaths[i] = os.path.join(folder, filename_end + filename_extension)
        return pd.Series(input_file_fullpaths)


    ### getter and checker methods of class attributes ####################################
    def __get_output_folder_path(self):
        """
        Returns
        -------
        path to simulation output folder
        """
        if self.isFilenameAbbreviated:
            return self.output_folder_path['short']
        else:
            return self.output_folder_path['original']


    def __get_output_subfolder_paths(self):
        """
        Returns
        -------
        list of paths to simulation output subfolders
        """
        if self.isFilenameAbbreviated:
            return self.outputs['output_subfolder_short']
        else:
            return self.outputs['output_subfolder_original']


    def __output_subfolders_exist(self):
        """
        Check if all output subfolders with the same input file name and sweep values exist
        (does not guarantee that the simulation output is up-to-date!)

        Returns
        -------
        bool

        """
        return all(os.path.isdir(path) for path in self.__get_output_subfolder_paths())


    def output_subfolders_exist_with_originalname(self):
        return all(os.path.isdir(path) for path in self.outputs['output_subfolder_original'])


    def __validate_sweep_variables(self, sweep_var):
        if not self.sweep_space.has_sweep_variable(sweep_var):
            if sweep_var not in self.master_input_file['original'].variables:
                raise KeyError(f"Variable {sweep_var} is not in the input file!")
            else:
                raise KeyError(f"Variable {sweep_var} has not been swept.")
        return


    def get_num_simulations(self):
        if self.inputs['sweep_coords'].size == 0:
            # empty SweepSpace means single simulation
            return 1
        else:
            return self.inputs['sweep_coords'].size


    ### auxillary postprocessing methods ####################################
    def __slice_data_for_colormap_2D(self, key, x_axis, y_axis, datatype=np.double):
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
        if key not in self.outputs.keys():
            raise ValueError(f"{key} is not calculated in this sweep!")
        self.__validate_sweep_variables(x_axis)
        self.__validate_sweep_variables(y_axis)

        x_values = self.sweep_space.get_values_by_variable_name(x_axis)
        y_values = self.sweep_space.get_values_by_variable_name(y_axis)

        # identify index of plot axes
        for i, var in enumerate(self.sweep_space.get_variable_names()):
            if var == x_axis:
                x_axis_variable_index = i
            if var == y_axis:
                y_axis_variable_index = i

        sweep_space_reduced = self.sweep_space.extract_2D_plane(x_axis, y_axis)

        # pick up sub-array of overlap data accordingly
        # res = [[0 for i in range(len(y_values))] for i in range(len(x_values))]
        res = np.zeros((len(y_values), len(x_values)), dtype=datatype)  # matplotlib pcolormesh assumes (num of y values)=(num of rows) and (num of x values)=(num of columns) for C axis data
        for coords, quantity in zip(self.outputs['sweep_coords'], self.outputs[key]):
            if not sweep_space_reduced.has_sweep_point(coords):
                continue

            xIndex = np.where(x_values == coords[x_axis_variable_index])   # find index for which arr == value
            yIndex = np.where(y_values == coords[y_axis_variable_index])
            res[yIndex, xIndex] = quantity  # matplotlib pcolormesh assumes (num of y values)=(num of rows) and (num of x values)=(num of columns) for C axis data

        return x_values, y_values, res


    def __slice_data_for_colormap_1D(self, key, x_axis, datatype=np.double):
        """
        Extract the data along one specified sweep parameter

        Parameters
        ----------
        key : str
            key in self.data object, specifying the data to be sliced
        x_axis : str
            sweep variable to be taken for x-axis.
        datatype : numpy.dtype, optional
            Data type of the sliced data will be set to this type.

        Returns
        -------
        x_values : 1D numpy array
            Values of the first sweep variable
        res : 1D numpy array
            Sliced data

        """
        # validate arguments
        if key not in self.outputs.keys():
            raise ValueError(f"{key} is not calculated in this sweep!")
        self.__validate_sweep_variables(x_axis)

        x_values = self.sweep_space.get_values_by_variable_name(x_axis)

        # identify index of plot axes
        for i, var in enumerate(self.sweep_space.get_variable_names()):
            if var == x_axis:
                x_axis_variable_index = i

        sweep_space_reduced = self.sweep_space.extract_1D_line(x_axis)

        # pick up sub-array of overlap data accordingly
        res = np.zeros(len(x_values), dtype=datatype)
        for coords, quantity in zip(self.outputs['sweep_coords'], self.outputs[key]):
            if not sweep_space_reduced.has_sweep_point(coords):
                continue

            xIndex = np.where(x_values == coords[x_axis_variable_index])   # find index for which array == value
            res[xIndex] = quantity

        return x_values, res


    def __setup_2D_color_plot(self, ax, x_axis, y_axis, x_label, y_label, plot_title, x_values, y_values, logscale):
        """
        Set labels, ticks and titles of a 2D colormap plot
        """
        if x_label is None: x_label = x_axis
        if y_label is None: y_label = y_axis

        if logscale:
            ax.set_xscale('log')
            ax.set_yscale('log')

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(plot_title)
        if logscale:
            ax.set_xlim(x_values[0], x_values[-1])
            ax.set_ylim(y_values[0], y_values[-1])
        else:
            plt.xticks(x_values)
            plt.yticks(y_values)

        # trim x- and y-axis tick labels, while keeping all the ticks present
        nSimulations_x = self.sweep_space.get_nPoints_by_variable_name(x_axis)
        nSimulations_y = self.sweep_space.get_nPoints_by_variable_name(y_axis)
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


    def __setup_1D_plot(self, ax, x_axis, x_label, plot_title, x_values):
        """
        Set labels, ticks and titles of a 1D plot
        """
        if x_label is None: x_label = x_axis

        ax.set_xlabel(x_label)
        ax.set_title(plot_title)
        plt.xticks(x_values)

        # trim x- and y-axis tick labels, while keeping all the ticks present
        nSimulations_x = self.sweep_space.get_nPoints_by_variable_name(x_axis)
        max_num_xticks = 6   # TBD
        if nSimulations_x > max_num_xticks:
            xticklabel_interval = int(nSimulations_x / max_num_xticks)
            for i, xtick in enumerate(ax.xaxis.get_ticklabels()):
                if i % xticklabel_interval != 0: xtick.set_visible(False)



    ### Sweep execution #####################################################
    def execute_sweep(self, convergenceCheck=True, show_log=False, parallel_limit=1, **kwargs):
        """
        Run simulations.

        Use the method 'recover_original_filenames()' to recover the original input file name in the output folders after nextnano execution
        if the input file name has been abbreviated due to limited output path length.

        Parameters
        ----------
        convergenceCheck : bool, optional
            The default is True.
        show_log : bool, optional
            The default is False.
        parallel_limit : int, optional
            Maximum number of parallel execution
        kwargs : optional
            other parameters accepted by nextnanopy.InputFile.execute()

        """
        import concurrent.futures

        self.save_sweep(parallel_limit)

        # execute sweep simulations
        # this writes output to self.data['output_subfolder_short']
        # NOTE: We avoid enumeration of output folder names (see `overwrite` option of nextnanopy.inputs > Sweep.execute_sweep()) for secure output data access.
        # NOTE: Do not delete input files! Otherwise SweepManager.execute_sweep() cannot be called independently of SweepManager instantiation.
        def run_input_file(input_file):
            input_file.execute(show_log=show_log, convergenceCheck=convergenceCheck, **kwargs)  # TODO: add option to use multiple threads in each simulation

        logging.info(f"Starting {self.get_num_simulations()} sweep simulations...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_limit) as executor:
            # submit jobs
            futures = [executor.submit(run_input_file, input_file) for input_file in self.inputs['obj']]

            # count the number of finished jobs
            n_finished_jobs = 0
            for future in concurrent.futures.as_completed(futures):
                n_finished_jobs += 1
                logging.info(f"Completed jobs: {n_finished_jobs}/{self.get_num_simulations()}")

        # point to the new output in case old simulation outputs exist with original file name
        self.isFilenameAbbreviated = (self.master_input_file['short'].fullpath != self.master_input_file['original'].fullpath)

        # If not given at the class instantiation, determine how many eigenstates to plot (states_to_be_plotted attribute)
        if self.states_to_be_plotted is None:   # by default, plot all states in the output data
            if os.path.exists(self.outputs.loc[0, 'output_subfolder_short']):
                try:
                    datafiles_probability = self.shortcuts.get_DataFile_probabilities_in_folder(self.outputs.loc[0, 'output_subfolder_short'])
                    self.states_to_be_plotted, num_evs = self.shortcuts.get_states_to_be_plotted(datafiles_probability)   # states_range_dict=None -> all states are plotted
                except (FileNotFoundError, ValueError):  # 1,2,3-band NEGF doesn't have probability output
                    warnings.warn("SweepManager.execute_sweep(): Probability distribution not found")


    def save_sweep(self, parallel_limit):
        """
        Create temporary input files for all sweep points.
        """
        # warn the user if many serial simulations are requested
        n = self.get_num_simulations()
        if parallel_limit == 1:
            if (n > 100 and self.shortcuts.product_name == 'nextnano++') or (n > 10 and self.shortcuts.product_name == 'nextnano.NEGF'):
                while (True):
                    choice = input(f"WARNING: {n} simulations requested without parallelization. Are you sure you want to run all of them one-by-one? [y/n]")
                    if choice == 'y': break
                    elif choice == 'n': raise RuntimeError('Nextnanopy terminated.')

        logging.info(f"Preparing {n} simulations for \n{self.master_input_file['short'].fullpath}")
        logging.info(f"Max. {parallel_limit} simulations are run simultaneously.")

        # Do not repeatedly call list.append()! Slow when the number of simulations is large.
        # Shallow copy should be enough because the only change is the input variables.
        self.inputs['obj'] = [copy.copy(self.master_input_file['short']) for _ in range(n)]

        var_names = self.sweep_space.get_variable_names()
        if var_names is not None:
            # empty SweepSpace means single simulation
            for i, row in self.inputs.iterrows():
                for var_name, var_value in zip(var_names, row['sweep_coords']):
                    row['obj'].set_variable(var_name, var_value)
                row['obj'].save(row['fullpaths_short'], overwrite=True)

        # i_input = 0
        # for input_path, coords in zip(self.inputs['fullpaths_short'], self.inputs['sweep_coords']):
        #     for var_name, var_value in zip(self.sweep_space.get_variable_names(), coords):
        #         self.inputs['obj'][i_input].set_variable(var_name, var_value)
        #     self.inputs['obj'][i_input].save(input_path, overwrite=True)
        #     i_input += 1


    def recover_original_filenames(self):
        if not self.isFilenameAbbreviated:
            return
        if not self.__output_subfolders_exist():
            logging.warning("Cannot recover output folder names because output subfolders do not exist.")
            return

        logging.info(f"Recovering original input file name in output folder names...")

        for short, original in zip(self.outputs['output_subfolder_short'], self.outputs['output_subfolder_original']):
            if os.path.exists(original):
                shutil.rmtree(original)
            os.makedirs(original, exist_ok=False)

            for item in os.listdir(short):
                source_item = os.path.join(short, item)
                destination_item = os.path.join(original, item)
                shutil.move(source_item, destination_item)

        if os.path.exists(self.output_folder_path['short']):
            shutil.rmtree(self.output_folder_path['short'])

        self.isFilenameAbbreviated = False


    def delete_input_files(self):
        """
        Delete the temporary input files of the sweep object.

        Notes
        -----
        Convenient to have a separate method so that self.execute_sweep() can be invoked independently of __init__().
        """
        input_file_fullpaths = self.inputs['fullpaths_original'] + self.inputs['fullpaths_short']
        paths = set(input_file_fullpaths)  # avoid duplicates (NOTE: set object does not preserve the order of elements!)
        for path in paths:
            if os.path.exists(path):
                os.remove(path)

        # delete the input file whose name has been abbreviated
        if self.isFilenameAbbreviated:
            if os.path.exists(self.master_input_file['short'].fullpath):
                os.remove(self.master_input_file['short'].fullpath)
        logging.info("Sweep (temporary) input files deleted.")


    def delete_output(self):
        """
        Delete the output data of the sweep object.
        """
        outfolder = self.__get_output_folder_path()
        if not os.path.exists(outfolder):
            warnings.warn("Output folder does not exist!")
            return
        try:
            shutil.rmtree(outfolder)
        except OSError as e:
            raise


    ### Slurm methods #######################################################
    def is_slurm_simulation(self):
        return self.slurm_data.partition is not None


    def submit_sweep_to_slurm(self, suffix='', partition='microcloud', nodelist=None, email=None, num_CPU=4, memory_limit='8G', time_limit_hrs=5, exe=None, output_folder=None, database=None):
        """
        Submit sweep simulations to Slurm workload manager.

        Parameters
        ----------
            partition : str
                name of the computer partition
            nodelist : str
                specifies which node of the partition to use
            email : str
                Email is sent when the last input file in the sweep has finished.
            num_CPU : int
                number of CPUs available. Used for the nextnano command line parameter '--threads'
                Number of physical cores = num_CPU
                Number of threads when hyperthreading = 2 * num_CPU
                Optimal number of threads for omp parallelism <= (2 * num_CPU) / 2 = num_CPU
        """
        self.generate_slurm_sbatches(suffix=suffix, partition=partition, nodelist=nodelist, email=email, num_CPU=num_CPU, memory_limit=memory_limit, time_limit_hrs=time_limit_hrs, exe=exe, output_folder=output_folder, database=database)

        num_sbatch_scripts = len(self.slurm_data.sbatch_script_paths)
        self.save_sweep(num_sbatch_scripts)

        num_metascripts = len(self.slurm_data.metascript_paths)
        for iMetascript, metascript_path in enumerate(self.slurm_data.metascript_paths):  # Currently, only one metascript is generated.
            logging.info(f"Submitting jobs to Slurm (metascript {metascript_path}, {iMetascript+1} / {num_metascripts})...")
            result = subprocess.run(['bash', metascript_path], capture_output=True, text=True, check=True)
            print(result.stdout)

            # store job ID to monitor status
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                id = line.split()[-1]
                self.slurm_data.job_ids.append(id)

        # point to the new output in case old simulation outputs exist with original file name
        self.isFilenameAbbreviated = (self.master_input_file['short'].fullpath != self.master_input_file['original'].fullpath)



    def generate_slurm_sbatches(self, suffix='', partition='microcloud', nodelist=None, email=None, num_CPU=4, memory_limit='8G', time_limit_hrs=5, exe=None, output_folder=None, database=None):
        """
        Generate sbatch files to be submitted to Slurm workload manager.
        """
        self.slurm_data.set(partition, nodelist, suffix, email, num_CPU, memory_limit, time_limit_hrs)

        # defaults
        if exe is None:           exe, = nn.config.get(self.shortcuts.product_name, 'exe'),
        if output_folder is None: output_folder = self.output_folder_path['short']
        if database is None:      database = nn.config.get(self.shortcuts.product_name, 'database')
        license = nn.config.get(self.shortcuts.product_name, 'license')

        input_fullpaths = list(set(self.inputs['fullpaths_short']))  # avoid duplicates. For some reason, input file paths are duplicated (NOTE: set object does not preserve the order of elements!)
        self.slurm_data.create_sbatch_scripts(input_fullpaths, exe, output_folder, database, license, self.shortcuts.product_name)


    def clean_slurm(self):
        self.wait_slurm_jobs()
        self.delete_input_files()
        self.slurm_data.delete_sbatch_scripts()


    def wait_slurm_jobs(self, username):
        """
        Wait until all jobs in the queue disappear.
        """
        job_ids = self.slurm_data.job_ids
        if len(job_ids) <= 5:
            job_ids_print = [int(id) for id in self.slurm_data.job_ids]
        else:
            job_ids_print = "[" + job_ids[0] + ", " + job_ids[1] + ", ..., " + job_ids[-2] + ", " + job_ids[-1] + "]"
        stopwatch = 0
        while self.slurm_data.job_remaining(username):
            time.sleep(10)
            stopwatch += 10
            logging.info(f"Slurm job(s) {job_ids_print} running... ({stopwatch} sec)")


    ### Import methods #######################################################
    def import_from_excel(self, excel_file_path):
        """
        Enables postprocessing without raw simulation data.
        Useful when the sweep output occupies a lot of memory and the user wishes to delete them.
        """
        if not os.path.exists(excel_file_path):
            raise ValueError(f"Excel file {excel_file_path} does not exist!")
        try:
            self.outputs = pd.read_excel(excel_file_path)
        except:
            raise
        logging.info("Imported data from Excel file.")
        self.__str__()


    ### Export methods #######################################################
    def export_to_excel(self, excel_file_path, force_lightHole=False, bias=0):
        """
        The sweep data can be exported to an Excel file by:
            SweepManager.outputs.to_excel()
        For the available options, see pandas.DataFrame.
        """
        if self.is_slurm_simulation():
            self.wait_slurm_jobs()

        self.__calc_output_data(force_lightHole, bias)

        logging.info(f"Exporting data to Excel file:\n{excel_file_path}")
        from pathlib import Path
        filepath = Path(excel_file_path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.outputs.to_excel(filepath, columns=['sweep_coords', 'overlap', 'transition_energy_meV', 'transition_energy_micron', 'HH1-LH1', 'HH1-HH2', 'absorption_at_transition_energy_TE', 'absorption_at_transition_energy_TM', *self.sweep_space.get_variable_names()])


    def export_to_csv(self, csv_file_path, force_lightHole=False, bias=0):
        """
        The sweep data can be exported to a CSV file by:
            SweepManager.outputs.to_csv()
        For the available options, see pandas.DataFrame.
        """
        if self.is_slurm_simulation():
            self.wait_slurm_jobs()

        self.__calc_output_data(force_lightHole, bias)

        logging.info(f"Exporting data to CSV file:\n{csv_file_path}")
        from pathlib import Path
        filepath = Path(csv_file_path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.outputs.to_csv(filepath, columns=['sweep_coords', 'overlap', 'transition_energy_meV', 'transition_energy_micron', 'HH1-LH1', 'HH1-HH2', 'absorption_at_transition_energy_TE', 'absorption_at_transition_energy_TM', *self.sweep_space.get_variable_names()])


    def __calc_output_data(self, force_lightHole, bias):
        self.__calc_overlap(force_lightHole)
        self.__calc_transition_energies(force_lightHole)
        self.__calc_HH1_LH1_energy_differences()
        self.__calc_HH1_HH2_energy_differences()
        if self.shortcuts.product_name == 'nextnano.NEGF++':
            self.__calc_absorption_at_transition_energy(bias)


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
            raise RuntimeError("Simulation output does not exist for this sweep!")

        # validate argument
        self.__validate_sweep_variables(sweep_variable)

        sweep_space_reduced = self.sweep_space.extract_1D_line(sweep_variable)

        # plot dispersions
        for model in self.states_to_be_plotted.keys():
            for coords, outfolder in zip(self.outputs['sweep_coords'], self.__get_output_subfolder_paths()):
                if sweep_space_reduced.has_sweep_point(coords):
                    self.shortcuts.plot_dispersion(outfolder, np.amin(self.states_to_be_plotted[model]), np.amax(self.states_to_be_plotted[model]), savePDF=savePDF)
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
            raise RuntimeError("Simulation output does not exist for this sweep!")

        # validate argument
        self.__validate_sweep_variables(sweep_variable)

        # generate GIF
        self.shortcuts.generate_gif(self.master_input_file['original'], sweep_variable, self.sweep_space[sweep_variable], self.states_to_be_plotted)


    ### Optics analysis #######################################################
    def __calc_overlap(self, force_lightHole):
        """
        Compute overlaps and store them in self.data
        if not all overlaps have been calculated.
        """
        if not self.outputs['overlap'].isna().any():
            return
        logging.info("Calculating overlap...")
        self.outputs['overlap'] = self.__get_output_subfolder_paths().apply(self.shortcuts.calculate_overlap, force_lightHole=force_lightHole)
        # self.data['overlap_squared'] = self.data['overlap'].apply(common.absolute_squared)   # BUG: somehow the results become complex128, not float --> cannot be plotted


    def __calc_transition_energies(self, force_lightHole):
        """
        Get transition energies and store them in self.data
        if not all transition energies have been calculated
        """
        if not self.outputs['transition_energy_meV'].isna().any():
            return

        logging.info("Calculating transition energies...")
        if self.shortcuts.product_name in ['nextnano3', 'nextnano++']:
            self.outputs['transition_energy_eV'] = self.__get_output_subfolder_paths().apply(self.shortcuts.get_transition_energy, force_lightHole=force_lightHole)
        elif self.shortcuts.product_name == 'nextnano.NEGF++':
            self.outputs['transition_energy_eV'] = self.__get_output_subfolder_paths().apply(self.shortcuts.get_transition_energy)

        # Convert units
        self.outputs['transition_energy_meV']    = self.outputs['transition_energy_eV'] * CommonShortcuts.scale1ToMilli
        self.outputs['transition_energy_micron'] = self.outputs['transition_energy_eV'].apply(CommonShortcuts.electronvolt_to_micron)
        self.outputs['transition_energy_nm']     = self.outputs['transition_energy_micron'] * 1e3


    def __calc_HH1_LH1_energy_differences(self):
        """
        Get the energy difference HH1 - LH1 and store them in self.data
        if not all HH1-LH1 have been calculated
        """
        if not self.outputs['HH1-LH1'].isna().any():
            return
        logging.info("Calculating energy difference HH1 - LH1...")
        self.outputs['HH1-LH1'] = self.__get_output_subfolder_paths().apply(self.shortcuts.get_HH1_LH1_energy_difference)


    def __calc_HH1_HH2_energy_differences(self):
        """
        Get the energy difference HH1 - HH2 and store them in self.data
        if not all HH1-HH2 have been calculated
        """
        if not self.outputs['HH1-HH2'].isna().any():
            return
        logging.info("Calculating energy difference HH1 - HH2...")
        self.outputs['HH1-HH2'] = self.__get_output_subfolder_paths().apply(self.shortcuts.get_HH1_HH2_energy_difference)


    def __calc_absorption_at_transition_energy(self, bias):
        if not self.outputs['absorption_at_transition_energy_TE'].isna().any() and not self.outputs['absorption_at_transition_energy_TM'].isna().any():
            return
        logging.info("Extracting optical absorption at transition energy (TE polarization)...")
        self.outputs['absorption_at_transition_energy_TE'] = self.__get_output_subfolder_paths().apply(self.shortcuts.get_absorption_at_transition_energy, args=('TE',), bias=bias)
        logging.info("Extracting optical absorption at transition energy (TM polarization)...")
        self.outputs['absorption_at_transition_energy_TM'] = self.__get_output_subfolder_paths().apply(self.shortcuts.get_absorption_at_transition_energy, args=('z',), bias=bias)


    def plot_overlap_squared(self,
                             x_axis,
                             y_axis,
                             x_label=None,
                             y_label=None,
                             force_lightHole=False,
                             plot_title='',
                             figFilename=None,
                             colormap='Greys',
                             contour_value=None
                             ):
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
        contour_value : float, optional
            Specify the overlap squared at which to draw a contour line on top of the colormap

        Returns
        -------
        fig : matplotlib.figure.Figure object

        """
        if not self.__output_subfolders_exist():
            raise RuntimeError("Simulation output does not exist for this sweep!")

        # validate input
        self.__validate_sweep_variables(x_axis)
        self.__validate_sweep_variables(y_axis)

        self.__calc_overlap(force_lightHole)  # TODO: implement overlap calculation for NEGF 8kp

        # x- and y-axis coordinates and 2D array-like of overlap data
        x_values, y_values, overlap = self.__slice_data_for_colormap_2D('overlap', x_axis, y_axis, datatype=np.cdouble)   # complex double = two double-precision floats
        overlap_squared = np.absolute(overlap)**2

        assert np.amin(overlap_squared) >= 0


        # instantiate 2D color plot
        fig, ax = plt.subplots()
        if not plot_title: plot_title = "Envelope overlap"
        self.__setup_2D_color_plot(ax, x_axis, y_axis, x_label, y_label, plot_title, x_values, y_values, False)

        # color setting
        contour_color = 'white'

        from matplotlib import colors
        divnorm = colors.Normalize(vmin=0.)   # set the colorscale minimum to 0
        pcolor = ax.pcolormesh(x_values, y_values, overlap_squared, cmap=colormap, norm=divnorm, shading='auto')
        # pcolor = ax.pcolormesh(x_values, y_values, overlap_squared, cmap=colormap, shading='auto')

        cbar = fig.colorbar(pcolor)
        cbar.set_label("Envelope overlap")

        if contour_value is not None:
            CommonShortcuts.draw_contour(ax, x_values, y_values, overlap_squared, contour_value, contour_color)
            
        fig.tight_layout()
        plt.show()

        if figFilename is None or figFilename == "":
            name = os.path.split(self.__get_output_folder_path())[1]
            figFilename = name + "_overlap"
        self.shortcuts.export_figs(figFilename, "png", output_folder_path=self.__get_output_folder_path(), fig=fig)


        # write info to a file
        max_val, indices = CommonShortcuts.find_maximum(overlap_squared)
        y_index, x_index = indices
        filepath = os.path.join(self.__get_output_folder_path(), os.path.join("nextnanopy", "info.txt"))
        logging.info(f"Writing info to:\n{filepath}")
        f = open(filepath, "w")  # w = write = overwrite existing content
        f.write(f"Overlap squared maximum {max_val} at:\n")
        f.write(f"{x_axis} = {x_values[x_index]}\n")
        f.write(f"{y_axis} = {y_values[y_index]}\n")
        f.close()

        return fig


    def plot_transition_energies(self,
                                 x_axis,
                                 y_axis=None,
                                 x_label=None,
                                 y_label=None,
                                 force_lightHole=False,
                                 plot_title='',
                                 figFilename=None,
                                 colormap=None,
                                 set_center_to_zero=False,
                                 unit='meV',
                                 export_data=False,
                                 contour_energy=None
                                 ):
        """
        Plot the transition energy (lowest electron eigenenergy - highest hole eigenenergy) colormap as a function of two selected sweep axes.

        Parameters
        ----------
        x_axis : str
            sweep variable for x-axis
        y_axis : str, optional
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
        export_data : bool, optional
            If True, return the processed data ready for manual plot.
            If False, return figure
        contour_energy : float, optional
            Specify the energy at which to draw a contour line on top of the colormap
            In units of 'unit' argument.

        Returns
        -------
        If export_data,
            x_values, y_values, scaled transition energies
        else,
            matplotlib.figure.Figure object

        """
        if not self.__output_subfolders_exist():
            raise RuntimeError("Simulation output does not exist for this sweep!")

        # validate input
        self.__validate_sweep_variables(x_axis)
        if y_axis is None:
            plot_2D = False
        else:
            plot_2D = True
            self.__validate_sweep_variables(y_axis)
        if unit not in ['meV', 'micron', 'um', 'nm']:
            raise ValueError(f"Energy unit {unit} is not supported.")

        self.__calc_transition_energies(force_lightHole)

        # x- and y-axis coordinates and 2D array-like of overlap data
        if plot_2D:
            if unit == 'meV':
                x_values, y_values, transition_energies = self.__slice_data_for_colormap_2D('transition_energy_meV', x_axis, y_axis, datatype=np.double)
            elif unit == 'micron' or unit == 'um':
                x_values, y_values, transition_energies = self.__slice_data_for_colormap_2D('transition_energy_micron', x_axis, y_axis, datatype=np.double)
            elif unit == 'nm':
                x_values, y_values, transition_energies = self.__slice_data_for_colormap_2D('transition_energy_nm', x_axis, y_axis, datatype=np.double)
        else:
            if unit == 'meV':
                x_values, transition_energies = self.__slice_data_for_colormap_1D('transition_energy_meV', x_axis, datatype=np.double)
            elif unit == 'micron' or unit == 'um':
                x_values, transition_energies = self.__slice_data_for_colormap_1D('transition_energy_micron', x_axis, datatype=np.double)
            elif unit == 'nm':
                x_values, transition_energies = self.__slice_data_for_colormap_1D('transition_energy_nm', x_axis, datatype=np.double)


        if export_data:
            if plot_2D:
                return x_values, y_values, transition_energies
            else:
                return x_values, transition_energies
        else:
            if plot_2D:
                fig = self.__plot_transition_energies_2D(x_axis, y_axis, x_label, y_label, x_values, y_values, plot_title, colormap, set_center_to_zero, unit, transition_energies, contour_energy=contour_energy)
            else:
                fig = self.__plot_transition_energies_1D(x_axis, x_label, x_values, plot_title, unit, transition_energies)

            if figFilename is None or figFilename == "":
                name = os.path.split(self.__get_output_folder_path())[1]
                figFilename = name + "_transitionEnergies"
            self.shortcuts.export_figs(figFilename, "png", output_folder_path=self.__get_output_folder_path(), fig=fig)
            return fig



    def __plot_transition_energies_2D(self, x_axis, y_axis, x_label, y_label, x_values, y_values, plot_title, colormap, set_center_to_zero, unit, transition_energies_scaled, contour_energy=None):
        if transition_energies_scaled.ndim != 2:
            raise ValueError("Transition_energies_scaled must be two dimensional!")

        # instantiate 2D color plot
        fig, ax = plt.subplots()
        if not plot_title: plot_title = "Transition energies"
        self.__setup_2D_color_plot(ax, x_axis, y_axis, x_label, y_label, plot_title, x_values, y_values, False)

        # color setting
        colormap, contour_color = self.__determine_contour_color(colormap, set_center_to_zero)

        if set_center_to_zero:
            from matplotlib import colors
            divnorm = colors.TwoSlopeNorm(vcenter=0.)
            pcolor = ax.pcolormesh(x_values, y_values, transition_energies_scaled, cmap=colormap, norm=divnorm, shading='auto')
        else:
            pcolor = ax.pcolormesh(x_values, y_values, transition_energies_scaled, cmap=colormap, shading='auto')

        if contour_energy is not None:
            CommonShortcuts.draw_contour(ax, x_values, y_values, transition_energies_scaled, contour_energy, contour_color)

        cbar = fig.colorbar(pcolor, ax=ax)
        if unit == 'meV':
            cbar.set_label("Transition energy ($\mathrm{meV}$)")
        elif unit == 'micron' or unit == 'um':
            cbar.set_label("Wavelength ($\mu\mathrm{m}$)")
        elif unit == 'nm':
            cbar.set_label("Wavelength ($\mathrm{nm}$)")
        fig.tight_layout()
        plt.show()

        return fig


    def __determine_contour_color(self, colormap, set_center_to_zero):
        """
        Decide on the color for contour line depending on the colormap of the 2D plot.
        If colormap is None, set it to default.

        Returns
        -------
        colormap : Colormap
            Colormap for the 2D plot
        contour_color : str
            Name of color for contour line
        """
        if colormap is None:
            # default colors
            if set_center_to_zero:
                colormap = self.default_colors.colormap['divergent_bright']
            else:
                colormap = self.default_colors.colormap['linear_bright_bg']

        if colormap == self.default_colors.colormap['divergent_bright']:
            return colormap, 'black'
        elif colormap == 'viridis' or colormap == self.default_colors.colormap['linear_bright_bg']:
            return colormap, self.default_colors.lines_on_colormap['bright_bg']


    def __plot_transition_energies_1D(self, x_axis, x_label, x_values, plot_title, unit, transition_energies_scaled):
        if transition_energies_scaled.ndim != 1:
            raise ValueError("Transition_energies_scaled must be one dimensional!")

        # instantiate 1D plot
        fig, ax = plt.subplots()
        if not plot_title: plot_title = "Transition energies"
        self.__setup_1D_plot(ax, x_axis, x_label, plot_title, x_values)

        if unit == 'meV':
            ax.set_ylabel("Transition energy ($\mathrm{meV}$)")
        elif unit == 'micron' or unit == 'um':
            ax.set_ylabel("Wavelength ($\mu\mathrm{m}$)")
        elif unit == 'nm':
            ax.set_ylabel("Wavelength ($\mathrm{nm}$)")
        ax.plot(x_values, transition_energies_scaled)
        fig.tight_layout()
        plt.show()

        return fig


    ### highest hole states ###################################################
    def plot_HH1_LH1_energy_difference(self,
                                       x_axis,
                                       y_axis,
                                       x_label=None,
                                       y_label=None,
                                       plot_title='',
                                       figFilename=None,
                                       colormap=None,
                                       set_center_to_zero=True,
                                       contour_energy_meV=None
                                       ):
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
        contour_energy_meV : float, optional
            Specify the energy at which to draw a contour line on top of the colormap

        Returns
        -------
        fig : matplotlib.figure.Figure object

        """
        if not self.__output_subfolders_exist():
            raise RuntimeError("Simulation output does not exist for this sweep!")

        # validate input
        self.__validate_sweep_variables(x_axis)
        self.__validate_sweep_variables(y_axis)

        self.__calc_HH1_LH1_energy_differences()

        # x- and y-axis coordinates and 2D array-like of overlap data
        x_values, y_values, EDifference = self.__slice_data_for_colormap_2D('HH1-LH1', x_axis, y_axis, datatype=np.double)


        # instantiate 2D color plot
        fig, ax = plt.subplots()
        if not plot_title: plot_title = "Energy difference HH1 - LH1"
        self.__setup_2D_color_plot(ax, x_axis, y_axis, x_label, y_label, plot_title, x_values, y_values, False)

        # color setting
        colormap, contour_color = self.__determine_contour_color(colormap, set_center_to_zero)

        if set_center_to_zero:
            from matplotlib import colors
            divnorm = colors.TwoSlopeNorm(vcenter=0.)
            pcolor = ax.pcolormesh(x_values, y_values, EDifference * CommonShortcuts.scale1ToMilli, cmap=colormap, norm=divnorm, shading='auto')
        else:
            pcolor = ax.pcolormesh(x_values, y_values, EDifference * CommonShortcuts.scale1ToMilli, cmap=colormap, shading='auto')

        if contour_energy_meV is not None:
            CommonShortcuts.draw_contour(ax, x_values, y_values, EDifference * CommonShortcuts.scale1ToMilli, contour_energy_meV, contour_color)
            
        cbar = fig.colorbar(pcolor, ax=ax)
        cbar.set_label("Hole energy difference HH1-LH1 ($\mathrm{meV}$)")
        fig.tight_layout()
        plt.show()

        if figFilename is None or figFilename == "":
            name = os.path.split(self.__get_output_folder_path())[1]
            figFilename = name + "_HH1_LH1_EnergyDifference"
        self.shortcuts.export_figs(figFilename, "png", output_folder_path=self.__get_output_folder_path(), fig=fig)

        return fig


    def plot_HH1_HH2_energy_difference(self,
                                       x_axis,
                                       y_axis,
                                       x_label=None,
                                       y_label=None,
                                       plot_title='',
                                       figFilename=None,
                                       colormap=None,
                                       set_center_to_zero=True,
                                       contour_energy_meV=None
                                       ):
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
        contour_energy_meV : float, optional
            Specify the energy at which to draw a contour line on top of the colormap

        Returns
        -------
        fig : matplotlib.figure.Figure object

        """
        if not self.__output_subfolders_exist():
            raise RuntimeError("Simulation output does not exist for this sweep!")

        # validate input
        self.__validate_sweep_variables(x_axis)
        self.__validate_sweep_variables(y_axis)

        self.__calc_HH1_HH2_energy_differences()

        # x- and y-axis coordinates and 2D array-like of overlap data
        x_values, y_values, EDifference = self.__slice_data_for_colormap_2D('HH1-HH2', x_axis, y_axis, datatype=np.double)


        # instantiate 2D color plot
        fig, ax = plt.subplots()
        if not plot_title: plot_title = "Energy difference HH1 - HH2"
        self.__setup_2D_color_plot(ax, x_axis, y_axis, x_label, y_label, plot_title, x_values, y_values, False)

        # color setting
        colormap, contour_color = self.__determine_contour_color(colormap, set_center_to_zero)

        if set_center_to_zero:
            from matplotlib import colors
            divnorm = colors.TwoSlopeNorm(vcenter=0.)
            pcolor = ax.pcolormesh(x_values, y_values, EDifference * CommonShortcuts.scale1ToMilli, cmap=colormap, norm=divnorm, shading='auto')
        else:
            pcolor = ax.pcolormesh(x_values, y_values, EDifference * CommonShortcuts.scale1ToMilli, cmap=colormap, shading='auto')

        if contour_energy_meV is not None:
            CommonShortcuts.draw_contour(ax, x_values, y_values, EDifference * CommonShortcuts.scale1ToMilli, contour_energy_meV, contour_color)
            
        cbar = fig.colorbar(pcolor, ax=ax)
        cbar.set_label("Hole energy difference HH1-HH2 ($\mathrm{meV}$)")
        fig.tight_layout()
        plt.show()

        if figFilename is None or figFilename == "":
            name = os.path.split(self.__get_output_folder_path())[1]
            figFilename = name + "_HH1_HH2_EnergyDifference"
        self.shortcuts.export_figs(figFilename, "png", output_folder_path=self.__get_output_folder_path(), fig=fig)

        return fig


    ### Transport analysis ####################################################
    def __calc_average_current(self, bias):
        """
        Compute spatial average of current density [A/cm^2] (1D structure) and store them in self.outputs
        if not all currents have been calculated.
        """
        if not self.outputs['ave_current'].isna().any():
            return
        logging.info("Calculating current...")
        self.outputs['ave_current'] = self.__get_output_subfolder_paths().apply(self.shortcuts.calculate_average_current, bias=bias, is_fullpath=True)

    def plot_average_current(self,
                               x_axis,
                               y_axis,
                               bias,
                               x_label=None,
                               y_label=None,
                               plot_title='',
                               figFilename=None,
                               colormap=None,
                               logscale=False
                               ):
        """
        Plot the colormap of the spatial average of current density as a function of two selected sweep axes.
        Note that, in steady states of periodic structures, the current density should be constant over the simulation region to conserve charge.

        Parameters
        ----------
        x_axis : str
            sweep variable for x-axis
        y_axis : str
            sweep variable for y-axis
        bias : float
            potential drop per period at which the current data should be extracted
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
        logscale : bool, optional
            If True, x, y, and z-axes are set to logscale.

        Returns
        -------
        fig : matplotlib.figure.Figure object

        """
        if not self.__output_subfolders_exist():
            raise RuntimeError("Simulation output does not exist for this sweep!")

        # validate input
        self.__validate_sweep_variables(x_axis)
        self.__validate_sweep_variables(y_axis)

        self.__calc_average_current(bias)

        # x- and y-axis coordinates and 2D array-like of overlap data
        x_values, y_values, ave_current = self.__slice_data_for_colormap_2D('ave_current', x_axis, y_axis, datatype=np.double)

        # instantiate 2D color plot
        fig, ax = plt.subplots()
        if not plot_title: plot_title = "Spatial average of current density"
        self.__setup_2D_color_plot(ax, x_axis, y_axis, x_label, y_label, plot_title, x_values, y_values, logscale)

        # color setting
        colormap, contour_color = self.__determine_contour_color(colormap, False)

        from matplotlib import colors
        if logscale:
            norm = colors.LogNorm()
        else:
            norm = colors.Normalize(vmin=0.)  # set the colorscale minimum to 0
        pcolor = ax.pcolormesh(x_values, y_values, ave_current, cmap=colormap, norm=norm, shading='auto')

        if logscale:
            format_str = '%.0e'
        else:
            format_str = None
        cbar = fig.colorbar(pcolor, ax=ax, format=format_str)
        cbar.set_label("Current density [$\mathrm{A}/\mathrm{cm}^2$]")

        fig.tight_layout()
        plt.show()

        if figFilename is None or figFilename == "":
            name = os.path.split(self.__get_output_folder_path())[1]
            figFilename = name + "_average_current"
        self.shortcuts.export_figs(figFilename, "png", output_folder_path=self.__get_output_folder_path(), fig=fig)

        return fig


    ### in-plane k ###########################################################
    def plot_inplaneK(self):
        if self.isFilenameAbbreviated:
            raise RuntimeError("recover_original_filenames() must be called before!")

        inplane_k = self.shortcuts.getKPointsData1D_in_folder(self.outputs.loc[0, 'output_subfolder_original'])   # assuming k points are identical to all the sweeps
        return self.shortcuts.plot_inplaneK(inplane_k)


