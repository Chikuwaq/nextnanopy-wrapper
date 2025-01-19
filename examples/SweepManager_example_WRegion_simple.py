"""
Created on 2022/05/22
Modified on 2025/01/19

This script demonstrates the use of SweepManager class.
Structure: W-shaped quantum well of the interband cascade laser design in [Vurgaftman2011]

@author: takuma.sato@nextnano.com
"""

import nextnanopy as nn
from boostsweep.sweep_manager import SweepManager

# Specify input file.
path = r"Density_WRegion_kp8_nnp_rescaleS1.in"

# Specify sweep variables, their ranges and number of simulation points.
# {'sweep variable': ([min, max], number of points)}
sweep_ranges = {
    'eWell1_width': ([2, 2.4], 4),
    # 'hWell_width': ([4, 5], 2),
    # 'InAsSb_well_left': ([2, 4], 2),
    'AlloyContent_hWell': ([0.5, 0.56], 4)
}

# Dispersion plot settings
key_for_dispersion_plot = 'InAs_well_left'
eigenstate_ranges = {
    # 'Gamma': [],
    'kp6': [17, 26],
    'kp8': [17, 26]
}

# Specify plot axes for 2D color plot.
x_axis = 'eWell1_width'
# y_axis = 'hWell_width'
# x_axis = 'InAsSb_well_left'
y_axis = 'AlloyContent_hWell'


# --- MAIN --------------------------------------------------------------------
helper = SweepManager(sweep_ranges, nn.InputFile(path), eigenstate_range=eigenstate_ranges, round_decimal=3)
helper.execute_sweep(parallel_limit=4)

helper.delete_input_files()

helper.plot_transition_energies(x_axis, y_axis,
                    x_label="InAs thickness (nm)",
                    y_label="GaInSb thickness (nm)",
                    # force_lightHole=True,
                    # plot_title="Transition energies",
                    # figFilename="",
                    colormap="viridis",  # see https://matplotlib.org/stable/tutorials/colors/colormaps.html
                    unit="micron",  # 'meV' 'micron' 'um' or 'nm'
                    )

helper.plot_HH1_LH1_energy_difference(x_axis, y_axis,
                    x_label="InAs thickness (nm)",
                    y_label="GaInSb thickness (nm)",
                    # plot_title="Hole energy difference",
                    # figFilename="",
                    colormap="seismic",  # see https://matplotlib.org/stable/tutorials/colors/colormaps.html
                    )

helper.plot_overlap_squared(x_axis, y_axis,
                    x_label="InAs thickness (nm)",
                    y_label="GaInSb thickness (nm)",
                    # force_lightHole=True,
                    # plot_title="Overlap",
                    # figFilename="",
                    colormap="Greys",  # see https://matplotlib.org/stable/tutorials/colors/colormaps.html
                    )

# print(helper.data)

# Export data to an Excel file
path_Excel = r"D:\nextnano Users\takuma.sato\nextnano\Output_temporary\nextnanopy\shortname_kp6.xlsx"

from pathlib import Path
filepath = Path(path_Excel)
filepath.parent.mkdir(parents=True, exist_ok=True)
helper.outputs.to_excel(filepath)
