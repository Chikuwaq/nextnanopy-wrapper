#!/path/to/my/folder/.venv/bin/python
"""
This is a template for postprocessing after a sweep simulation.
See 'job_sweep.py' for the job submission.
This separation of input file execution and postprocessing frees the user from keeping the terminal open for the whole duration of the sweep simulation,
which can be days if single simulation takes long or the number of sweep parameter combinations is huge.

Paths, input file names, and sweep ranges must be adjusted by the user.
"""

import os

import nextnanopy as nn
from boostsweep.sweep_manager import SweepManager


def calculate_something(shortcut, outfolder_path, bias_mV):
    datafile_density = shortcut.get_DataFile_NEGF_atBias(["CarrierDensity_ElectronHole"], outfolder_path, bias_mV, is_fullpath=True)
    e_density = datafile_density.variables['Electron density'].value
    h_density = datafile_density.variables['Hole density'].value
    x = datafile_density.coords['Position'].value

    # --- Integrate carrier density in real space  ---------------------------------------------------------------
    from scipy import integrate

    e_sheet_density = integrate.simpson(e_density, x=x)
    print(f"Electron sheet density (one period) = {e_sheet_density} [10^11 cm^-2]")

    # --- Extract gain ------------------------------------------------------------
    datafile_gain = shortcut.get_DataFile_in_folder(["Gain_vs_Voltage", ".dat"], outfolder_path)
    gain = datafile_gain.variables["Maximum gain"].value[0]
    print(f"Gain at threshold = {gain} (cm^-1)")
    return gain


# --- specify sweep
input_folder = r"/path/to/input_files"

appendix = "dz0.15_NVBO_axial1000500_screen4"
input_filename = f"ICL_Vurgaftman_{appendix}"
input_path_sweepV = os.path.join(input_folder, input_filename + ".negf") 

sweep_var_key = 'doping2D'
sweep_var_values = [2.38e11, 4.76e11, 9.4e11, 18.8e11, 47e11, 65.8e11, 88e11, 107e11]
# sweep_var_key = 'In'
# sweep_var_values = ([0.0, 0.07], 8)

sweep_ranges = {
    sweep_var_key: sweep_var_values,
    'In': [0.0]  # a fixed value
}


# --- main function
manager = SweepManager(sweep_ranges, nn.InputFile(input_path_sweepV), round_decimal=3)

if not manager.output_subfolders_exist_with_originalname():
    # NOTE: using output directory specified in nextnanopy config.
    # Output subfolders might not exist if you have submitted many simulations to the SLURM queue and have not given it enough time to digest.
    raise RuntimeError("Output subfolders do not exist!")

sweep_values = []
threshold_Vs = []
calculated_values = []

for i, row in manager.outputs.iterrows():
    subfolder = row['output_subfolder_original']
    
    sweep_var_value = row[sweep_var_key]
    print(f"\nPostprocessing simulation at {sweep_var_key} = {sweep_var_value}")
    sweep_values.append(sweep_var_value)

    V = manager.shortcuts.extract_threshold_voltage(subfolder)
    threshold_Vs.append(f"{V:.3f}")

    calculated_value = calculate_something(manager.shortcuts, subfolder, V)
    calculated_values.append(calculated_value)


print("\nSummary of sweep:")
print("Sweep values: ", sweep_values)
print("Threshold voltage: ", threshold_Vs)
print("Calculated values: ", calculated_values)
