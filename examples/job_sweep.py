#!/path/to/my/folder/.venv/bin/python
"""
This is a template for submitting a sweep simulation to Slurm workload manager commonly used in linux computing clusters.

Paths, input file names, and sweep ranges must be adjusted by the user.
"""

import nextnanopy as nn

from boostsweep.sweep_manager import SweepManager

# Specify sweep variables, their ranges and number of simulation points.
# { 'sweep variable': ([start, last], number of points) }
# or
# { 'sweep variable': [list of values] }  # note that the colormap plot will always be equidistant


# quasi-Eq ICL series
appendix = "dz0.15_NVBO_axial1000500_screen4"

input_path = r"/path/to/input_files/ICL_Vurgaftman_" + appendix + ".negf"

def submit_sweep(sweep_ranges, input_path):
    manager = SweepManager(sweep_ranges, nn.InputFile(input_path), round_decimal=3)
    manager.submit_sweep_to_slurm(suffix='', partition='zencloud', nodelist=None, email="takuma.sato@nextnano.com", num_CPU=32, memory_limit='120G', time_limit_hrs=10)


sweep_ranges = {
    'doping2D': [2.38e11, 4.76e11, 9.4e11, 18.8e11, 47e11, 65.8e11, 88e11, 107e11],
    'In': ([0.0, 0.07], 8)
}

submit_sweep(sweep_ranges, input_path)

