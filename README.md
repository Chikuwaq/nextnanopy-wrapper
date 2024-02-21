# nextnanopy-wrapper
The package `nnShortcuts` contains shortcut functions using nextnanopy features and simplifies the pre- and postprocessing of nextnano simulations.
A big advantage of `nnShortcuts.common` is that you get a DataFile without knowing the exact output file name. You can narrow down the output data by specifying a list of keywords and exclude keywords.

The package `sweep-helper` is an object-oriented wrapper around [nextnanopy](https://github.com/nextnanopy/nextnanopy). The class `SweepHelper` 
1. facilitates preparing, running and postprocessing nextnano sweep simulations by bridging the simulation inputs and outputs. This is in contrast to nextnanopy, which does not associate simulation input with output.
2. create a copy of your input file with a shorter name if the original filename is too long, so that the output path length does not exceed the system limit. The method `SweepHelper.execute_sweep()` will bring back the original filename to the output folders after the simulations have finished.
3. can submit jobs to Slurm workload manager. It waits for completion of all jobs before proceeding with postprocessing.

![alt text](/docs/images/nnp_shortcuts.png)
![alt text](/docs/images/NEGF_shortcuts.png)


## How to use
After 'git pull'ing this repo, the shortcut functions are available if you import respective modules:
```python
# nextnano++ shortcut functions
import nnShortcuts.nnp_shortcuts as nnp

# nextnano3 shortcut functions
import nnShortcuts.nn3_shortcuts as nn3

# nextnano.NEGF shortcut functions
import nnShortcuts.NEGF_shortcuts as negf

```

A SweepHelper object allows you sweep execution and its postprocessing. 
```python
import nextnanopy as nn
from sweep_helper import SweepHelper

helper = SweepHelper(<sweep ranges>, nn.InputFile(<input file path>), <kwargs options>)

helper.execute_sweep(parallel_limit=<number of CPUs available>)

helper.plot_transition_energies(<x axis>, <y axis>, <kwargs options>)
...
```

## Examples
Please refer to the examples under `examples/`. 
The following example in the [nextnanopy](https://github.com/nextnanopy/nextnanopy) repository also uses this wrapper to perform a voltage sweep and postprocessing:
* `nextnanopy/templates/InterbandTunneling_Duboz2019_doc.ipynb` (documentation)
* `InterbandTunneling_Duboz2019_formulation.pdf` (formulation)
* `InterbandTunneling_Duboz2019_nnp.py` (Python script)



## Limitations
Currently, the SweepHelper class only supports sweep simulation of nextnano++ and nextnano.NEGF (C++ version).


## Contributing
Any idea of improvement, extension and bug fix is welcome. If you already know the solution, please create a branch and submit a pull request. Else, feel free to file an issue.