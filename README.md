# nextnanopy-wrapper
The package `nnShortcuts` contains shortcut functions using nextnanopy features and simplifies the pre- and post-processing of nextnano simulations.
A big advantage of `nnShortcuts.common` is that you get a DataFile without knowing the exact output file name. You can narrow down the output data by specifying a list of keywords and exclude keywords.

The module `nnHelpers` is an object-oriented wrapper around [nextnanopy](https://github.com/nextnanopy/nextnanopy). The class `SweepManager` object 
1. facilitates preparing, running and post-processing nextnano sweep simulations by bridging the simulation inputs and outputs. This is in contrast to nextnanopy, which does not associate simulation input and output.
2. create a copy of your input file with a shorter name if the original filename is too long, so that the output path length does not exceed the system limit. The method `SweepManager.execute_sweep()` will bring back the original filename to the output folders after the simulations have finished.

![alt text](/docs/images/nnp_shortcuts.png)
![alt text](/docs/images/NEGF_shortcuts.png)


## How to use shortcuts
After 'git pull'ing this repo, the shortcut functions are available if you import respective modules:
```python
# nextnano++ shortcut functions
from nnShortcuts.nnp_shortcuts import nnpShortcuts

# nextnano3 shortcut functions
from nnShortcuts.nn3_shortcuts import nn3Shortcuts

# nextnano.NEGF shortcut functions
from nnShortcuts.NEGF_shortcuts import NEGFShortcuts

```

## How to use SweepManager
A SweepManager object allows you sweep execution and its post-processing. 
```python
from boostsweep.sweep_manager import SweepManager

manager = SweepManager(<sweep ranges>, <nextnanopy.InputFile object>, <kwargs options>)

manager.execute_sweep(parallel_limit=<number of CPUs available>)

manager.plot_transition_energies(<x axis>, <y axis>, <kwargs options>)
...
```

## How to use nextnano colormap
![alt text](/docs/images/nextnano_colors.png)
```python
from nnShortcuts.nextnano_colors import NextnanoColor

cmap1 = NextnanoColor.cmap['divergent_dark']
cmap2 = NextnanoColor.cmap['divergent_bright']
cmap3 = NextnanoColor.cmap['linear']
# Use one of these for e.g. matplotlib's colormap parameter.

```


## Examples
Please refer to the examples in `examples/`. 
The following example in the [nextnanopy](https://github.com/nextnanopy/nextnanopy) repository also uses this wrapper to perform a voltage sweep and postprocessing:
* `nextnanopy/templates/InterbandTunneling_Duboz2019_doc.ipynb` (documentation)
* `InterbandTunneling_Duboz2019_formulation.pdf` (formulation)
* `InterbandTunneling_Duboz2019_nnp.py` (Python script)



## Limitations


## Contributing
Any idea of improvement, extension and bug fix is welcome. If you already know the solution, please create a branch and submit a pull request. Else, feel free to file an issue.