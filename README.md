# nextnanopy-wrapper
The package `nnShortcuts` contains shortcut functions using nextnanopy features and simplifies the pre- and post-processing of nextnano simulations.

The package `SweepHelper` is an object-oriented wrapper around nextnanopy. This class object facilitates preparing, running and post-processing nextnano sweep simulations by bridging the simulation inputs and outputs.


## How to use
The shortcut functions are available if you import respective modules:
```python
# nextnano++ shortcut functions
import nnShortcuts.nnp_shortcuts as nnp

# nextnano3 shortcut functions
import nnShortcuts.nn3_shortcuts as nn3

# nextnano.NEGF shortcut functions
import nnShortcuts.NEGF_shortcuts as negf

```

A SweepHelper object allows you to execute a sweep simulation and its post-processing. 
```python
from nnHelpers import SweepHelper

helper = SweepHelper(<sweep ranges>, <nextnanopy.InputFile `object`>, <kwargs options>)

helper.execute_sweep(parallel_limit=<number of CPUs available>)

helper.plot_transition_energies(<x axis>, <y axis>, <kwargs options>)
```
See examples for implemented features and the options.


## Limitations
Currently, the SweepHelper class only supports sweep simulation of nextnano++.


## Contributing
Any idea of improvement, extension and bug fix is welcome. If you already know the solution, please create a branch and submit a pull request. Else, feel free to file an issue.