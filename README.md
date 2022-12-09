# nextnanopy-wrapper
The package `nnShortcuts` contains shortcut functions using nextnanopy features and simplifies the pre- and post-processing of nextnano simulations.

The package `SweepHelper` is an object-oriented wrapper around nextnanopy. This class object facilitates preparing, running and post-processing nextnano sweep simulations by bridging the simulation inputs and outputs.

## How to use
```python
# nextnano++ shortcut functions
import nnShortcuts.nnp_shortcuts as nnp

# nextnano3 shortcut functions
import nnShortcuts.nn3_shortcuts as nn3

# nextnano.NEGF shortcut functions
import nnShortcuts.NEGF_shortcuts as negf

# class module for sweep simulations
import SweepHelper as helper
```

## Limitations
Currently, the SweepHelper class only supports sweep simulation of nextnano++.

## Contributing
Any idea of improvement, extension and bug fix is welcome. If you already know the solution, please create a branch and submit a pull request. Else, feel free to file an issue.