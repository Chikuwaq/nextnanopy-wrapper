# add the nnShortcut directory to sys.path, so that this package can import modules from there
import sys

module_path = '../nnShortcuts'

if not module_path in sys.path:
  sys.path.append(module_path)

from nnShortcuts import common, nnp_shortcuts