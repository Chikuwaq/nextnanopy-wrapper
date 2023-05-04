import unittest

from nnShortcuts.NEGF_shortcuts import NEGFShortcuts
import nextnanopy
import numpy as np

some_positive_int = 3
some_negative_int = -2
some_ndArray = np.array([3, 5.2, -2.4])

some_file_path_linux_negf = r'/home/user/nextnano/input_file.negf'

# some_input_file_obj_negf = nextnanopy.InputFile(r'tests/sample_negf.negf')


class Test_NEGF_shortcuts(unittest.TestCase):
    shortcuts = NEGFShortcuts()
    folder = r"tests/8band_WRegion_InAsGaInSb_nnpParam_strained"


    def test_get_transition_energy(self):
        dE = self.shortcuts.get_transition_energy(self.folder)
        self.assertEqual(round(dE, 3), 0.356)


    def test_get_DataFiles_in_folder(self):
        dfs = self.shortcuts.get_DataFiles_in_folder(['spinor_composition'], self.folder)
        self.assertEqual(len(dfs), 1)  # NEGF outputs one in SXYZ basis, another in CbHhLhSo basis. However, I deleted the first in unittest material


    def test_get_states_to_be_plotted(self):
        pass

    def test_get_DataFile_probabilities_in_folder(self):
        pass


    def test_find_highest_hole_state_atK0(self):
        index = self.shortcuts.find_highest_hole_state_atK0(self.folder)
        self.assertEqual(index, 7)  # state index is base 0


    def test_find_lowest_electron_state_atK0(self):
        index = self.shortcuts.find_lowest_electron_state_atK0(self.folder)
        self.assertEqual(index, 8)  # state index is base 0

if __name__ == '__main__':
    unittest.main()