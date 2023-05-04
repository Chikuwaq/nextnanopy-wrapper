import unittest

import nextnanopy as nn
from nnShortcuts.nnp_shortcuts import nnpShortcuts

class Test_nnp_shortcuts(unittest.TestCase):
    shortcuts = nnpShortcuts()
    folder = r"tests/output/WRegion_BorislavDesign2_wfTest_kp8_nnp_0field_eval_rescaleS1"


    def test_get_DataFiles_in_folder(self):
        dfs = self.shortcuts.get_DataFiles_in_folder(['spinor_composition', 'CbHhLhSo'], self.folder)
        self.assertEqual(len(dfs), 25)


    def test_get_states_to_be_plotted(self):
        datafiles_probability_dict = self.shortcuts.get_DataFile_probabilities_with_name(self.folder)
        
        # state range
        states_to_be_plotted, num_evs = self.shortcuts.get_states_to_be_plotted(datafiles_probability_dict, states_range_dict={'kp8': [9, 16]})
        self.assertIsInstance(states_to_be_plotted['kp8'], list)
        self.assertEqual(num_evs['kp8'], 16)  # number of all eigenvalues
        self.assertEqual(states_to_be_plotted['kp8'], [8,9,10,11,12,13,14,15])  # state index is base 0

        # state list
        states_to_be_plotted, num_evs = self.shortcuts.get_states_to_be_plotted(datafiles_probability_dict, states_list_dict={'kp8': ['lowestElectron', 'highestHole', 15, 16]})
        self.assertIsInstance(states_to_be_plotted['kp8'], list)
        self.assertEqual(num_evs['kp8'], 16)  # number of all eigenvalues
        self.assertEqual(states_to_be_plotted['kp8'], [10,13,14,15])  # state index is base 0

        # occupied states
        with self.assertRaises(ValueError):
            self.shortcuts.get_states_to_be_plotted(datafiles_probability_dict, states_list_dict={'kp8': ['occupied']})
        
        states_to_be_plotted, num_evs = self.shortcuts.get_states_to_be_plotted(datafiles_probability_dict, states_list_dict={'kp8': ['occupied'], 'cutoff_occupation': 1e12})
        self.assertIsInstance(states_to_be_plotted['kp8'], list)
        self.assertEqual(num_evs['kp8'], 16)  # number of all eigenvalues
        self.assertEqual(states_to_be_plotted['kp8'], [0,1,2,3,4,5,6,7,8,9,10,11])  # state index is base 0


    def test_get_DataFile_probabilities_in_folder(self):
        dfs_probabilities = self.shortcuts.get_DataFile_probabilities_in_folder(self.folder)

        # number of probabilities_shift data files
        self.assertEqual(len(dfs_probabilities['kp8']), 25)  
        self.assertEqual(len(dfs_probabilities.keys()), 1)
        
        for df in dfs_probabilities['kp8']:
            self.assertIsInstance(df, nn.DataFile)


    def test_find_highest_hole_state_atK0(self):
        index = self.shortcuts.find_highest_hole_state_atK0(self.folder)
        self.assertEqual(index, 13)  # state index is base 0


    def test_find_lowest_electron_state_atK0(self):
        index = self.shortcuts.find_lowest_electron_state_atK0(self.folder)
        self.assertEqual(index, 10)  # state index is base 0


if __name__ == '__main__':
    unittest.main()