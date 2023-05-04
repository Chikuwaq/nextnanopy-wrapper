import unittest

from nnShortcuts.common import CommonShortcuts

import nextnanopy
import numpy as np

some_positive_int = 3
some_negative_int = -2
some_ndArray = np.array([3, 5.2, -2.4])

some_file_path_linux_in   = r'/home/user/nextnano/input_file.in'
some_file_path_linux_negf = r'/home/user/nextnano/input_file.negf'

some_input_file_obj_nnp  = nextnanopy.InputFile(r'tests/sample_nnp.in')
some_input_file_obj_negf = nextnanopy.InputFile(r'tests/sample_negf.negf')


class Test_common(unittest.TestCase):
    shortcuts = CommonShortcuts()
    

    def test_is_harf_integer(self):
        self.assertTrue(self.shortcuts.is_half_integer(0.5))
        self.assertTrue(self.shortcuts.is_half_integer(1.5))
        self.assertFalse(self.shortcuts.is_half_integer(some_positive_int))
        self.assertFalse(self.shortcuts.is_half_integer(some_negative_int))

    def test_find_maximum(self):
        max_val, indices = self.shortcuts.find_maximum(some_ndArray)
        self.assertEqual(max_val, 5.2)
        self.assertEqual(indices[0], 1)
        self.assertRaises(TypeError, self.shortcuts.find_maximum, some_positive_int)

    def test_find_minimum(self):
        min_val, indices = self.shortcuts.find_minimum(some_ndArray)
        self.assertEqual(min_val, -2.4)
        self.assertEqual(indices[0], 2)
        self.assertRaises(TypeError, self.shortcuts.find_minimum, some_positive_int)

    def test_separate_file_extension(self):
        filename_no_ext, ext = self.shortcuts.separate_extension(some_file_path_linux_in)
        self.assertEqual(filename_no_ext, 'input_file')
        self.assertEqual(ext, '.in')

        filename_no_ext, ext = self.shortcuts.separate_extension(some_file_path_linux_negf)
        self.assertEqual(filename_no_ext, 'input_file')
        self.assertEqual(ext, '.negf')

        self.assertRaises(RuntimeError, self.shortcuts.separate_extension, 'file.txt')

    def test_get_shortcut(self):
        from nnShortcuts.nnp_shortcuts import nnpShortcuts
        obj = self.shortcuts.get_shortcut(some_input_file_obj_nnp)
        self.assertTrue(isinstance(obj, nnpShortcuts))
        
        from nnShortcuts.NEGF_shortcuts import NEGFShortcuts
        obj = self.shortcuts.get_shortcut(some_input_file_obj_negf)
        self.assertTrue(isinstance(obj, NEGFShortcuts))
        




    
if __name__ == '__main__':
    unittest.main()