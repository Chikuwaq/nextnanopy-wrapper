import unittest

from nnShortcuts.common import CommonShortcuts

class Test_common(unittest.TestCase):
    common = CommonShortcuts()
    def test_is_harf_integer(self):
        self.assertTrue(self.common.is_half_integer(1.5))
        self.assertFalse(self.common.is_half_integer(3))

if __name__ == '__main__':
    unittest.main()