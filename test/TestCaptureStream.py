import unittest
from src import CaptureStream as cs


# Unit Tests for CaptureStream.py
class TestCaptureStream(unittest.TestCase):

        def test_get_device_list(self):
            device_list = [0]
            self.assertListEqual(cs.get_device_list(), device_list)

if __name__ == '__main__':
    unittest.main()