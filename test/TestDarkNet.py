import unittest
from src import DarkNet as dn

# Unit Tests for CaptureStream.py
class TestDarkNet(unittest.TestCase):

        def test_default_constructor(self):
            darknet =  dn.DarkNet()
            self.assertEqual(darknet.darknet_base_dir, '/root/darknet')
            self.assertEqual(darknet.darknet_object_data, 'object.data')
            self.assertEqual(darknet.yolov3_config_file, 'yolov3.cfg')
            self.assertEqual(darknet.yolov3_weights_file, 'darknet53.conv.74')
            self.assertEqual(darknet.log_file, 'out.log')

        def test_default_override_constructor(self):
            darknet =  dn.DarkNet('a', 'b', 'c', 'd', 'e')
            self.assertEqual(darknet.darknet_base_dir, 'a')
            self.assertEqual(darknet.darknet_object_data, 'b')
            self.assertEqual(darknet.yolov3_config_file, 'c')
            self.assertEqual(darknet.yolov3_weights_file, 'd')
            self.assertEqual(darknet.log_file, 'e')

if __name__ == '__main__':
    unittest.main()