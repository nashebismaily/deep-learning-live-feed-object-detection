import unittest
from src.main.model import DarkNet as dn

# Unit Tests for CaptureStream.py
class TestDarkNet(unittest.TestCase):

        darknet_base_dir = '/root/darknet'
        darknet_object_data = 'object.data'
        yolov3_config_file = 'yolov3.cfg'
        yolov3_weights_file = 'darknet53.conv.74'
        log_file = 'out.log'
        graph_file = 'darknetlossgraph.png'

        # tests the constructor for DarkNet()
        def test_default_constructor(self):
            darknet =  dn.DarkNet(self.darknet_base_dir, self.darknet_object_data, self.yolov3_config_file, self.yolov3_weights_file, self.log_file, self.graph_file)
            self.assertEqual(darknet.darknet_base_dir, '/root/darknet')
            self.assertEqual(darknet.darknet_object_data, 'object.data')
            self.assertEqual(darknet.yolov3_config_file, 'yolov3.cfg')
            self.assertEqual(darknet.yolov3_train_weights_file, 'darknet53.conv.74')
            self.assertEqual(darknet.log_file, 'out.log')

if __name__ == '__main__':
    unittest.main()