import unittest
import cv2 as cv
from src import CaptureStream as cs

# Unit Tests for CaptureStream.py
class TestCaptureStream(unittest.TestCase):

        def test_get_device_list(self):
            device_list = [0]
            self.assertListEqual(cs.get_device_list(), device_list)

        def test_get_camera_stream(self):
            self.assertIsInstance(cs.get_camera_stream(0), cv.VideoCapture)

        def test_close_stream(self):
            video_capture = cs.get_camera_stream(0)
            cs.close_stream(video_capture)
            self.assertFalse(video_capture.read()[0])

if __name__ == '__main__':
    unittest.main()