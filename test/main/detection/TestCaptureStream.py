import unittest
from unittest import mock
import cv2 as cv
from src.main.detection.CaptureStream import CaptureStream

# Unit Tests for CaptureStream.py
class TestCaptureStream(unittest.TestCase):

        def test_scan_devices(self):
            capture_stream = CaptureStream()
            self.assertTrue(capture_stream.scan_devices())

        def test_get_camera_stream(self):
            device_id = 0
            capture_stream = CaptureStream(device_id)
            self.assertIsInstance(capture_stream.get_camera_stream(), cv.VideoCapture)
            capture_stream.close_camera_stream()

        def test_close_stream(self):
            device_id = 0
            capture_stream = CaptureStream(device_id)
            webcam = capture_stream.get_camera_stream()
            capture_stream.close_camera_stream()
            self.assertFalse(webcam.read()[0])

if __name__ == '__main__':
    unittest.main()