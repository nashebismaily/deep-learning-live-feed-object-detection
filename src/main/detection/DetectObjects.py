import cv2 as cv
from src.main.detection.CaptureStream import CaptureStream
from src.main.utils import Enums as e
from configparser import ConfigParser
import os

# main function
def main():
    parser = ConfigParser()
    parser.read(os.path.join(os.path.dirname(__file__), '../../../config', 'detection.cfg'))
    device_id = parser.get('video_capture_config', 'device_id')
    frame_refresh_interval_ms = parser.get('video_capture_config', 'frame_refresh_interval_ms')

    capture_stream = CaptureStream(device_id)
    webcam = capture_stream.get_camera_stream()

    if webcam.isOpened():
        while True:
            ret_val, frame  = webcam.read()
            cv.imshow("device: " + str(device_id), frame)

            interrupt_key = cv.waitKey(int(frame_refresh_interval_ms))
            if interrupt_key == e.KeyInterrupt.ESCAPE.value:
                break

        capture_stream.close_camera_stream()
        capture_stream.destory_all_windows()

if __name__ == "__main__":
    main()