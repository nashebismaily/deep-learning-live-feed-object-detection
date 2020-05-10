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

    cs = CaptureStream(device_id)
    video_capture = cs.get_camera_stream()

    if video_capture.isOpened():
        while True:
            ret_val, frame  = video_capture.read()
            cv.imshow("device: " + str(device_id), frame)

            interrupt_key = cv.waitKey(int(frame_refresh_interval_ms))
            if interrupt_key == e.KeyInterrupt.ESCAPE.value:
                break

        cs.close_camera_stream()
        cs.destory_all_windows()

if __name__ == "__main__":
    main()