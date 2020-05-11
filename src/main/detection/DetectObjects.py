import cv2 as cv
from src.main.detection.CaptureStream import CaptureStream
from src.main.utils import Enums as e
from configparser import ConfigParser
from src.main.utils.Utils import Utils

# main function
def main():
    parser = ConfigParser()
    parser.read(parser.read(Utils.get_relative_path(__file__, 'config/detection.cfg', 4)))

    device_id = parser.get('video_capture_config', 'device_id')
    frame_refresh_interval_ms = parser.get('video_capture_config', 'frame_refresh_interval_ms')

    capture_stream = CaptureStream(device_id)
    web_camera = capture_stream.get_camera_stream()

    if web_camera.isOpened():
        while True:
            ret_val, frame  = web_camera.read()
            cv.imshow("device: " + str(device_id), frame)

            interrupt_key = cv.waitKey(int(frame_refresh_interval_ms))
            if interrupt_key == e.KeyInterrupt.ESCAPE.value:
                break

        capture_stream.close_camera_stream()
        capture_stream.destory_all_windows()

if __name__ == "__main__":
    main()