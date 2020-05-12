import cv2 as cv
from src.main.detection.CaptureStream import CaptureStream
from src.main.utils import Enums
from configparser import ConfigParser
from src.main.utils.Utils import Utils

# index of file in project
file_index = 4

# main function
def main():
    parser = ConfigParser()
    parser.read(parser.read(Utils.get_relative_path(__file__, file_index, 'config/detection.cfg')))
    image_mode =  parser.get('program_mode', 'image')
    video_mode =  parser.get('program_mode', 'video')
    camera_mode = parser.get('program_mode', 'camera')

    # process an image
    if image_mode == 1:
        print("TODO")
    # process a video file
    elif video_mode == 1:
        print("TODO")
    # process a video camera
    elif camera_mode == str(1):
        device_id = parser.get('camera', 'device_id')
        frame_refresh_interval_ms = parser.get('camera', 'frame_refresh_interval_ms')

        capture_stream = CaptureStream(device_id)
        camera = capture_stream.get_camera_stream()

        if camera.isOpened():
            while True:
                ret_val, frame  = camera.read()
                cv.imshow("device: " + str(device_id), frame)

                interrupt_key = cv.waitKey(int(frame_refresh_interval_ms))
                if interrupt_key == Enums.KeyInterrupt.ESCAPE.value:
                    break

            capture_stream.close_camera_stream()
            capture_stream.destory_all_windows()

if __name__ == "__main__":
    main()