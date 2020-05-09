import cv2 as cv
from src.main.detection import CaptureStream as cs
from src.main.utils import Enums as e


# main function
def main():
    # These will become input arguments
    device_id = 0
    frame_refresh_interval_ms = 1

    device_list = cs.get_device_list()
    video_capture = cs.get_camera_stream(device_id)

    if video_capture.isOpened():
        while True:
            ret_val, frame  = video_capture.read()
            cv.imshow("device: " + str(device_id), frame)

            interrupt_key = cv.waitKey(frame_refresh_interval_ms)
            if interrupt_key == e.KeyInterrupt.ESCAPE.value:
                break

        cs.close_stream()

if __name__ == "__main__":
    main()