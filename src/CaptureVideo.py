import numpy as np
import cv2 as cv
from enum import Enum

# get_device_list() enumerates the available cameras connected to the device and
# returns a list of device id's which can be used with the cv2 VideoCapture construct.
def get_device_list():
    device_id = 0
    device_list = []
    while True:
        if cv.VideoCapture(device_id).read()[0]:
            device_list.append(device_id)
            device_id += 1
        else:
            break

    return device_list

# get_camera_stream opens the camera specified by the device_id and returns the video stream.
def get_camera_stream(device_id):
    return cv.VideoCapture(device_id)

# KEYBOARD_INTERRUPT defines the decimal representation of keyboard keys
class KeyboardInterrupt(Enum):
    ESCAPE = 27
    SPACE = 32

def main():
    # These will become input arguments
    device_id = 0
    frame_refresh_interval_ms = 1

    device_list = get_device_list()

    video_capture = get_camera_stream(device_id)

    if video_capture.isOpened():
        while True:
            ret_val, frame  = video_capture.read()
            cv.imshow("device: " + str(device_id), frame)

            interrupt_key = cv.waitKey(frame_refresh_interval_ms)

            if interrupt_key == KeyboardInterrupt.ESCAPE.value:
                break

        video_capture.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()