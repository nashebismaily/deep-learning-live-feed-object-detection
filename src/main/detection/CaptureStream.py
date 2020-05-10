import cv2 as cv

# CaptureStream is a utility class for calling the OpenCV2 video capture library
class CaptureStream:

    video_capture = None

    # default constructor initializes the video capture with the device_id
    def __init__(self,device_id):
        self.video_capture = cv.VideoCapture(int(device_id))

    # scan_devices returns the available cameras connected to the device
    def scan_devices(self):
        device_list = []
        id = 0
        while True:
            vc = cv.VideoCapture(id)
            if vc.read()[0]:
                self.device_list.append(id)
                id += 1
                cv.destroyAllWindows()
            else:
                break
            vc.release()
        cv.destroyAllWindows()

        return device_list

    # get_camera_stream opens the camera specified by the device_id and returns the video stream.
    def set_camera_stream(self,device_id):
        self.video_capture = cv.VideoCapture(device_id)

    def get_camera_stream(self):
        return self.video_capture

    # close_stream stops the video stream from the camera
    def close_camera_stream(self):
        self.video_capture.release()

    # destory_all_windows closes all frame capture windows
    def destory_all_windows(self):
        cv.destroyAllWindows()