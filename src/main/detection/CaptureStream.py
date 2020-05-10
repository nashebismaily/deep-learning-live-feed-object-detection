import cv2 as cv

# CaptureStream extrapolates the OpenCV2 video capture library by providing helper functions
class CaptureStream:

    # VideoCapture object
    video_capture = None

    #  constructor initializes the video capture with the device_id
    def __init__(self,device_id=None):
        if device_id != None:
            self.video_capture = cv.VideoCapture(int(device_id))

    # scans available devices and returns the available cameras connected to the device
    def scan_devices(self):
        device_list = []
        id = 0
        while True:
            vc = cv.VideoCapture(id)
            if vc.read()[0]:
                device_list.append(id)
                id += 1
                cv.destroyAllWindows()
            else:
                break
            vc.release()
        cv.destroyAllWindows()

        return device_list

    # opens the camera specified by the device_id
    def set_camera_stream(self,device_id):
        self.video_capture = cv.VideoCapture(device_id)

    # returns the camera stream
    def get_camera_stream(self):
        return self.video_capture

    # stops the video stream from the camera
    def close_camera_stream(self):
        self.video_capture.release()

    # closes all frame capture windows
    def destory_all_windows(self):
        cv.destroyAllWindows()