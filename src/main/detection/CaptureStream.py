import cv2 as cv

# CaptureStream is a utility class for calling the OpenCV2 video capture library
class CaptureStream:

    # get_device_list() enumerates the available cameras connected to the device and
    # returns a list of device id's which can be used with the cv2 VideoCapture construct.
    def get_device_list(self):
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
    def get_camera_stream(self,device_id):
        return cv.VideoCapture(device_id)

    # close_stream stops the video stream from the camera and closes all frame capture windows
    def close_stream(self,video_capture):
        video_capture.release()
        cv.destroyAllWindows()
