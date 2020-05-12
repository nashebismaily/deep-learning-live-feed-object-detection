import cv2 as cv
from src.main.detection.CaptureStream import CaptureStream
from src.main.detection.BoundingBox import BoundingBox
from src.main.utils import Enums
from configparser import ConfigParser
from src.main.utils.Parser import Parser

# main function
def main():
    parser = ConfigParser()
    d_program_mode = Parser.get_config(parser, 'detection.cfg', 'program_mode')
    d_bounding_box = Parser.get_config(parser, 'boundingbox.cfg', 'blob')
    image_mode = int(d_program_mode.get('image'))
    video_mode = int(d_program_mode.get('video'))
    camera_mode = int(d_program_mode.get('camera'))

    # process an image
    if image_mode == 1:
        print("TODO")
    # process a video file
    elif video_mode == 1:
        print("TODO")
    # process a video camera
    elif camera_mode == 1:
        d_camera = Parser.get_config(parser, 'detection.cfg', 'camera')
        device_id = d_camera.get('device_id')
        frame_refresh_interval_ms = d_camera.get('frame_refresh_interval_ms')

        capture_stream = CaptureStream(device_id)
        camera = capture_stream.get_camera_stream()

        if camera.isOpened():
            while True:
                ret_val, frame  = camera.read()

                #bounding_box = BoundingBox(frame,scale_factor,output_length,output_width,mean,swap_rb,crop,d_depth)
                #del bounding_box

                cv.imshow("device: " + str(device_id), frame)

                interrupt_key = cv.waitKey(int(frame_refresh_interval_ms))
                if interrupt_key == Enums.KeyInterrupt.ESCAPE.value:
                    break

            capture_stream.close_camera_stream()
            capture_stream.destory_all_windows()

if __name__ == "__main__":
    main()