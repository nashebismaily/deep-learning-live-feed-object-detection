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
    image_mode = int(d_program_mode.get('image'))
    video_mode = int(d_program_mode.get('video'))
    camera_mode = int(d_program_mode.get('camera'))

    d_bounding_box = Parser.get_config(parser, 'boundingbox.cfg', 'blob')
    mean = d_bounding_box.get('mean')
    crop = d_bounding_box.get('crop')
    d_depth = d_bounding_box.get('d_depth')
    swap_rb = d_bounding_box.get('swap_rb')
    scale_factor = d_bounding_box.get('scale_factor')
    output_width = d_bounding_box.get('output_width')
    output_length = d_bounding_box.get('output_length')

    d_darknet = Parser.get_config(parser, 'darknet.cfg', 'darknet')
    yolov_labels = d_darknet.get('yolov3_labels')
    yolov_weights = d_darknet.get('yolov3_weights_file')
    yolov_config = d_darknet.get('yolov3_config_file')

    bounding_box = BoundingBox(scale_factor, output_length, output_width, mean, swap_rb,
                               crop, d_depth, yolov_labels, yolov_config, yolov_weights)

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

                blob = bounding_box.get_blob(frame)
                output_layers = bounding_box.get_output_layers(blob)

                cv.imshow("device: " + str(device_id), frame)

                interrupt_key = cv.waitKey(int(frame_refresh_interval_ms))
                if interrupt_key == Enums.KeyInterrupt.ESCAPE.value:
                    break

            capture_stream.close_camera_stream()
            capture_stream.destory_all_windows()

if __name__ == "__main__":
    main()