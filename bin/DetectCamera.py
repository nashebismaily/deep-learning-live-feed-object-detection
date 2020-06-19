import cv2 as cv
from src.main.detection.CaptureStream import CaptureStream
from src.main.detection.BoundingBox import BoundingBox
from src.main.utils import Keys
from configparser import ConfigParser
from src.main.utils.Parser import Parser

# Detect objects from a live camera feed
def main():
    parser = ConfigParser()

    # Read boundingbox.cfg
    d_bounding_box = Parser.get_config(parser, 'boundingbox.cfg', 'blob')
    mean = d_bounding_box.get('mean')
    crop = d_bounding_box.get('crop')
    swap_rb = d_bounding_box.get('swap_rb')
    scale_factor = d_bounding_box.get('scale_factor')
    output_width = d_bounding_box.get('output_width')
    output_length = d_bounding_box.get('output_length')
    confidence = d_bounding_box.get('confidence')
    threshold = d_bounding_box.get('threshold')

    # Read darknet.cfg
    d_darknet = Parser.get_config(parser, 'darknet.cfg', 'yolov3')
    yolov_labels = d_darknet.get('yolov3_labels')
    yolov_weights = d_darknet.get('yolov3_test_weights_file')
    yolov_config = d_darknet.get('yolov3_config_file')

    # Read detection.cfg'
    d_camera = Parser.get_config(parser, 'detection.cfg', 'camera')
    device_id = d_camera.get('device_id')
    frame_wait_ms = d_camera.get('frame_wait_ms')
    frame_pause_interval = d_camera.get('frame_pause_interval')

    bounding_box = BoundingBox(scale_factor, output_length, output_width, mean, swap_rb, crop,
                               yolov_labels, yolov_config, yolov_weights, confidence, threshold)

    capture_stream = CaptureStream(device_id)
    camera = capture_stream.get_camera_stream()

    frame_count = 1
    if camera.isOpened():
        while True:
            # Read each frame
            ret_val, frame  = camera.read()

            # Draw new bounding box
            if  frame_count == 1 or frame_count % int(frame_pause_interval) == 0:
                input_height, input_width = frame.shape[:2]
                blob = bounding_box.get_blob(frame)
                output_layers = bounding_box.get_output_layers(blob)
                boxes, confidences, class_identifiers = bounding_box.get_bounding_boxes(output_layers, input_height,
                                                                                        input_width)
                suppression = bounding_box.suppress_weak_overlapping_boxes(boxes, confidences)
                frame_with_boxes = bounding_box.draw_boxes_labels(frame, boxes, confidences, class_identifiers,
                                                                  suppression)
            # Use existing bounding box
            else:
                frame_with_boxes = bounding_box.draw_boxes_labels(frame, boxes, confidences, class_identifiers, suppression)

            frame_count += 1
            cv.imshow("device: " + str(device_id), frame_with_boxes)

            # End camera capture
            interrupt_key = cv.waitKey(int(frame_wait_ms))
            if interrupt_key == Keys.KeyInterrupt.ESCAPE.value:
                break

        # Cleanup
        capture_stream.close_camera_stream()
        capture_stream.destory_all_windows()

if __name__ == "__main__":
    main()