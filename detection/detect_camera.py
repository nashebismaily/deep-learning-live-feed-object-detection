import cv2 as cv
from configparser import ConfigParser
from utils.capture_stream import CaptureStream
from utils.bounding_box import BoundingBox

# Detect objects from a live camera feed
def main():
    # create parser instance
    config = ConfigParser()

    # Read detection.cfg
    config.read('config/detection.cfg')
    crop = config.get('blob', 'crop')
    swap_rb = config.get('blob', 'swap_rb')
    scale_factor = config.get('blob', 'scale_factor')
    output_width = config.get('blob', 'output_width')
    output_length = config.get('blob', 'output_length')
    confidence = config.get('blob', 'confidence')
    threshold = config.get('blob', 'threshold')
    yolov_labels = config.get('yolov3', 'yolov3_labels')
    yolov_weights = config.get('yolov3', 'yolov3_test_weights_file')
    yolov_config = config.get('yolov3', 'yolov3_config_file')
    device_id = config.get('camera', 'device_id')
    frame_wait_ms = config.get('camera', 'frame_wait_ms')
    frame_pause_interval = config.get('camera', 'frame_pause_interval')

    bounding_box = BoundingBox(scale_factor, output_length, output_width, swap_rb, crop,
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
                boxes, confidences, class_identifiers = bounding_box.get_bounding_boxes(output_layers, input_height, input_width)
                suppression = bounding_box.suppress_weak_overlapping_boxes(boxes, confidences)
                frame_with_boxes = bounding_box.draw_boxes_labels(frame, boxes, confidences, class_identifiers, suppression)
            # Use existing bounding box
            else:
                frame_with_boxes = bounding_box.draw_boxes_labels(frame, boxes, confidences, class_identifiers, suppression)

            frame_count += 1
            cv.imshow("device: " + str(device_id), frame_with_boxes)

            # End camera capture with escape key
            interrupt_key = cv.waitKey(int(frame_wait_ms))
            if interrupt_key == 27:
                break

        # Cleanup
        capture_stream.close_camera_stream()
        capture_stream.destory_all_windows()

if __name__ == "__main__":
    main()
