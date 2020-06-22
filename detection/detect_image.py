import cv2 as cv
from configparser import ConfigParser
from utils.bounding_box import BoundingBox

# Detect objects in an image file
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
    input_image = config.get('image', 'input_image')
    output_image = config.get('image', 'output_image')

    bounding_box = BoundingBox(scale_factor, output_length, output_width, swap_rb, crop,
                               yolov_labels, yolov_config, yolov_weights, confidence, threshold)

    # Read frame
    frame = cv.imread(input_image)
    input_height, input_width = frame.shape[:2]
    blob = bounding_box.get_blob(frame)
    output_layers = bounding_box.get_output_layers(blob)

    # Get and draw bounding boxes
    boxes, confidences, class_identifiers = bounding_box.get_bounding_boxes(output_layers, input_height,input_width)
    suppression = bounding_box.suppress_weak_overlapping_boxes(boxes, confidences)
    frame_with_boxes = bounding_box.draw_boxes_labels(frame, boxes, confidences, class_identifiers, suppression)

    # Write resulting frame
    cv.imwrite(output_image,frame_with_boxes)
    print("Wrote image to: {}".format(output_image))

    # Cleanup
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
