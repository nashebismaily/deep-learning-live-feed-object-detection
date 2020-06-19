import cv2 as cv
from src.main.detection.BoundingBox import BoundingBox
from configparser import ConfigParser
from src.main.utils.Parser import Parser

# Detect objects in an image file
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

    # Read detection.cfg
    d_image = Parser.get_config(parser, 'detection.cfg', 'image')
    input_image = d_image.get('input_image')
    output_image = d_image.get('output_image')

    bounding_box = BoundingBox(scale_factor, output_length, output_width, mean, swap_rb, crop,
                               yolov_labels, yolov_config, yolov_weights, confidence, threshold)

    # Read frame
    frame = cv.imread(input_image)
    input_height, input_width = frame.shape[:2]
    blob = bounding_box.get_blob(frame)
    output_layers = bounding_box.get_output_layers(blob)

    # Get and draw bounding boxes
    boxes, confidences, class_identifiers = bounding_box.get_bounding_boxes(output_layers, input_height,
                                                                            input_width)
    suppression = bounding_box.suppress_weak_overlapping_boxes(boxes, confidences)
    frame_with_boxes = bounding_box.draw_boxes_labels(frame, boxes, confidences, class_identifiers, suppression)

    # Write resulting frame
    cv.imwrite(output_image,frame_with_boxes)
    print("Wrote image to: {}".format(output_image))

    # Cleanup
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()