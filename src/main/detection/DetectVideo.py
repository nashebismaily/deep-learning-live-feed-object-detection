import cv2 as cv
import imutils
import time
from src.main.detection.BoundingBox import BoundingBox
from configparser import ConfigParser
from src.main.utils.Parser import Parser

# Detect objects in a video file and write back out as .avi video
def main():
    parser = ConfigParser()
    d_bounding_box = Parser.get_config(parser, 'boundingbox.cfg', 'blob')
    mean = d_bounding_box.get('mean')
    crop = d_bounding_box.get('crop')

    swap_rb = d_bounding_box.get('swap_rb')
    scale_factor = d_bounding_box.get('scale_factor')
    output_width = d_bounding_box.get('output_width')
    output_length = d_bounding_box.get('output_length')
    confidence = d_bounding_box.get('confidence')
    threshold = d_bounding_box.get('threshold')

    d_darknet = Parser.get_config(parser, 'darknet.cfg', 'yolov3')
    yolov_labels = d_darknet.get('yolov3_labels')
    yolov_weights = d_darknet.get('yolov3_test_weights_file')
    yolov_config = d_darknet.get('yolov3_config_file')

    bounding_box = BoundingBox(scale_factor, output_length, output_width, mean, swap_rb, crop,
                               yolov_labels, yolov_config, yolov_weights, confidence, threshold)

    d_video = Parser.get_config(parser, 'detection.cfg', 'video')
    input_video = d_video.get('input_video')
    output_video = d_video.get('output_video')
    frames_per_second = int(d_video.get('frames_per_second'))
    color = Parser.string_to_bool(d_video.get('color'))

    video_stream = cv.VideoCapture(input_video)
    writer = None

    try:
        if imutils.is_cv2():
            prop = cv.cv.CV_CAP_PROP_FRAME_COUNT
        else :
            prop = cv.CAP_PROP_FRAME_COUNT
        total = int(video_stream.get(prop))
        print("Total frames in video: {}".format(total))
    except:
        print("Could not determine # of frames in video")
        total = -1

    frame_count = 1
    while True:
        ret_val, frame = video_stream.read()
        if not ret_val:
            break
        start_time = time.time()
        input_height, input_width = frame.shape[:2]
        blob = bounding_box.get_blob(frame)
        output_layers = bounding_box.get_output_layers(blob)
        boxes, confidences, class_identifiers = bounding_box.get_bounding_boxes(output_layers, input_height,
                                                                                input_width)
        suppression = bounding_box.suppress_weak_overlapping_boxes(boxes, confidences)
        frame_with_boxes = bounding_box.draw_boxes_labels(frame, boxes, confidences, class_identifiers, suppression)
        end_time = time.time()

        if writer is None:
            four_cc = cv.VideoWriter_fourcc(*"MJPG")
            writer = cv.VideoWriter(output_video, four_cc, frames_per_second, (frame.shape[1], frame.shape[0]), color)
            if total > 0:
                elapsed_time = (end_time - start_time)
                print("Single frame took:  {:.4f} seconds".format(elapsed_time))
                print("Estimated total time to finish: {:.4f} seconds".format(elapsed_time * total))

        if frame_count % 10 == 0:
            print("Total frames written: {}".format(frame_count))

        frame_count += 1
        writer.write(frame_with_boxes)

    print("Wrote video to: {}".format(output_video))

    video_stream.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()