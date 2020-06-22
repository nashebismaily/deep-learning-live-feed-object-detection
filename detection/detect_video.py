import cv2 as cv
import imutils
import time
from configparser import ConfigParser
from utils.bounding_box import BoundingBox

# Detect objects in a video file and write back out as .avi video
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
    input_video = config.get('video', 'input_video')
    output_video = config.get('video', 'output_video')
    frames_per_second = int(config.get('video', 'frames_per_second'))

    # convert string to boolean
    if config.get('video', 'color') == 'True':
        color=True
    else:
        color=False

    bounding_box = BoundingBox(scale_factor, output_length, output_width, swap_rb, crop,
                               yolov_labels, yolov_config, yolov_weights, confidence, threshold)

    video_stream = cv.VideoCapture(input_video)
    writer = None

    # Obtain video frame metadata
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
        # Read frame
        ret_val, frame = video_stream.read()
        if not ret_val:
            break
        start_time = time.time()
        input_height, input_width = frame.shape[:2]

        # Get and draw bounding boxes
        blob = bounding_box.get_blob(frame)
        output_layers = bounding_box.get_output_layers(blob)
        boxes, confidences, class_identifiers = bounding_box.get_bounding_boxes(output_layers, input_height, input_width)
        suppression = bounding_box.suppress_weak_overlapping_boxes(boxes, confidences)
        frame_with_boxes = bounding_box.draw_boxes_labels(frame, boxes, confidences, class_identifiers, suppression)
        end_time = time.time()

        # Write frame
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

    # Cleanup
    video_stream.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
