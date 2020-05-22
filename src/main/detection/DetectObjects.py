import cv2 as cv
import imutils
import time
from src.main.detection.CaptureStream import CaptureStream
from src.main.detection.BoundingBox import BoundingBox
from src.main.utils import Keys
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

    # process an image
    if image_mode == 1:
        d_image = Parser.get_config(parser, 'detection.cfg', 'image')
        input_image = d_image.get('input_image')
        output_image = d_image.get('output_image')

        frame = cv.imread(input_image)
        input_height, input_width = frame.shape[:2]
        blob = bounding_box.get_blob(frame)
        output_layers = bounding_box.get_output_layers(blob)
        boxes, confidences, class_identifiers = bounding_box.get_bounding_boxes(output_layers, input_height,
                                                                                input_width)
        suppression = bounding_box.suppress_weak_overlapping_boxes(boxes, confidences)
        frame_with_boxes = bounding_box.draw_boxes_labels(frame, boxes, confidences, class_identifiers, suppression)

        cv.imwrite(output_image,frame_with_boxes)
        print("Wrote image to: {}".format(output_image))
        cv.destroyAllWindows()

    # process a video file
    if video_mode == 1:
        d_video = Parser.get_config(parser, 'detection.cfg', 'video')
        input_video = d_video.get('input_video')
        output_video = d_video.get('output_video')
        frames_per_second = d_video.get('frames_per_second')
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

    # process a video camera
    if camera_mode == 1:
        d_camera = Parser.get_config(parser, 'detection.cfg', 'camera')
        device_id = d_camera.get('device_id')
        frame_wait_ms = d_camera.get('frame_wait_ms')
        frame_pause_interval = d_camera.get('frame_pause_interval')

        capture_stream = CaptureStream(device_id)
        camera = capture_stream.get_camera_stream()

        frame_count = 1
        if camera.isOpened():
            while True:
                ret_val, frame  = camera.read()

                if  frame_count == 1 or frame_count % int(frame_pause_interval) == 0:
                    input_height, input_width = frame.shape[:2]
                    blob = bounding_box.get_blob(frame)
                    output_layers = bounding_box.get_output_layers(blob)
                    boxes, confidences, class_identifiers = bounding_box.get_bounding_boxes(output_layers, input_height,
                                                                                            input_width)
                    suppression = bounding_box.suppress_weak_overlapping_boxes(boxes, confidences)
                    frame_with_boxes = bounding_box.draw_boxes_labels(frame, boxes, confidences, class_identifiers,
                                                                      suppression)
                else:
                    frame_with_boxes = bounding_box.draw_boxes_labels(frame, boxes, confidences, class_identifiers, suppression)

                frame_count += 1
                cv.imshow("device: " + str(device_id), frame_with_boxes)

                interrupt_key = cv.waitKey(int(frame_wait_ms))
                if interrupt_key == Keys.KeyInterrupt.ESCAPE.value:
                    break

            capture_stream.close_camera_stream()
            capture_stream.destory_all_windows()

if __name__ == "__main__":
    main()