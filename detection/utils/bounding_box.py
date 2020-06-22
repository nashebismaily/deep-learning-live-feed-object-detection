import cv2 as cv
import numpy as np

class BoundingBox:

    # loaded weights and configs for yolov3
    net = None
    # output layers from yolov3
    layer_names = None
    # flag which indicates that swap first and last channels in 3-channel image is necessary
    swap_rb = False
    # flag which indicates whether image will be cropped after resize or not
    crop = False
    # labels for object classification
    labels = ''
    # width of the frame output
    output_width = ''
    # length of the frame output
    output_length = ''
    # width of the frame input
    scale_factor = ''
    # confidence interval for bounding boxes
    confidence = ''
    # threshold interval for bounding boxes
    threshold = ''
    # colors for each label
    colors =''

    # constructor initializes the video capture with the device_id
    def __init__(self, scale_factor, output_length, output_width, swap_rb, crop, yolov_labels, yolov_config,
                 yolov_weights, confidence, threshold, file_index=4):

        self.output_width = int(output_width)
        self.output_length = int(output_length)
        self.scale_factor = float(scale_factor)
        if swap_rb == "True":
            self.swap_rb = True
        else:
            self.swap_rb = False
        if crop == "True":
            self.crop = True
        else:
            self.crop = False
        self.confidence = float(confidence)
        self.threshold = float(threshold)
        self.labels = open('../model/yolov/' + yolov_labels).read().strip().split('\n')
        self.net = cv.dnn.readNetFromDarknet('../model/yolov/' + yolov_config, '../model/weights/' + yolov_weights)
        self.layer_names = self.net.getLayerNames()
        self.layer_names = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.randint(0, 255, size=(len(self.labels*2), 3), dtype='uint8')
    
    # return the blob from the input frame
    def get_blob(self,frame):
        return cv.dnn.blobFromImage(frame,
                                    self.scale_factor,
                                    (self.output_length, self.output_width),
                                    swapRB=self.swap_rb,
                                    crop=self.crop)

    # get the output layers from the blob
    def get_output_layers(self, blob):
        self.net.setInput(blob)
        return self.net.forward(self.layer_names)

    def get_bounding_boxes(self, output_layers, input_height, input_width):
        boxes = []
        confidences = []
        class_identifiers = []
        for output in output_layers:
            for detection in output:
                scores = detection[5:]
                class_identifier = np.argmax(scores)
                confidence = scores[class_identifier]
                if confidence > self.confidence and class_identifier <= len(self.labels):
                    box = detection[0:4] * np.array([input_width, input_height, input_width, input_height])
                    (centerX, centerY, box_width, box_height) = box.astype("int")
                    x = int(centerX - (box_width / 2))
                    y = int(centerY - (box_height / 2))
                    boxes.append([x, y, int(box_width), int(box_height)])
                    confidences.append(float(confidence))
                    class_identifiers.append(class_identifier)
        return boxes, confidences, class_identifiers

    # suppress weak and overlapping boxes
    def suppress_weak_overlapping_boxes(self, boxes, confidences):
        return cv.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

    # draw the bounding box for the object and return the frame
    def draw_boxes_labels(self, frame, boxes, confidences, class_identifiers, suppression):
        if len(suppression) > 0:
            for i in suppression.flatten():
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]
                color = [int(c) for c in self.colors[class_identifiers[i]]]
                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:4f}".format(self.labels[class_identifiers[i]], confidences[i])
                cv.putText(frame, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    # convert string to boolean
    def string_to_bool(string):
        if string == 'True':
            return True
        else:
            return False


