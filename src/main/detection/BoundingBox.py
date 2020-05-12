from configparser import ConfigParser
from src.main.utils.PathBuilder import PathBuilder
import cv2 as cv

# index of file in project
file_index = 4

class BoundingBox:

    net = None
    blob = None
    layer_names = None
    swap_rb = False
    crop = False
    mean = ''
    labels = ''
    d_depth = ''
    output_width = ''
    output_length = ''
    input_width = ''
    input_length = ''
    scale_factor = ''
    project_dir = ''

    # constructor initializes the video capture with the device_id
    def __init__(self, net, labels, scale_factor, output_length, output_width, mean,
                 swap_rb, crop, d_depth, yolov_labels, yolov_config, yolov_weights, file_index=4):

        self.output_width = int(output_width)
        self.output_length = int(output_length)
        self.scale_factor = float(scale_factor)
        self.swap_rb = bool(swap_rb)
        self.crop = bool(crop)

        try:
            self.mean = float(mean)
        except:
            self.mean = mean

        try:
            self.d_depth = float(d_depth)
        except:
            self.d_depth = d_depth

        self.project_dir = PathBuilder.get_relative_path(__file__, file_index)

        self.labels = open(self.project_dir + '/' + yolov_labels).read().strip().split('\n')
        self.net = cv.dnn.readNetFromDarknet(self.project_dir + '/' + yolov_config,
                                             self.project_dir + '/' + yolov_weights)
        self.layer_names = net.getLayerNames()
        self.layer_names = [self.layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def get_blob(self,frame):
        return cv.dnn.blobFromImage(frame,
                                    self.scale_factor,
                                    (self.length, self.width),
                                    swapRB=self.swap_rb,
                                    crop=self.crop,
                                    mean=self.mean,
                                    ddepth=self.d_depth)

    def get_output_layers(self, blob):
        return self.net.setInput(blob).forward(self.layer_names)