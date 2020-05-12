from configparser import ConfigParser
from src.main.utils.PathBuilder import PathBuilder
import cv2 as cv

# index of file in project
file_index = 4

class BoundingBox:

    # loaded weights and configs for yolov3
    net = None
    # output layers from yolov3
    layer_names = None
    # flag which indicates that swap first and last channels in 3-channel image is necessary
    swap_rb = False
    # flag which indicates whether image will be cropped after resize or not
    crop = False
    # scalar with mean values which are subtracted from channels
    mean = ''
    # labels for object classification
    labels = ''
    # width of the frame output
    output_width = ''
    # length of the frame output
    output_length = ''
    # width of the frame input
    input_width = ''
    # width of the frame length
    input_length = ''
    # multiplier for image values
    scale_factor = ''
    # project_base_dir
    project_dir = ''

    # constructor initializes the video capture with the device_id
    def __init__(self, scale_factor, output_length, output_width, mean,
                 swap_rb, crop, yolov_labels, yolov_config, yolov_weights, file_index=4):

        self.output_width = int(output_width)
        self.output_length = int(output_length)
        self.scale_factor = float(scale_factor)
        self.swap_rb = bool(swap_rb)
        self.crop = bool(crop)

        try:
            self.mean = float(mean)
        except:
            self.mean = None

        self.project_dir = PathBuilder.get_relative_path(__file__, file_index)

        self.labels = open(self.project_dir + '/' + yolov_labels).read().strip().split('\n')
        self.net = cv.dnn.readNetFromDarknet(self.project_dir + '/' + yolov_config,
                                             self.project_dir + '/' + yolov_weights)
        self.layer_names = self.net.getLayerNames()
        self.layer_names = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    # return the blob from the input frame
    def get_blob(self,frame):
        return cv.dnn.blobFromImage(frame,
                                    self.scale_factor,
                                    (self.output_length, self.output_width),
                                    swapRB=self.swap_rb,
                                    crop=self.crop,
                                    mean=self.mean)

    # get the output layers from the blob
    def get_output_layers(self, blob):
        self.net.setInput(blob)
        return self.net.forward(self.layer_names)