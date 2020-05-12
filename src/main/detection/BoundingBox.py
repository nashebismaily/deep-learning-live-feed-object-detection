from configparser import ConfigParser
from src.main.utils.PathBuilder import PathBuilder
import cv2 as cv

# index of file in project
file_index = 4

class BoundingBox:

    image = None
    blob = None
    output_width = ''
    output_length = ''
    input_width = ''
    input_length = ''
    mean = ''
    scale_factor = ''
    swap_rb = False
    crop = False
    d_depth = ''

    # constructor initializes the video capture with the device_id
    def __init__(self, image, scale_factor, output_length, output_width, mean, swap_rb, crop, d_depth):
        self.image = image
        self.output_width = output_width
        self.output_length = output_length
        self.mean = mean
        self.scale_factor = scale_factor
        self.swap_rb = swap_rb
        self.crop = crop
        self.d_depth = d_depth

        blob = cv.dnn.blobFromImage(self.image,
                                    self.scale_factor,
                                    (self.length, self.width),
                                    mean=self.mean,
                                    swapRB=self.swap_rb,
                                    crop=self.crop,
                                    ddepth=self.d_depth)

        #net.setInput(blob)
        #layerOutputs = net.forward(ln)
