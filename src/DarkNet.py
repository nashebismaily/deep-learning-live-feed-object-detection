import subprocess
import sys

# DarkNet is a utility class for calling the DarkNet Detector function.
# The DarkNet home directory, object data file, yolov3 config file, and yolov3 weights file is expected.
class DarkNet:

    # parameterized constructor
    def __init__(self, darknet_base_dir='/root/darknet',
                 darknet_object_data='object.data',
                 yolov3_config_file='yolov3.cfg',
                 yolov3_weights_file='darknet53.conv.74'):
        self.darknet_base_dir = darknet_base_dir
        self.darknet_object_data = darknet_object_data
        self.yolov3_config_file = yolov3_config_file
        self.yolov3_weights_file = yolov3_weights_file

    # train calls the DarkNet detector function with the train argument
    def train(self):
        linux_cmd = str(self.darknet_base_dir) \
                    + "/darknet detector train " \
                    + str(self.yolov3_config_file) \
                    + str(self.yolov3_weights_file)

        # Run DarkNet training as a linux subprocess
        p = subprocess.Popen(linux_cmd, shell=True, stderr=subprocess.PIPE)
        while True:
            out = p.stderr.read(1)
            if out == '' and p.poll() != None:
                break
            if out != '':
                sys.stdout.write(out)
                sys.stdout.flush()