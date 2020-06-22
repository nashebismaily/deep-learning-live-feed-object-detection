import subprocess
from subprocess import STDOUT
from subprocess import PIPE
import matplotlib.pyplot as plt
import os

# DarkNet is a utility class for calling the DarkNet Detector function.
# The DarkNet home directory, object data file, yolov3 config file, and yolov3 weights file is expected.
class DarkNet:

    # Base directory where DarkNet is installed
    darknet_base_dir = ''
    # object.data file
    darknet_object_data = ''
    # Configuration file used for Yolov3 Training
    yolov3_config_file = ''
    # Default weights file used for Training
    yolov3_train_weights_file = ''
    # Log file for training output
    log_file = ''
    # Graph file for average loss
    graph_file = ''
    # project_base_dir
    project_dir = ''
    # Runtime for program
    shell = True

    # constructor
    def __init__(self, darknet_base_dir, darknet_object_data, yolov3_config_file, yolov3_train_weights_file,
                       log_file, graph_file, shell):
        self.darknet_base_dir = darknet_base_dir
        self.darknet_object_data = darknet_object_data
        self.yolov3_config_file = yolov3_config_file
        self.yolov3_train_weights_file = yolov3_train_weights_file
        self.log_file = log_file
        self.graph_file = graph_file
        self.shell = shell

       # build absolute path for darknet.py
        parent_dir = __file__
        file = None
        index = 3
        while index > 0:
            parent_dir = os.path.dirname(parent_dir)
            index -= 1
        if file:
            parent_dir = parent_dir + '/' + file
        self.project_dir = parent_dir

        print(self.project_dir)

    # calls the DarkNet detector function with the train argument
    def train(self):

        # Shell mode
        if self.shell:
            linux_cmd = str(self.darknet_base_dir) \
                        + "/darknet detector train " \
                        + 'yolov/' + str(self.darknet_object_data) + " "\
                        + 'yolov/' + str(self.yolov3_config_file) + " "\
                        + 'weights/' + str(self.yolov3_train_weights_file)+  " "\
                        + " > " + 'logs/' + self.log_file
        # IDE mode
        else:
            linux_cmd = str(self.darknet_base_dir) \
                        + "/darknet detector train " \
                        + self.project_dir + '/model/yolov/' + str(self.darknet_object_data) + " " \
                        + self.project_dir + '/model/yolov/' + str(self.yolov3_config_file) + " " \
                        + self.project_dir + '/model/weights/' + str(self.yolov3_train_weights_file) + " " \
                        + " > " + self.project_dir + '/' + self.log_file

        print("Running Shell Command: {}".format(linux_cmd))
        process = subprocess.Popen(linux_cmd , shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
        return process

    # wait for a process to end
    def wait(self,process):
        print("Dark Net Log file: " + str('logs/' + self.log_file))
        process.wait()
        print("done waiting")

    # plots the iterations and average loss from the DarkNet training
    def plot_loss(self):
        lines = []

        # Shell mode
        if self.shell:
            log_file = 'logs/' + self.log_file
        # IDE mode
        else:
            log_file = self.project_dir + '/' + self.log_file

        for line in open(log_file):
            print(line)
            if 'avg' in line:
                lines.append(line)

        iterations = []
        loss = []
        for i in range(len(lines)):
            line_parsed = lines[i].split(',')
            iterations.append(int(line_parsed[0].split(':')[0]))
            loss.append(float(line_parsed[1].split()[0]))

        for i in range(0, len(lines)):
            plt.plot(iterations[i:i + 2], loss[i:i + 2], 'r.-')

        plt.xlabel('Iteration Number')
        plt.ylabel('Average Loss')
        plt.savefig(str('graphs/' + self.graph_file))
        plt.show()