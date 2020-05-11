import subprocess
from subprocess import PIPE, STDOUT
import matplotlib.pyplot as plt
import os

# DarkNet is a utility class for calling the DarkNet Detector function.
# The DarkNet home directory, object data file, yolov3 config file, and yolov3 weights file is expected.
class DarkNet:

    # Base directory where DarkNet is installed
    darknet_base_dir =''
    # object.data file
    darknet_object_data =''
    # Configuration file used for Yolov3 Training
    yolov3_config_file =''
    # Default weights file used for Training
    yolov3_weights_file =''
    # Log file for training output
    log_file =''

    # constructor
    def __init__(self, darknet_base_dir, darknet_object_data, yolov3_config_file, yolov3_weights_file, log_file):
        self.darknet_base_dir = darknet_base_dir
        self.darknet_object_data = darknet_object_data
        self.yolov3_config_file = yolov3_config_file
        self.yolov3_weights_file = yolov3_weights_file
        self.log_file = log_file

    # calls the DarkNet detector function with the train argument
    def train(self):

        linux_cmd = str(self.darknet_base_dir) \
                    + "/darknet detector train " \
                    + str(self.darknet_object_data) + " "\
                    + str(self.yolov3_config_file) + " "\
                    + str(self.yolov3_weights_file)+  " "\
                    + " > " + self.log_file

        process = subprocess.Popen(linux_cmd , shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
        return process

    # calls a subprocess to tail the output log. Once training is complete, the processes are killed
    def tail_log(self, process):
        linux_cmd = "tail -f " + self.darknet_base_dir + "/" + self.log_file
        tail = subprocess.Popen(linux_cmd , shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
        while True:
            poll = process.poll()
            if poll == None:
                line = tail.stdout.readline()
                print(line)
            else:
                process.kill()
                tail.kill()
                break

    # plots the iterations and average loss from the DarkNet training
    def plot_loss(self):
        matches = ["avg", "rate", "seconds", "images"]

        lines = []
        for line in open(self.darknet_base_dir + "/" +self.log_file):
            for match in matches:
                if match in line:
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
        plt.savefig(os.path.join(os.path.dirname(__file__), '../../../model/graphs', 'average_loss.png'))
        plt.show()