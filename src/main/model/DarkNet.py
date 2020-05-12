import subprocess
from subprocess import STDOUT
import matplotlib.pyplot as plt
from src.main.utils.PathBuilder import PathBuilder
from subprocess import PIPE

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
    yolov3_weights_file = ''
    # Log file for training output
    log_file = ''
    # Graph file for average loss
    graph_file = ''
    # project_base_dir
    project_dir = ''

    # constructor
    def __init__(self, darknet_base_dir, darknet_object_data, yolov3_config_file, yolov3_weights_file, log_file, graph_file, file_index=4):
        self.darknet_base_dir = darknet_base_dir
        self.darknet_object_data = darknet_object_data
        self.yolov3_config_file = yolov3_config_file
        self.yolov3_weights_file = yolov3_weights_file
        self.log_file = log_file
        self.graph_file = graph_file
        self.file_index = int(file_index)
        self.project_dir = PathBuilder.get_relative_path(__file__, file_index)

    # calls the DarkNet detector function with the train argument
    def train(self):

        linux_cmd = str(self.darknet_base_dir) \
                    + "/darknet detector train " \
                    + self.project_dir + '/' + str(self.darknet_object_data) + " "\
                    + self.project_dir + '/' + str(self.yolov3_config_file) + " "\
                    + self.project_dir + '/' + str(self.yolov3_weights_file)+  " "\
                    + " > " + self.project_dir + '/' + self.log_file
        process = subprocess.Popen(linux_cmd , shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
        return process

    # wait for a process to end
    def wait(self,process):
        print("Log file: " + str(self.project_dir + '/' + self.log_file))
        process.wait()

    # plots the iterations and average loss from the DarkNet training
    def plot_loss(self):
        lines = []
        for line in open(self.project_dir + '/' + self.log_file):
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
        plt.savefig(str(self.project_dir + '/' + self.graph_file))
        plt.show()