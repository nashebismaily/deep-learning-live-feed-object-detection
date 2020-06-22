from configparser import ConfigParser
from utils.darknet import DarkNet

# main function
def main():
    # create parser instance
    config = ConfigParser()

    # Read detection.cfg
    config.read('config/model.cfg')
    darknet_base_dir = config.get('darknet', 'darknet_base_dir')
    darknet_object_data = config.get('darknet', 'darknet_object_data')

    yolov3_config_file = config.get('yolov3', 'yolov3_config_file')
    yolov3_train_weights_file = config.get('yolov3', 'yolov3_train_weights_file')

    log_file = config.get('output', 'log_file')
    graph_file = config.get('output', 'graph_file')

    if config.get('run_mode', 'shell') == "True":
        shell = True
    else:
        shell = False

    darknet = DarkNet(darknet_base_dir, darknet_object_data, yolov3_config_file, yolov3_train_weights_file,
                      log_file, graph_file, shell)

    # Begin model training
    process = darknet.train()

    # Wait for training to complete
    darknet.wait(process)

    # Plot Neural Network loss graph
    darknet.plot_loss()

if __name__ == "__main__":
    main()
