from src.main.model import DarkNet as dn
from configparser import ConfigParser
from src.main.utils.Parser import Parser

# main function
def main():
    parser = ConfigParser()

    # Read darknet.cfg, darknet section
    d_darknet = Parser.get_config(parser, 'darknet.cfg', 'darknet')
    darknet_base_dir = d_darknet.get('darknet_base_dir')
    darknet_object_data = d_darknet.get('darknet_object_data')

    # Read darknet.cfg, yolov3 section
    d_darknet = Parser.get_config(parser, 'darknet.cfg', 'yolov3')
    yolov3_config_file = d_darknet.get('yolov3_config_file')
    yolov3_train_weights_file = d_darknet.get('yolov3_train_weights_file')

    # Read darknet.cfg, output section
    d_darknet = Parser.get_config(parser, 'darknet.cfg', 'output')
    log_file = d_darknet.get('log_file')
    graph_file = d_darknet.get('graph_file')

    darknet = dn.DarkNet(darknet_base_dir,
                      darknet_object_data,
                      yolov3_config_file,
                      yolov3_train_weights_file,
                      log_file,
                      graph_file)

    # Begin model training
    process = darknet.train()

    # Wait for training to complete
    darknet.wait(process)

    # Plot Neural Network loss graph
    darknet.plot_loss()

if __name__ == "__main__":
    main()