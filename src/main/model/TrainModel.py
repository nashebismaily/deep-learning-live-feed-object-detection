from src.main.model import DarkNet as dn
from configparser import ConfigParser
from src.main.utils.Parser import Parser

# main function
def main():
    parser = ConfigParser()
    d_darknet = Parser.get_config(parser, 'darknet.cfg', 'darknet')
    darknet_base_dir = d_darknet.get('darknet_base_dir')
    darknet_object_data = d_darknet.get('darknet_object_data')
    yolov3_config_file = d_darknet.get('yolov3_config_file')
    yolov3_train_weights_file = d_darknet.get('yolov3_train_weights_file')
    log_file = d_darknet.get('log_file')
    graph_file = d_darknet.get('graph_file')

    darknet = dn.DarkNet(darknet_base_dir,
                      darknet_object_data,
                      yolov3_config_file,
                      yolov3_train_weights_file,
                      log_file,
                      graph_file)

    process = darknet.train()
    darknet.wait(process)
    darknet.plot_loss()

if __name__ == "__main__":
    main()