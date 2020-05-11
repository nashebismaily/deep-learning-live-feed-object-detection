from src.main.model import DarkNet as dn
from configparser import ConfigParser
from src.main.utils.Utils import Utils

# index of file in project
file_index = 4

# main function
def main():
    parser = ConfigParser()
    parser.read(Utils.get_relative_path(__file__, file_index, 'config/darknet.cfg'))
    darknet_base_dir = parser.get('darknet_config', 'darknet_base_dir')
    darknet_object_data = parser.get('darknet_config', 'darknet_object_data')
    yolov3_config_file = parser.get('darknet_config', 'yolov3_config_file')
    yolov3_weights_file = parser.get('darknet_config', 'yolov3_weights_file')
    log_file = parser.get('darknet_config', 'log_file')
    graph_file = parser.get('darknet_config', 'graph_file')


    darknet = dn.DarkNet(darknet_base_dir,
                      darknet_object_data,
                      yolov3_config_file,
                      yolov3_weights_file,
                      log_file,
                      graph_file,
                      file_index)

    process = darknet.train()
    darknet.wait(process)
    darknet.plot_loss()

if __name__ == "__main__":
    main()