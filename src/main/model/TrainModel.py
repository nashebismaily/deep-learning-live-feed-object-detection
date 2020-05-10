from src.main.model import DarkNet as dn
from configparser import ConfigParser
import os

# main function
def main():
    parser = ConfigParser()
    parser.read(os.path.join(os.path.dirname(__file__), '../../../config', 'darknet.cfg'))
    darknet_base_dir = parser.get('darknet_config', 'darknet_base_dir')
    darknet_object_data = parser.get('darknet_config', 'darknet_object_data')
    yolov3_config_file = parser.get('darknet_config', 'yolov3_config_file')
    yolov3_weights_file = parser.get('darknet_config', 'yolov3_weights_file')
    log_file = parser.get('darknet_config', 'log_file')

    darknet = dn.DarkNet(darknet_base_dir,
                      darknet_object_data,
                      yolov3_config_file,
                      yolov3_weights_file,
                      log_file)

    process = darknet.train()
    darknet.tail_log(process)
    darknet.plot_loss()

if __name__ == "__main__":
    main()