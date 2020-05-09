from src.main.model import DarkNet as dn
from configparser import ConfigParser

# main function
def main():

    parser = ConfigParser()
    parser.read('darknet.cfg')
    darknet_base_dir = parser.get('video_capture_config', 'darknet_base_dir')
    darknet_object_data = parser.get('video_capture_config', 'darknet_object_data')
    yolov3_config_file = parser.get('video_capture_config', 'yolov3_config_file')
    yolov3_weights_file = parser.get('video_capture_config', 'yolov3_weights_file')
    log_file = parser.get('video_capture_config', 'log_file')

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