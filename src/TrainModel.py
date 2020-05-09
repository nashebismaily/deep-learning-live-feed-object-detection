from src import DarkNet as dn

# main function
def main():
    # These will become input arguments
    darknet_base_dir = '/root/darknet'
    darknet_object_data = 'object.data'
    yolov3_config_file = 'yolov3.cfg'
    yolov3_weights_file = 'darknet53.conv.74'

    darknet = dn.DarkNet(darknet_base_dir, darknet_object_data, yolov3_config_file, yolov3_weights_file)
    darknet.train()

if __name__ == "__main__":
    main()