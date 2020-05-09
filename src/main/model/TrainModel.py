from src.main.model import DarkNet as dn

# main function
def main():
    darknet = dn.DarkNet('/root/darknet',
                      'stopsign-obj.data',
                      'stopsign-yolov3.cfg',
                      'darknet53.conv.74',
                      'out.log')

    process = darknet.train()
    darknet.tail_log(process)
    darknet.plot_loss()

if __name__ == "__main__":
    main()