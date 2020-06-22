import cv2 as cv

# scans available devices and returns the available cameras connected to the device
def main():
    device_list = []
    id = 0
    while True:
        vc = cv.VideoCapture(id)
        if vc.read()[0]:
            device_list.append(id)
            id += 1
            cv.destroyAllWindows()
        else:
            break
        vc.release()
    cv.destroyAllWindows()
    print(device_list)

if __name__ == "__main__":
    main()
