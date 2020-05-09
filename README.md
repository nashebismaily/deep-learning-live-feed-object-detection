# Deep Learning Live Feed Object Detection

This project uses a Convolutional Neural Network (CNN) to automatically detect and track objects in real-time video feeds. The images used for training are annotated in yolov3 format with labelimg. DarkNet is used to train the model on GPU's and the output is collected to perform an analysis on the loss function. The OpenCV library is used to capture live stream video. Each frame is captured, processed with the model, integrated with an object detection bounding box, and displayed to the end user in real-time. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Python3
cv2
matplotlib
google_images_search

```
pip3 install cv2 
pip3 matplotlib 
pip3 google_images_search
```

### Installing

Clone this repository

```
https://github.com/nashebismaily/deep-learning-live-feed-object-detection.git
```

Use DownloadImages.py to download raw images for the classification objects

```
python3 DownloadImages.py <output_directory>
```

Use labelimg to annotate each image

![alt text](resources/icons/labelImg.png)

Train the Model and determine the appropriate model weights to choose based on the average loss

```
python3 TrainModel.py
```

![alt text](resources/icons/darknetlossgraph.png)

Run the live stream video detection. This requires a connected web camera to your device.

```
python3 DetectObjects.py
```

Here's the result

TODO

## Running the tests

Each module has its own set of unit test that can be run.

```
python3 TestDarkNet.py
python3 TestCaptureStraem.py
```

## Deployment

The CaptureStream class contains helper functions which will detect available web cameras connected to your device.
After you determine which device you wish to use, add its device ID into the DetectObjects.py configuration file.

## Built With

* [DarkNet](https://pjreddie.com/darknet/)
* [Yolov3](https://pjreddie.com/darknet/yolo/)
* [OpenCV2](https://pypi.org/project/opencv-python/)

## Authors

* **Nasheb Ismaily** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


