# Deep Learning Live Feed Object Detection

This project uses a Convolutional Neural Network (CNN) to automatically detect and track objects in real-time video feeds. The images used for training are annotated in yolov3 format with labelimg. DarkNet is used to train the model on GPU's and the output is collected to perform an analysis on the loss function. The OpenCV library is used to capture live stream video. Each frame is captured, processed with the model, integrated with an object detection bounding box, and displayed to the end user in real-time. 


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

You can chose to use the pre-trained model

Or you can train your own model

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

### Clone Repository

```
https://github.com/nashebismaily/deep-learning-live-feed-object-detection.git
```

### Option 1: Use the Pre-Trained Model

Download the pre-trained model from here 

[ML Model](https://srv-file6.gofile.io/download/Bfj367/yolov3.weights)

Place yolov3.weights into this folder:

```
deep-learning-live-feed-object-detection/model/config
```

### Option 2: Train a custom Model

1. Download a set of images for an object you want to classify.

2. Use labelImg to annotate each image, ensure YOLO is set as the output type

    ```
    labelImg
    ```

    ![alt text](resources/icons/labelImg.png)

3. Update the objects.names file to contain a list of objects that have been labeled

    ```
    deep-learning-live-feed-object-detection/model/config/objects.names 
    ```

4. Train the Model

    ```
    python3 deep-learning-live-feed-object-detection/src/main/model/TrainModel.py
    ```

5. Once training is complete, observe the loss graph found here

    ```
    deep-learning-live-feed-object-detection/model/graphs
    ```

    ![alt text](resources/icons/darknetlossgraph.png)

6. Select the appropriate weight that correlates to the elbow in the loss graph and move that here

    ```
    deep-learning-live-feed-object-detection/model/config/yolov3.weights
    ```

### Run Image Detection


### Run Video Detection


### Run Live Camera Detection


### Results:

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


