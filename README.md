# Deep Learning Live Feed Object Detection

This project uses a Convolutional Neural Network (CNN) to automatically detect and track objects in real-time video feeds. The images used for training are annotated in yolov3 format with labelimg. DarkNet is used to train the model on GPU's and the output is collected to perform an analysis on the loss function. The OpenCV library is used to capture live stream video. Each frame is captured, processed with the model, integrated with an object detection bounding box, and displayed to the end user in real-time. 

 The CNN splits the input image into a grid of N×N cells. The center of the defined bouding box(i.e. ground truth) is identified as one of the cells, then that cell becomes responsible for detecting that object
 
* Coordinates of B bounding boxes - the CNN predicts 4 coordinates for each bounding box (tx,ty,tw,th). Where x and y are set to be offset of a cell location.
* Objectness score - the probability that the cell contains an object. The objectness score is passed through a sigmoid function to be treated as a probability with a value range between 0 and 1. 
* Class prediction - if the bounding box contains an object, the network predicts the probability of K number of classes. Where K is the total number of classes in your problem.

**Follow [this guide](#getting-started) and you will be able to create your own live stream object detection:**

![car](resources/gifs/nasheb-car-short.gif)


**DEVELOPMENT IN PROGRESS**

## Getting Started

These instructions will allow you to run your own live-stream object detection.

You can chose to [use the pre-trained model](#option-1-use-the-pre-trained-model)
or you can [train your own model](#option-2-train-a-custom-model)

### Prerequisites

* python3
* opencv-python
* matplotlib
* labelimg
* imutils
* google_images_search

### Clone Repository

This codebase has been tested on RHEL7 and MacOS-PyCharm

1. Clone this repository
    ```
    https://github.com/nashebismaily/deep-learning-live-feed-object-detection.git
    ```
2. Install packages
    ```
    pip3 install opencv-python
    pip3 install matplotlib
    pip3 install imutils
    pip3 install google_images_search
    
    yum install labelimg
    ```
    **Note**: Ue brew instead of yum on MacOSx
    
3. **OPTIONAL**: Donwload the [PyCharm IDE](https://www.jetbrains.com/pycharm/download/)

4. **OPTIONAL**: Configure your project interpreter as follows:

    ![interpreter](resources/icons/interpreter.png)

### Download Base Weights Files

[Base Weights File](https://srv-file14.gofile.io/download/KySboR/darknet53.conv.74)

Place darknet53.conv.74 into this folder:

```
deep-learning-live-feed-object-detection/model/weights
```

### OPTION 1: Use the Pre-Trained Model

Download my pre-trained model:

[Pre-Trained ML Model](https://srv-file6.gofile.io/download/Bfj367/yolov3.weights)

Place yolov3.weights into this folder:

```
deep-learning-live-feed-object-detection/model/weights
```

### OPTION 2: Train a Custom Model

1. Add your google api keys, update your image search, and download images:

    ```
    deep-learning-live-feed-object-detection/images/config/images.cfg
    
    [image_download]
    search_string=car,truck,buss,motercycle,bicycle,dog,cat,fire hydrant,stop sign
    base_download_dir=/Downloads
    google_api_key=
    google_project_cx=
    
    deep-learning-live-feed-object-detection/images/DownloadImages.py
    ```
    
    **Note:** If you don't have google api keys and want a quick workaround to download images in bulk, use this [Chrome Extension](https://chrome.google.com/webstore/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm?hl=en)

2. Use labelImg to annotate each image

    * Select input directory (open dir) which contains the downloaded images
    * Select output directory (save dir) to be the same as the input directory
    * Ensure YOLO is set as the output type (Instead of the default: <Pascal/VOC>)
    * Draw and label a bounding box for EACH object that appears in the photo
    * Save the photo when you are done and proceed to the next one.
    ```
    labelImg
    ```

    ![labelImg](resources/icons/labelImg_example.png)
    
    **Note:** This will result in a .txt file for each photo which contains:
    ```
    <object-class> <x> <y> <width> <height>
    ```   

3. Update the objects.names file to contain a list of objects that have been labeled, one label per line

    ```
    deep-learning-live-feed-object-detection/model/yolov/objects.names 
    ```

4. Split the data into test and training sets. 
    
    * train.txt will contains absolute path to images for training (75% images)
    * test.txt will contains absolute path to images for testing (25% images)
    * label files need to be in the same directory as images
     
    ```
    deep-learning-live-feed-object-detection/model/yolov/train.txt
    
    /root/photos/image1.jpg
    /root/photos/image2.jpg
    /root/photos/image3.jpg
    ...
    
    deep-learning-live-feed-object-detection/model/yolov/test.txt

    /root/photos/image4.jpg
    /root/photos/image5.jpg
    /root/photos/image6.jpg  
    ...
    ```
5. Update the object definition files:
       
    * If you plan on running this on the command line

    ```   
    deep-learning-live-feed-object-detection/model/yolov/object.data
    
    classes:    number of labels (from objects.names)
    # Leave everything else as is
    ```
    
    * If you plan on running this in PyCharm
    ```   
    deep-learning-live-feed-object-detection/model/yolov/object_ide.data
    
    classes:    number of labels (from objects.names)
    train:      full/absolute path to train.txt
    valid:      full/absolute path to test.txt
    names:      full/absolute path to objects.names
    backup:     full/absolute path to weights directory
    ```
    
6. Download and install darknet to help model training

    ```
    cd /root # Or any other directory
    git clone https://github.com/pjreddie/darknet
    cd darknet
    make 
    ```
    **Note:** You will need to install your OS development tools for the 'make' command to work. 
    
     ```
     For RHEL:   yum groupinstall 'Development Tools'
     For MacOSx: Install command Line Tools for Xcode
     ``` 
    
7.  Configure darknet based directory to the install location of darknet

    ```
    deep-learning-live-feed-object-detection/model/config/model.cfg
    
    [darknet]
    darknet_base_dir=/root/darknet
    
    ```
8. Fine tune the model detection algorithm
    ```
    deep-learning-live-feed-object-detection/model/yolov/yolov3.cfg 
    
    Set 'batch' to 16
    Set 'subdivisions' to 4
    Set 'classes' to the number of objects to detect
    Set 'filters' to (classes + 5)* 3
    Set 'max_batches' to the number of classes * 2000 
    Set 'steps' to 80% and 90% of max_batches.
    ```
    
    **Note:** Use yolov3-tiny.cfg instead of yolov3.cfg if your running on a laptop or a compute constrained device. In this case, update the model configuration:
                            
    ```
    deep-learning-live-feed-object-detection/model/config/model.cfg
    
    [yolov3]
    yolov3_config_file=yolov3-tiny.cfg
    ```     
9.  Update your run mode:   

    ```
    deep-learning-live-feed-object-detection/model/config/model.cfg
    
    [run_mode]
    shell=True  #For command line use
    shell=False #For PyCharm IDE use
    ```

10. Train the Model

    ```
    cd deep-learning-live-feed-object-detection/model
    python3 train_model.py
    ```
    
11. View the training output here:

    ```
    deep-learning-live-feed-object-detection/model/logs
    ```

12. Once training is complete, observe the loss graph found here:

    ```
    deep-learning-live-feed-object-detection/model/graphs
    ```

    ![darknetlossgraph](resources/icons/darknetlossgraph.png)

13. Select the appropriate weight that correlates to the elbow in the loss graph (or choose the final weight) and move/rename that to:

    ```
    deep-learning-live-feed-object-detection/model/weights/yolov3.weights
    ```

    
### Detect Objects in Static Images

1. Add the path for the source and destination images into the config under the image section

    ``` 
    deep-learning-live-feed-object-detection/detection/config/detection.cfg

    input_image=/root/photos/input_photo.jpg
    output_image=/root/photos/output_photo.jpg
    ```

2. Run the static image object detection

    ```
    cd deep-learning-live-feed-object-detection/detection
    python3 detect_image.py
    ```

### Detect Objects in Video

1. Add the path for the source and destination videos into the config under the video section.
  
    ```
    deep-learning-live-feed-object-detection/detection/config/detection.cfg

    input_video=/root/videos/input_video.mov
    output_video=/root/videos/output_video.avi
    ```
    **Note:** The output video needs to be in .avi format. The input video can be any format.

2. Run the video object detection

    ```
    cd deep-learning-live-feed-object-detection/detection
    python3 detect_video.py
    ```

### Detect Objects in Camera

1. Scan your cameras to detect available camera id's

    ```
    cd deep-learning-live-feed-object-detection/detection/utils
    python3 scan_cameras.py

    [0]
    ```

2. Update detection.cfg with your selected device id

    ```
    deep-learning-live-feed-object-detection/detection/config/detection.cfg

    device_id=0
    ```

3. Run the live-stream  object detection

    ```
    cd deep-learning-live-feed-object-detection/detection
    python3 detect_camera.py
    ```

### OPTIONAL: Fine Tune the Bounding Box Blob

Fine tune the 4-dimensional blob from the input frame:

```
deep-learning-live-feed-object-detection/detection/config/detection.cfg

[blob]
output_width:   spatial width for output image
output_length:  spatial length for output image
scalefactor:    multiplier for image values
swapRB:	        flag which indicates that swap first and last channels in 3-channel image is necessary
crop:	        flag which indicates whether image will be cropped after resize or not
confidence:     requried confidence for succesfully detecting an image
threshold:      threshold used in non maximum suppression
```

### OPTIONAL: Fine Tune Video Capture

Fine tune the frame capture rate:

```
deep-learning-live-feed-object-detection/detection/config/detection.cfg

[video]
frames_per_second:  fps for output video
color:              color (True) or bw (False)
```

### OPTIONAL: Fine Tune Camera Capture

Fine tune the camera's frame output:

```
deep-learning-live-feed-object-detection/detection/config/detection.cfg

[camera]
frame_wait_ms:          amount of time to dispaly each processed frame
frame_pause_interval:   frame interval for bounding box reuse
```

## Built With

* [DarkNet](https://pjreddie.com/darknet/)
* [Yolov3](https://pjreddie.com/darknet/yolo/)
* [OpenCV2](https://pypi.org/project/opencv-python/)
* [Deep Learning](https://livebook.manning.com/book/grokking-deep-learning-for-computer-vision/chapter-7/v-8/1)

## Authors

* **Nasheb Ismaily** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details