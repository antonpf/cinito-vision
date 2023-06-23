# GStreamer object detection with Coral

This folder contains code using [GStreamer](https://github.com/GStreamer/gstreamer) to
obtain camera images and perform object detection on the Edge TPU.

This code works on the Coral Dev Board mini using the Coral Camera.

## Set up your device

1.  First, be sure you have completed the [setup instructions for your Coral
    device](https://coral.ai/docs/setup/). If it's been a while, repeat to be sure
    you have the latest software.

    Importantly, you should have the latest TensorFlow Lite runtime installed
    (as per the [Python quickstart](
    https://www.tensorflow.org/lite/guide/python)). You can check which version is installed
    using the ```pip3 show tflite_runtime``` command.

    > **Note**
    > Also make sure the correct date is set, otherwise there may be problems with updates or cloning the Ropo. The current date can be set with the command 
    > ```
    > sudo date +%Y%m%d -s "YYYYMMDD"
    > ```

2.  Clone this Git repo onto your computer or Dev Board:

    ```
    git clone https://github.com/antonpf/cinito-vision.git
    cd cinito-vision
    ```

3.  Install the GStreamer libraries (if you're using the Coral Dev (mini) Board, you can skip this):

    ```
    cd gstreamer

    bash install_requirements.sh
    ```

## Run the detection demo (Fine-tuned SSD model)

```
python3 detect.py
```

Likewise, you can change the model and the labels file using ```--model``` and ```--labels```.

By default, the object detection use the attached Coral Camera. If you want to use a USB camera,
edit the ```gstreamer.py``` file and change ```device=/dev/video0``` to ```device=/dev/video1```.

