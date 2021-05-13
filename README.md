A Python based Image and Video Editor
=====================================

The documentation, including a thorough explination of the methods implemented in the editor, is included in this [Final Project Report](https://drive.google.com/file/d/1-aH2Rr_w4T_-lZrQVSJTMinX2hhDBoSt/view?usp=sharing). You can find examples of the video processing [here](https://www.youtube.com/watch?v=bZTFA6XCHXo&t=91s&ab_channel=ColeBateman) and [here](https://www.youtube.com/watch?v=d1EGuYXyT1Q&ab_channel=ColeBateman).

## Image and Video Processor

Our application can take in either a .mp4, .jpg, or .png and apply a wide variety of filters and tone mappings to them. Additionally, we have implemented an automatic background remover tool. 

## Setup

Navigate to the main directory and  run the following in the command line to ensure that your branch is properly set up and trained to run our code.

```bash 
bash setup.sh
```

You will also require the following packages for python. Please be sure to have them installed before hand. 

```bash
tensorflow, PIL, PySimpleGUI, cv2, numpy, io, os, sys, datetime
```

If you are not sure that these packages are installed, you can run the following code to update your environement:

```bash
pip3 install -r requirements.txt
```

Note, you may also need to use ``` pip ``` instead

Additionally, you may need to instal TKinter. If for some reason pip cannot successfully retrieve the library, you can use the following:

```bash
sudo apt-get install tk
```

## Usage

All you need to do to run the code is the following:

```bash
python3 interface.py
```					

## Getting Started 

You can use the integrated file explorer to search through any directory for your inputted image.

However, if you plan on using the foreground-background separator you must have the image saved to the included Images folder.  

Any processed images or video will be saved to the same directory that the code was run in. You can name your processed file from the box next to the Save button in the interface. To render video, you must select the Render checkbox at the top left of the application. Video will be saved under render.avi, and cannot be changed. Make sure that you save this render somewhere else before rendering a new video, or else you will overwrite the original. 

Finally, we have provided a series of images in the Test Images directory as a useful start to your exploration of the system

## Limitations

We hope you have fun playing with our application! Note, some combinations of effects can cause the system to crash, namely those including Threshold, Canny, Hue, or Enhance. We believe this to be due to the change in color channels these filters create in the input image.

Finally, some functionality is not usable between both image and video processing, either due to processing constraints or unfinished implementation. For example, we cannot run our background-foreground separation utility on a video without the process taking extremely long.

Enjoy playing with the system! 