# End-to-end autonomous driving using deep neural networks with Udacity (open source) virtual driving environment
---

## Overview

The goal for this project was to:
* Use the simulator to collect data of good driving behavior, including some edge cases and examples of recovery when driving behavior becomes critical/eratic.
* Build a convolution neural network using Keras that predicts steering angles from images, captured from (virtual) sensors/cameras on the body of the car
* Train and validate a model based on Nvidia's end-to-end deep learning architectures
* Test that the model successfully drives around the track safely (without leaving the road)
* Summarize the results with a written report

<b> Result : The vehicle was able to drive autonomously around the track within the lane lines at all times. </b>

[//]: # (Image References)

[image1]: ./images/images/center_driving.jpg "Image collected from center camera"
[image2]: ./images/images/left_recovery.jpg "Vehicle recovering to the right as it approaches the left lane"
[image3]: ./images/images/right_recovery.jpg "Vehicle recovering to the left as it approaches the right lane"
[image4]: ./images/images/flipped.jpg "For augmentation, each image is flipped (left-to-right)"
[image5]: ./images/images/nvidia.png "Nvidia's model architecture for end-to-end deep learning autonomous driving"

---
### Files & Code 

The project includes the following files:
* model.py: containing the script to create and train the model. The annotated code shows the pipeline used for training and validating the model.
* drive.py: for driving the car in autonomous mode within Udacity's virtual driving environment
* model.h5: containing the trained convolution neural network (too large to put on Github)

#### Running the autonomous vehicle within the virtual enviroment
Using the Udacity-provided simulator and the above drive.py code, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
### Collecting and pre-processing training data

Training data was chosen to represent various driving styles, including "examples" of safe driving and how to recover from unsafe situations. 

<table class="image">
<caption align="bottom">{{ Borejwiorjiowew }}</caption>
<tr><td><img src="./images/images/center_driving.jpg" alt="{{ SECOND DES }}"/></td></tr>
</table>


The data is a combination of:
* <b>Center lane driving: </b>recorded from two laps in the forward (anti-clockwise) direction plus one lap in the backward (clockwise) direction. Here is an example image of center lane driving (Fig 1):
* <b>Recovery driving: </b>recorded when the car is centering itself from the edge of the lane. This recovery data was recorded from both sides of the lane (left- and right-recovery). This was so that the vehicle could learn to recover whenever it approached the lane lines. 
* <b>Data augmentation</b>: the track contains mostly left turns. Hence, the training images were flipped (left-to-right) to increase the model's performance on right turns. For example, here is an image that has then been flipped:
![Fig1][image1]
![Fig2][image2]
![Fig3][image3]
![Fig4][image4]

Ultimately, during training approximately <b> 42,000 </b> images were collected from the three camera sensors around the vehicle (left/center/right). 

The images were then cropped, as only ~50% of their height contained useful information (the lower section of the image was the vehicle's hood/side, while the upper section displayed the sky, which isn't constructive to the model). Finally, within the model, they were normalized so image values (across the three RGB channels) were between -0.5 and 0.5.

### Model Architecture, design, and tuning

#### 1. Architecture
The model is based on Nvidia's "End-to-End Deep Learning for Self-Driving Cars" found <a href = "https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/"> here </a>. It is a convolution neural network that is constructed using the following architecture: 
* <b> images are normalized </b> (line 88) so that their values lie between [-0.5, 0.5]
* <b> images put through four convolutional feature maps </b> with three with 5x5 kernels, then two with 3x3 kernels, (lines 91-94), followed by
* <b> four fully-connected layers </b> (lines 98-101). The model includes ReLU layers in between the convolution layers to introduce non-linearlity. 

#### 2. Solution Design Approach

The high-level strategy for deriving a model architecture was to
* <b> start with a simple model </b> (e.g. simple linear regression, not a deep neural network), and see where it failed. The linear regression managed to drive the car for ~5 seconds before going off the track. This was to be expected as a linear regression is far too simple...

* <b> improving the model by adding more layers </b>, beginning with LeNet and then gradually converging on the Nvidia end-to-end deep neural network. LeNet worked well on a previous project (identifying traffic signs), however it had trouble turning the car around corners accurately. Nvidia's solution has proved effective in real-world scenarios, and so it ultimately replaced LeNet in this model. 

* <b> testing the model to see where the vehicle failed on the track. </b> At those points of the track, I collected additional training data. For example, there were instances where one wheel left the lane on a turn – in that case, I added more training data collected from that position in the track.

#### 3. Model parameter tuning and reducing model overfitting 

The data was separated between training (80%) and validation (20%). The model was tested by running it through the simulator in autonomous mode and ensuring that the vehicle beyhaved safely (e.g. stayed on the track and did not perform any behavior that would be dangerous to passengers in the vehicle).

The model used an adam optimizer, so the learning rate was not tuned manually (line 105). Three epochs were run as, beyond that, the model began to overfit the data. 



