# **Behavioral Cloning Project** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/center.jpg "center lane"
[image3]: ./examples/recover1.jpg "Recovery Image1"
[image4]: ./examples/recover2.jpg "Recovery Image2"
[image5]: ./examples/recover3.jpg "Recovery Image3"
[image6]: ./examples/normal.jpg "Normal Image"
[image7]: ./examples/flipped.jpg "Flipped Image"
[image8]: ./examples/middle.jpg "middle Image"
[image9]: ./examples/left.jpg "left Image"
[image10]: ./examples/right.jpg "right Image"

### Rubric Points
##### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
##### **Files Submitted & Code Quality**

**1. Submission includes all required files and can be used to run the simulator in autonomous mode**

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 showing the recorded successful drive of track1 

**2. Submission includes functional code**
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

**3. Submission code is usable and readable**
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

##### Model Architecture and Training Strategy

**1. An appropriate model architecture has been employed**
My model is based on the model used in [NVIDIA's paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) on end-to-end learning. It starts with a crop of the image to its interest area, followed by mormalization to the range of [-1,1] using Keras lambda layer. Then, 5 convolution layers are stacked with dpeths growing from 24 to 64. Each layer is followed by an activation of RELU. Lastly, the network is connected to 4 fully-connected layers. A Dropout layer (with keep probability of 0.5) is applied to each of these fully-connected layers. The last layer is connected to 1 output which has no activation.

**2. Attempts to reduce overfitting in the model**
The model contains dropout layers (with keep probability of 0.5) between all the fully connecteed layers, in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

**3. Model parameter tuning**
The model used an adam optimizer, so the learning rate was not tuned manually 

**4. Appropriate training data**
Training data was chosen to keep the vehicle driving on the middle of the road. I used a combination of center lane driving - CW and CCW - of track 1, recovering from the left and right sides of the road, a few smooth turnings and a recoring of the dirt turn.
For details about how I created the training data, see the next section. 

##### Model Architecture and Training Strategy
**1. Solution Design Approach**
The overall strategy for deriving a model architecture was to start with a simple CNN to have fast traning sets, so I could quickly check that my code is working. Later I based my model on [NVIDIA's paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) which performed very well.

My first step was to use a convolution neural network model similar to LeNet. I thought this model might be appropriate because it is strong enough for callsification tasks, and it is reletavly quick for traning. Aftet I validated my code works as excpected, I tried to achieve good results with this model, but with no success. Therfore, I used a model based on [NVIDIA's paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) which performed well.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 
To combat the overfitting, I modified the model with dropout layers between the fully connected layers. Furthermore, I recorded more generic data and augmented more data.

Then I ran the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially at the dirt curve after the bridge. To improve the driving behavior in these cases, I recorded more data for these specific areas, and retrained the already trained model on these data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

**2. Final Model Architecture**
My model is based on the model used in [NVIDIA's paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) on end-to-end learning. It starts with a crop of the image to its interest area, followed by mormalization to the range of [-1,1] using Keras lambda layer. Then, 3 convolution layers are stacked with a kernel size of 5x5, a stride of 2x2 and depths of 24, 36, 48 by order, where each layer is followed by an activation of RELU. Afterwards, 2 convolution layers are stacked with a kernel size of 3x3, a stride of 1x1 and depths 64 each, where each layer is followed by an activation of RELU. Lastly, the network is connected to 4 fully-connected layers with sizes of 1164, 100, 50 and 10 by order, where the first 2 are with a RELU activation, and the last 2 are with no activation. A Dropout layer (with keep probability of 0.5) is applied to each of these fully-connected layers. The last 10 neurons are connected to 1 output which has no activation.

**3. Creation of the Training Set & Training Process**
To capture good driving behavior, I first trained my driving in the simulator. I used the mouse controlling option instead of the arrows in order to generate smoother data. 
The I recorded two laps - one CW and one CCW - on track 1 using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover in cases where it loses the center and swerve to either side. These images show what a recovery looks like. The first one is the beginning of the recovery where the car drives close to the lane line (the recording starts at this point but with a strong wheel angle); the second is the middle of the recovery, where the car starts to go back to the center of the lane; and the last is the end of the recovery.

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data set, I also flipped images and angles thinking that this would help to generalize and help overcome overfitting. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Lastly, I used the left and right camera data for each image, adding a corresponding 0.1 deflection to the steering angle. Here are center, left and right images of the same loaction:

![alt text][image8]
![alt text][image9]
![alt text][image10]

After the collection and augmentation process, I had ~60.000 images. I then preprocessed this data by croping it with 63 pixels form the top and 23 pixels from the bottom. Then I normalized each pixel in the image with: (val - 127.5)/127.5 which normalize to the range of [-1,1] 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by the growing loss of the validation set after this epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
