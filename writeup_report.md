
# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/nvidia_model.png "Nvidia model"
[image3]: ./examples/figure_data_distribution.png "Data distribution"
[image4]: ./examples/figure_data_distribution_after_deleting.png "Data distribution modified"
[image5]: ./examples/figure_own_data_two_tracks_with_deleted_data.png "Loss"


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

In this project we are encouraged to use a working [CNN](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) developed by Nvidia. The network architecture is shown in the image below. It consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. The input image is split into YUV planes and passed to the network.

![alt text][image2]

The first layer of the network performs image normalization. The convolutional layers were designed to perform feature extraction and were chosen empirically through a series of experiments that varied layer configurations. They use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers.

They follow the five convolutional layers with three fully connected layers leading to an output control value which is the inverse turning radius.

I decided to use a this convolutional neural network with small changes. In this case, the input image is not split into YUV planes but we work with RGB images directly, and its shape is different. Besides, within the model itself, the input image is preprocessed by a cropping layer, which is useful for choosing the area of interest that excludes the sky and/or the hood of the car, and a normalization layer (BatchNormalization in keras). As for the activation function, I used ELUs to achieve smoother transitions.

Moreover, in order to prevent the overfitting, I decided to implement this time the L2 regularization method. It is quite straightforward using it with Keras, you only need to use the W_regularizer parameter in the definition of the layers.

The weights of the network were trained to minimize the mean squared error between the steering command output by the network and the command of either the human driver. This minimization was carried out using the Adam optimizer, so the learning rate was not tuned manually.

In order to gauge how well the model was working, I split the image and steering angle data into a training and validation set. It should be noted too that I used generators due to the great amount of data generated.

After several trials I got the best results augmenting and reducing some data to achieve a well balanced training set (see subsection 3) and the evolution of the mse for both sets (training and validation) for this case is shown below:

![alt text][image5]

At the end of the process, the vehicle is able to drive autonomously around the track 1 and track 2 without leaving the road.


#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes. Here is a visualization of the architecture.

![alt text][image1]


Total params: 982,179

Trainable params: 981,999

Non-trainable params: 180

#### 3. Creation of the Training Set & Training Process

In order to achieve a balanced dataset, I decided to record three clockwise laps and two counterclockwise for track 1 and two clockwise laps and one counterclockwise lap for track 2. On the other hand, to avoid the recovery laps, I used all three camera images. After the collection process, I had 55938 number of data points.

Here is the original distribution of the training data (the x-axis corresponds to steering angles and the y-axis is the data point count for that angle range; the black line represents what would be a uniform distribution):

![alt text][image3]

As you can see, because the test track includes long sections with very slight or no curvature, the data captured from it tended to be heavily skewed toward low and zero turning angles. This created a problem for the neural network, which then became biased toward driving in a straight line and was easily confused by sharp turns. That's why I dropped a bunch of those data points in the middle, and the result looked like this:

![alt text][image4]

Now, we have a more balanced set. However, there still are less occurrencies for big angles. In order to compensate this, I decided to augment the data by flipping some images. This is carried out in the generator (line 09 - model.py). I choose a random image and if this has an associated angle greater than 0.33, I create a new image flipping the original one.

At this point, the model performed very well — driving reliably around both test tracks multiple times.

### Conclusion

This project made one thing very clear — it is very important to be able to record smooth and balanced data. With this solution the car is able to drive itself reliably around both tracks, however if I change the quality of the graphics in the simulator, the vehicle in track 2 does not have a good performance due to changes in lightness. In order to achieve a good performance in these terms it would be necessary more preprocessing.



```python

```
