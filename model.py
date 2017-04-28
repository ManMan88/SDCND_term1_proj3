# Generate a model for the behavioral cloning project - SDCND project3
# Created by Ron Danon - rondanon@gmail.com

#%% define the generator %%#
# import modules
import cv2
import numpy as np
from sklearn.utils import shuffle

angleDeflection = 0.1  #the deflection of the steering angle for the side cameras 
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images =[]
            angles = []
            for batch_sample in batch_samples:
                #Iterate over the 3 images (middle left and right) and the corresponding deflections
                for i,deflection in zip(range(3),[0.0,angleDeflection,-angleDeflection]):
                    originPath = batch_sample[i].split('\\')
                    path = '../recorded_data/' + originPath[-3] + '/IMG/' + originPath[-1]
                    image = cv2.imread(path)
                    measurement = float(batch_sample[3]) + deflection
                    image_flipped = np.fliplr(image)    #create flipped image
                    measurement_flipped = -measurement  #create flipped steering angles
                    images.extend([image,image_flipped])
                    angles.extend([measurement,measurement_flipped]) 
            
            #Transform lists to numpy arrays for keras
            X = np.array(images)
            y = np.array(angles)
            yield (X, y)

#%% load the data %%#
import csv
samples = []
#Define the dolders with the training data
dataFolders = ['lap22','lap22_rev',
                     'strong_turn_left1','strong_turn_right1','strong_turn_left2','strong_turn_right2',
                     'strong_turn_left3','strong_turn_right3','strong_turn_left4','strong_turn_right4', 
                     'recovering1','recovering2',
#                     'track2_lap1','track2_rev_lap1',
                     'left_dirt_turn1', 'more_turns']
for folder in dataFolders:
    path = '../recorded_data/' + folder + '/driving_log.csv'
    with open(path) as csvFile:
        reader = csv.reader(csvFile)
        for line in reader:
            samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, valid_samples = train_test_split(samples, test_size = 0.2) #Set the validation set to 20% of the data


#%% the keras model %%#
BATCH_SIZE = 128
EPOCHS = 2
topCrop = 63
botCrop = 23
drop = 0.5

#set the generator
train_generator = generator(train_samples,batch_size=BATCH_SIZE)
valid_generator = generator(valid_samples,batch_size=BATCH_SIZE)

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Cropping2D, Convolution2D
from keras.layers.pooling import MaxPooling2D

#Create the Nvidia CNN architecture
model = Sequential()
model.add(Cropping2D(cropping=((topCrop,botCrop),(0,0)), input_shape=(160,320,3))) #output 74x320x3
model.add(Lambda(lambda x: x/127.5-1.0))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu')) #output 35x158x24
model.add(Convolution2D(32,5,5,subsample=(2,2),activation='relu')) #output 16x77xx32
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu')) #output 6x37xx48
model.add(Convolution2D(64,3,3,activation='relu')) #output 4x35xx64
model.add(Convolution2D(64,3,3,activation='relu')) #output 2x33xx64
model.add(Flatten())
model.add(Dense(1164,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))
#model.load_weights('model.h5')

#%% train model %%#
model.compile(loss='mse',optimizer='adam',)
history = model.fit_generator(train_generator,samples_per_epoch=len(train_samples)*2*3,
                              validation_data=valid_generator,nb_val_samples=len(valid_samples),
                              nb_epoch=EPOCHS)
model.save('model.h5')