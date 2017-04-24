# Generate a model for the behavioral cloning project - SDCND project3
# Created by Ron Danon - rondanon@gmail.com

#%% define the generator %%#
import cv2
import numpy as np
from sklearn.utils import shuffle

angleDeflection = 0.3
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images =[]
            angles = []
            for batch_sample in batch_samples:
                for i,deflection in zip(range(3),[0.0,angleDeflection,-angleDeflection]):
                    originPath = batch_sample[i].split('\\')
                    path = '../recorded_data/' + originPath[-3] + '/IMG/' + originPath[-1]
                    image = cv2.imread(path)
                    measurement = float(batch_sample[3]) + deflection
                    image_flipped = np.fliplr(image)
                    measurement_flipped = -measurement
                    images.extend([image,image_flipped])
                    angles.extend([measurement,measurement_flipped]) 
#                originPath = batch_sample[0].split('\\')
#                path = '../recorded_data/' + originPath[-3] + '/IMG/' + originPath[-1]
#                img = cv2.imread(path)
#                images.append(img)
#                angles.append(float(batch_sample[3]))
            
            X = np.array(images)
            y = np.array(angles)
            yield (X, y)

#%% load the data %%#
import csv
samples = []
dataFolders = ['lap11','lap22','lap11_rev','lap22_rev',
                     'strong_turn_left1','strong_turn_right1','strong_turn_left2','strong_turn_right2',
                     'strong_turn_left3','strong_turn_right3','strong_turn_left4','strong_turn_right4', 
                     'recovering1','recovering2',
                     'track2_lap1','track2_rev_lap1',
                     'left_dirt_turn1']
for folder in dataFolders:
    path = '../recorded_data/' + folder + '/driving_log.csv'
    with open(path) as csvFile:
        reader = csv.reader(csvFile)
        for line in reader:
            samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, valid_samples = train_test_split(samples, test_size = 0.25)


#%% the keras model %%#
BATCH_SIZE = 32
EPOCHS = 1
topCrop = 63
botCrop = 23
drop = 0.5

train_generator = generator(train_samples,batch_size=BATCH_SIZE)
valid_generator = generator(valid_samples,batch_size=BATCH_SIZE)

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Cropping2D, Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((topCrop,botCrop),(0,0)), input_shape=(160,320,3))) #output 74x320x3
model.add(Lambda(lambda x: x/127.5-1.0))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu')) #output 70x316x24
#model.add(MaxPooling2D(pool_size=(2, 2)))  #output 35x158xx24
model.add(Convolution2D(32,5,5,subsample=(2,2),activation='relu')) #output 31x154xx32
#model.add(MaxPooling2D(pool_size=(2, 2)))  #output 16x77xx32
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu')) #output 12x73xx48
#model.add(MaxPooling2D(pool_size=(2, 2)))  #output 6x37xx48
model.add(Convolution2D(64,3,3,activation='relu')) #output 4x35xx64
model.add(Convolution2D(64,3,3,activation='relu')) #output 2x33xx64
model.add(Flatten())
model.add(Dense(1164,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

#%% train model %%#
model.compile(loss='mse',optimizer='adam',)
history = model.fit_generator(train_generator,samples_per_epoch=len(train_samples),
                              validation_data=valid_generator,nb_val_samples=len(valid_samples),
                              nb_epoch=EPOCHS)
model.save('model.h5')