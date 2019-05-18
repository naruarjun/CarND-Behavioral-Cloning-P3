import csv
import cv2
import numpy as np
from scipy import ndimage
import copy

#CREATING ARRAYS FOR IMAGES AND STEERING ANGLE
images = []
measurements = []

#READING IN ANTICLOCKWISE DATA
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#Sort through all lines - Extract path from file - Add the image to the array
for line in lines:
    for i in range(3):
        #EXTRACT PATH FROM CSV
        source_path = line[i]
        #READ IN IMAGE
        image = ndimage.imread(source_path)
        #GRAYSCALE
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        #READ IN CORRESPONDING STEERING ANGLE
        measurement = float(line[3])
        if i ==0:
            c = 0
        elif i == 1:
            c = 0.2
        else:
            c = -0.2
        #ADDING CORRECTION FACTOR IF LEFT OR RIGHT CAMERA IMAGE
        measurement = measurement+c
        #IF CENTER CAMERA IMAGE SHADOW AUGMENTATION IS APPLIED
        if i==0:
            new_image = copy.deepcopy(image)
            h, w = image.shape[0], image.shape[1]
            [x1, x2] = np.random.choice(w, 2, replace=False)
            k = h / (x2 - x1)
            b = - k * x1
            for j in range(h):
                c = int((j - b) / k)
                new_image[j, :c] = (image[j, :c] * .5).astype(np.int32)
            #APPENDING SHADOW AUGMENTED IMAGE TO DATASET
            images.append(new_image)
            measurements.append(measurement)
        #ADDING REGULAR IMAGE TO DATASET
        images.append(image)   
        measurements.append(measurement)

print(np.array(images).shape)
print(np.array(measurements).shape)


#READING IN CLOCKWISE DATA
lines = []
with open('./data_clockwise/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

for line in lines:
    for i in range(3):
        #EXTRACT PATH FROM CSV
        source_path = line[i]
        #READ IN IMAGE
        image = ndimage.imread(source_path)
        #GRAYSCALE
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        #READ IN CORRESPONDING STEERING ANGLE
        measurement = float(line[3])
        if i ==0:
            c = 0
        elif i == 1:
            c = 0.2
        else:
            c = -0.2
        #ADDING CORRECTION FACTOR IF LEFT OR RIGHT CAMERA IMAGE
        measurement = measurement+c
        #IF CENTER CAMERA IMAGE SHADOW AUGMENTATION IS APPLIED
        if i==0:
            new_image = copy.deepcopy(image)
            h, w = image.shape[0], image.shape[1]
            [x1, x2] = np.random.choice(w, 2, replace=False)
            k = h / (x2 - x1)
            b = - k * x1
            for j in range(h):
                c = int((j - b) / k)
                new_image[j, :c] = (image[j, :c] * .5).astype(np.int32)
            #APPENDING SHADOW AUGMENTED IMAGE TO DATASET
            images.append(new_image)
            measurements.append(measurement)
        #ADDING REGULAR IMAGE TO DATASET
        images.append(image)   
        measurements.append(measurement)

print(np.array(images).shape)
print(np.array(measurements).shape)

#AUGMENTING BY ADDING FLIPPED IMAGE FOR EACH IMAGE ALREADY IN ARRAY
augmented_images, augmented_measurements = [],[]
for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

print(np.array(augmented_images).shape)
print(np.array(augmented_measurements).shape)


X_train = np.expand_dims(np.array(augmented_images),axis=3)
y_train = np.array(augmented_measurements)

print(X_train.shape)
print(y_train.shape)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


#CREATING THE MODEL 
model = Sequential()
#NORMALIZE AND MEAN CENTER
model.add(Lambda(lambda x:((x/255.0)-0.5),input_shape = (160,320,1)))
#DATA IS CROPPED TO REMOVE UNNECESSARY PARTS
model.add(Cropping2D(cropping=((70,25),(0,0))))
#CONVOLUTION: FILTER-5X5 DEPTH-6 ACTIVATION-RELU
model.add(Convolution2D(6,5,5,activation='relu'))
#MAX POOLING: FILTER-2X2
model.add(MaxPooling2D((2,2)))
#CONVOLUTION: FILTER-5X5 DEPTH-18 ACTIVATION-RELU
model.add(Convolution2D(18,5,5,activation='relu'))
#MAX POOLING: FILTER-2X2
model.add(MaxPooling2D((2,2)))
#CONVOLUTION: FILTER-3X3 DEPTH-36 ACTIVATION-RELUConvolution layer with filter of 3X3 and depth of 36 with a relu activation
model.add(Convolution2D(36,3,3,activation='relu'))
#MAX POOLING: FILTER-2X2
model.add(MaxPooling2D((2,2)))
#CONVOLUTION: FILTER-3X3 DEPTH-4 ACTIVATION-RELUConvolution layer with filter of 3X3 and depth of 4 with a relu activation
model.add(Convolution2D(48,3,3,activation='relu'))
#MAX POOLING: FILTER-2X2
model.add(MaxPooling2D((2,2)))
#DROPOUT OT REDUCE OVERFITTING
model.add(Dropout(0.5))
#FLATTEN
model.add(Flatten())
#OUTPUT->512
model.add(Dense(512))
model.add(Activation('relu'))
#OUTPUT->256
model.add(Dense(256))
model.add(Activation('relu'))
#OUTPUT->128
model.add(Dense(128))
model.add(Activation('relu'))
#OUTPUT->32
model.add(Dense(32))
model.add(Activation('relu'))
#OUTPUT->STEERING ANGLE
model.add(Dense(1))


print(model.summary())



#ADAM OPTIMIZER
model.compile(loss ='mse',optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch=3)
model.save('model.h5')
