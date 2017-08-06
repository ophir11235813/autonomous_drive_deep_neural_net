import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
 
lines = []

with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)


train_samples, validation_samples = train_test_split(lines, test_size = 0.2)

def generator(samples,batch_size = 32):
	num_samples = len(samples)
	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0,num_samples, batch_size):
			batch_samples = samples[offset:offset + batch_size]
         
			images = []
			measurements = []

		for batch_sample in batch_samples:
			center_angle = float(batch_sample[3])
			center_image = cv2.imread("./data/IMG/" + batch_sample[0].split('/')[-1])
			
			if center_image != None:
				images.append(center_image)
				measurements.append(center_angle)
            
            
			left_image = cv2.imread("./data/IMG/" + batch_sample[1].split('/')[-1])
			if left_image != None:
				images.append(left_image)
				measurements.append(center_angle + 0.2)
			
				
			right_image = cv2.imread("./data/IMG/" + batch_sample[2].split('/')[-1])
			if right_image != None:
				images.append(right_image)
				measurements.append(center_angle - 0.3)


		augmented_images, augmented_measurements = [], []
		for image, measurement in zip(images, measurements):
			augmented_images.append(image)
			augmented_images.append(cv2.flip(image,1))
			augmented_measurements.append(measurement)
			augmented_measurements.append(measurement*-1.0)

		X_train = np.array(augmented_images)
		y_train = np.array(augmented_measurements)

		yield sklearn.utils.shuffle(X_train,y_train)

train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)


from keras.models import Sequential
from keras.layers import Flatten, Activation, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
#model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping = ((50,30),(0,0)), input_shape = (160,320,3)))
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = (80,320,3), output_shape = (80,320,3)))
model.add(Convolution2D(24,5,5,subsample = (2,2), border_mode = 'valid', activation = "relu"))
model.add(Convolution2D(36,5,5,subsample = (2,2), activation="relu"))
model.add(Convolution2D(48,5,5, activation = "relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))



model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = 3)

# model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, nb_epoch = 3)

model.save('model.h5')


