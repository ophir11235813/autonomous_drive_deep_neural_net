import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

# STEP 1: Open the csv file that holds paths to images (captured from car sensors)
# and the corresponding angle of the steering wheel
lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# STEP 2: Split into training and validation sets (80% / 20%) 
train_samples, validation_samples = train_test_split(lines, test_size = 0.2)

# STEP 3: Define a generator function, to handle large sets of images in small batches
def generator(samples,batch_size = 32):
	num_samples = len(samples)
	
	while 1: # generators need to run while always true
		
		sklearn.utils.shuffle(samples)
		for offset in range(0,num_samples, batch_size):
			batch_samples = samples[offset:offset + batch_size]
         	
         	# Initialize lists of images. These WON'T be reset each time the generator is called
			images = []
			measurements = []

		for batch_sample in batch_samples:
			center_angle = float(batch_sample[3]) # Steering angle
			center_image = cv2.imread("./data/IMG/" + batch_sample[0].split('/')[-1])
			
			if center_image != None:
				# Ensure image isn't empty and add images/angles to training/validation set
				images.append(center_image)
				measurements.append(center_angle)
            
            
			left_image = cv2.imread("./data/IMG/" + batch_sample[1].split('/')[-1])
			if left_image != None:
				images.append(left_image)
				measurements.append(center_angle + 0.2) # Picture captured from left side of car, 
														# so ADD angle to steering wheel as a correction
			
				
			right_image = cv2.imread("./data/IMG/" + batch_sample[2].split('/')[-1])
			if right_image != None:
				images.append(right_image)
				measurements.append(center_angle - 0.3) # Picture captured from right side of car, 
														# so SUBTRACT angle from steering wheel as correction

		# The dataset has a left-turn bias. We can generate new images by flipping the 
		# images from left-to-right (and flip steering angles accordingly)
		augmented_images, augmented_measurements = [], []
		for image, measurement in zip(images, measurements):
			augmented_images.append(image)
			augmented_images.append(cv2.flip(image,1))
			augmented_measurements.append(measurement)
			augmented_measurements.append(measurement*-1.0)

		# Convert to float arrays. Note that this is step that substantially increases size
		X_train = np.array(augmented_images)
		y_train = np.array(augmented_measurements)

		yield sklearn.utils.shuffle(X_train,y_train) 

# STEP 4: Generate a training and validation set in batches of 32. 
train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)

# STEP 5: Set up deep neural net (DNN), using Keras libraries 
from keras.models import Sequential
from keras.layers import Flatten, Activation, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# STEP 6: Build the DNN, based on Nvidia's end-to-end architecture
model = Sequential()

# We only need PART of the image (e.g. the sky doesn't give much information), so crop the image and reduce
# file size 
model.add(Cropping2D(cropping = ((50,30),(0,0)), input_shape = (160,320,3)))

# Next normalize so that all values are between [-0.5,0.5]
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = (80,320,3), output_shape = (80,320,3)))

# Build the network as per Nvidia model
model.add(Convolution2D(24,5,5,subsample = (2,2), border_mode = 'valid', activation = "relu"))
model.add(Convolution2D(36,5,5,subsample = (2,2), activation="relu"))
model.add(Convolution2D(48,5,5, activation = "relu"))
model.add(Convolution2D(64,3,3, activation="relu"))

# Four fully connected layers
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# STEP 7: Compile the model using the ADAM optimizer. We use a mean square error as the dependent variable 
# here is a SCALAR steering angle
model.compile(loss = 'mse', optimizer = 'adam')

# STEP 8: Run the model over three epochs (more than that will begin overfitting)
model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = 3)

# STEP 9: Save the model
model.save('model.h5')


