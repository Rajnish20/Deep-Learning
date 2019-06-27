# Convolutional Neural Network

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Convolution layer 
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = "relu"))

# Pooling 
classifier.add(MaxPooling2D(pool_size = (2, 2)))