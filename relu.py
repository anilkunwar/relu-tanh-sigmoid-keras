# Load dependencies
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Load data
dataset = np.loadtxt('./pima_dataset.csv', delimiter=',')

# Separate input data and target data
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Create the Perceptron
model = Sequential()

# Model layers
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Model compilation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training
model.fit(X, Y, epochs=225, batch_size=25, verbose=1, validation_split=0.2)