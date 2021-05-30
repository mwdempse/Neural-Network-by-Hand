"""
Two Layer Neural Network with Keras

"""

from keras.layers import Dense, Activation
from keras.models import Sequential
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
import matplotlib.pyplot as plt

# Generate a dataset and plot it
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.15)
y = y.reshape(200,1)

# Create model
model = Sequential()
model.add(Dense(3,input_dim=2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#Model summary
model.summary()

# compile model
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['acc'])

# Train Model
history = model.fit(X,y,epochs=900)





