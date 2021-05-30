"""
One Layer Neural Network: Logistic Regressor

"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Set up
X = np.array([[0,1,0],
              [1,0,0],
              [1,1,1],
              [0,1,1]])

y = np.array([[0,1,1,0]]).T

# sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Loss function
def bce_loss(y,y_hat):
  N = y.shape[0]
  loss = -1/N * np.sum(y*np.log(y_hat) + (1 - y)*np.log(1-y_hat))
  return loss 

# Initalize model
W = 2*np.random.random((3,1))-1
b=0

# hyper parameters
alpha = 1 #learning rate/step size
epochs = 20 # number of times to run the training process

N = y.shape[0] #number of samples
losses = [] #loss over time

for i in range(epochs):
    # Forward pass
    z = X.dot(W) + b
    A = sigmoid(z)
    
    # Calculate Loss
    loss = bce_loss(y,A)
    print('Epoch:',i,'Loss',round(loss,3)) # rounded for readability
    losses.append(loss)
    
    # Calculate Derivatives
    dz = (A-y)
    dW = 1/N * np.dot(X.T,dz)
    db = 1/N * np.sum(dz,axis=0,keepdims=True)
    
    # Parameter updates
    W -= alpha * dW
    b -= alpha * db

plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss rate improves over time')
plt.show()
