"""
Two Layer Neural Network by Hand

"""

# math
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

# plots
import matplotlib
import matplotlib.pyplot as plt

#%matplotlib inline
#matplotlib.RcParams['figure.figsize'] = (10.0,8.0)

# sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Loss function
def bce_loss(y,y_hat):
  minval = 0.000000000001
  N = y.shape[0]
  loss = -1/N * np.sum(y*np.log(y_hat.clip(min = minval)) + (1 - y)*np.log(1-y_hat.clip(min = minval)))
  return loss 

# log loss derivative
def bce_loss_deriv(y,y_hat):
    return (y_hat-y)

# forward propigation
def forward_prop(model,a0):
    # params from model
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # linear step 1
    z1 = a0.dot(W1) + b1
    #activation 1
    a1 = np.tanh(z1)
    # linear step 2
    z2 = a1.dot(W2) + b2
    # activation 2
    a2 = sigmoid(z2)
    
    cache = {'a0':a0,'z1':z1,'a1':a1,'z1':z1,'a2':a2}
    return cache

# Hyperbolic tangent function derivative
def tanh_deriv(x):
    return (1-np.power(x,2))

# backward propigation
def back_prop(model,cache,y):
    # params from model
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # params from forward propigation
    a0,a1, a2 = cache['a0'],cache['a1'],cache['a2']
    
    # loss deriv respect to output
    dz2 = bce_loss_deriv(y=y,y_hat=a2)
    # loss deriv respect to 2nd layer weights
    dW2 = (a1.T).dot(dz2)
    # loss deriv respect to 2nd layer bias
    db2 = np.sum(dz2, axis=0, keepdims=True)
    # loss deriv respect to 1st layer
    dz1 = dz2.dot(W2.T)*tanh_deriv(a1)
    # loss deriv respect to 1st layer weights
    dW1 = np.dot(a0.T,dz1)
    # loss deriv respect to 1st layer bias
    db1 = np.sum(dz1, axis=0)
    
    # Gradients
    grads = {'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1}
    return grads

# helper function for ploting decision boundry
def plot_dec_boun(pred_func):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.Spectral)
    
    
# test dataset for plot
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.15)
y = y.reshape(200,1)
plt.scatter(X[:,0], X[:,1], s=40, c=y.flatten(), cmap=plt.cm.Spectral)

# predict function
def predict(model, x):
    fp = forward_prop(model,x)
    y_hat = fp['a2']
    
    y_hat[y_hat<0.5] = 1
    y_hat[y_hat<0.5] = 0
    return y_hat

# accuracy function
def model_accuracy(model,x,y):
    m = y.shape[0]
    pred = predict(model,x)
    pred = pred.reshape(y.shape)
    error = np.sum(np.abs(pred-y))
    return (m-error)/(m*100)

# initialize parameters
def intial_param(nn_input_dim,nn_hdim,nn_output_dim):
    # first layer weights
    W1 = 2*np.random.randn(nn_input_dim,nn_hdim)-1
    #first layer bias
    b1 = np.zeros((1,nn_hdim))
    # second layer weights
    W2 = 2*np.random.randn(nn_hdim,nn_output_dim)-1
    # second layer bias
    b2 = np.zeros((1,nn_output_dim))
    
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model

# update parameters with gradient
def update_param(model, grads, learning_rate):
    # params
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # update param
    W1 -= learning_rate * grads['dW1']
    b1 -= learning_rate * grads['db1']
    W2 -= learning_rate * grads['dW2']
    b2 -= learning_rate * grads['db2']
    #update model
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model   
    
# training model
def train(model, X_,y_,learning_rate,num_passes=20000,print_loss = False):
    #gradient descent
    for i in range(0,num_passes):
        # forward propigation
        cache = forward_prop(model,X_)
        # backward propigation
        grads = back_prop(model,cache,y)
        #update param
        model = update_param(model=model,grads=grads,learning_rate=learning_rate)
        
        # print loss?
        if print_loss and i % 100 == 0:
            y_hat = cache['a2']
            print('Loss after iteration', i, ':', bce_loss(y,y_hat))
            print('Accuracy after iteration',i,':',model_accuracy(model,X_,y_),'%')
    return(model)
  
    
# hyper parameters
hiden_layer_size = 3
learning_rate = 0.01

# Initialize the parameters to random values.
np.random.seed(0)
model = intial_param(nn_input_dim=2, nn_hdim= hiden_layer_size, nn_output_dim= 1)
model = train(model,X,y,learning_rate=learning_rate,num_passes=1000,print_loss=True)

# decision boundary
plot_dec_boun(lambda x: predict(model,x))
plt.title("Decision Boundary for hidden layer size 3")

# Now with more noise
# Generate a dataset and plot it
np.random.seed(0)
# The data generator alows us to regulate the noise level
X, y = sklearn.datasets.make_moons(200, noise=0.3)
y = y.reshape(200,1)
plt.scatter(X[:,0], X[:,1], s=40, c=y.flatten(), cmap=plt.cm.Spectral)

#too few layers
# Hyper parameters
hiden_layer_size = 1
# I picked this value because it showed good results in my experiments
learning_rate = 0.01

# Initialize the parameters to random values. We need to learn these.
np.random.seed(0)
# This is what we return at the end
model = intial_param(nn_input_dim=2, nn_hdim= hiden_layer_size, nn_output_dim= 1)
model = train(model,X,y,learning_rate=learning_rate,num_passes=1000,print_loss=True)

# Plot the decision boundary
plot_dec_boun(lambda x: predict(model,x))
plt.title("Decision Boundary for hidden layer size 1")

# toio may layers
# Hyper parameters
hiden_layer_size = 500
# I picked this value because it showed good results in my experiments
learning_rate = 0.01

# Initialize the parameters to random values. We need to learn these.
np.random.seed(0)
# This is what we return at the end
model = intial_param(nn_input_dim=2, nn_hdim= hiden_layer_size, nn_output_dim= 1)
model = train(model,X,y,learning_rate=learning_rate,num_passes=1000,print_loss=True)

# Plot the decision boundary
# This might take a little while as our model is very big now
plot_dec_boun(lambda x: predict(model,x))
plt.title("Decision Boundary for hidden layer size 500")








    
    
    
    
    