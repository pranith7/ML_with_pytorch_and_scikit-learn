#!/usr/bin/env python
# coding: utf-8

# # Chapter 11 - Implementing a Multi-layer Artificial Neural Network from Scratch

# ### The topics that we will cover in this chapter are as follows:
# 
#     •Gaining a conceptual understanding of multilayer NNs
#     •Implementing the fundamental backpropagation algorithm for NN training from scratch
#     •Training a basic multilayer NN for image classification

# In[2]:


get_ipython().system('pip install -q pandas scikit-learn matplotlib')


# ### Overview

# - [Modeling complex functions with artificial neural networks](#Modeling-complex-functions-with-artificial-neural-networks)
#   - [Single-layer neural network recap](#Single-layer-neural-network-recap)
#   - [Introducing the multi-layer neural network architecture](#Introducing-the-multi-layer-neural-network-architecture)
#   - [Activating a neural network via forward propagation](#Activating-a-neural-network-via-forward-propagation)
# - [Classifying handwritten digits](#Classifying-handwritten-digits)
#   - [Obtaining the MNIST dataset](#Obtaining-the-MNIST-dataset)
#   - [Implementing a multi-layer perceptron](#Implementing-a-multi-layer-perceptron)
#   - [Coding the neural network training loop](#Coding-the-neural-network-training-loop)
#   - [Evaluating the neural network performance](#Evaluating-the-neural-network-performance)
# - [Training an artificial neural network](#Training-an-artificial-neural-network)
#   - [Computing the loss function](#Computing-the-loss-function)
#   - [Developing your intuition for backpropagation](#Developing-your-intuition-for-backpropagation)
#   - [Training neural networks via backpropagation](#Training-neural-networks-via-backpropagation)
# - [Convergence in neural networks](#Convergence-in-neural-networks)
# - [Summary](#Summary)

# In[3]:


from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# # Modeling complex functions with artificial neural networks

# ...

# ## Single-layer neural network recap

# In[7]:


Image(filename='figures/11_01.png', width=800) 


# ## Introducing the multi-layer neural network architecture

# In[9]:


Image(filename='figures/11_02.png', width=900) 


# In[11]:


Image(filename='figures/11_03.png', width=600)


# ## Activating a neural network via forward propagation
# 

# The MNIST dataset is publicly available at http://yann.lecun.com/exdb/mnist/ and consists of the following four parts:
# 
# - Training set images: train-images-idx3-ubyte.gz (9.9 MB, 47 MB unzipped, 60,000 examples)
# - Training set labels: train-labels-idx1-ubyte.gz (29 KB, 60 KB unzipped, 60,000 labels)
# - Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 7.8 MB, 10,000 examples)
# - Test set labels: t10k-labels-idx1-ubyte.gz (5 KB, 10 KB unzipped, 10,000 labels)
# 
# 

# ### Classifying handwritten digits

# In[4]:


import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1,return_X_y=True)
X = X.values
y = y.astype(int).values


# In scikit-learn, the fetch_openml function downloads the MNIST dataset from OpenML (https://www.
# openml.org/d/554) as pandas DataFrame and Series objects, which is why we use the .values attribute to
# obtain the underlying NumPy arrays. (If you are using a scikit-learn version older than 1.0, fetch_openml
# downloads NumPy arrays directly so you can omit using the .values attribute.) The n×m dimensional
# X array consists of 70,000 images with 784 pixels each, and the y array stores the corresponding 70,000
# class labels, which we can confirm by checking the dimensions of the arrays as follows:

# In[8]:


print(X.shape)
print(y.shape)


# The images in the MNIST dataset consist of 28×28 pixels, and each pixel is represented by a grayscale
# intensity value. Here, fetch_openml already unrolled the 28×28 pixels into one-dimensional row
# vectors, which represent the rows in our X array (784 per row or image) above. The second array (y)
# returned by the fetch_openml function contains the corresponding target variable, the class labels
# (integers 0-9) of the handwritten digits.
# Next, let’s normalize the pixels values in MNIST to the range –1 to 1 (originally 0 to 255) via the fol-
# lowing code line:

# Normalize to [-1, 1] range:

# In[12]:


X = ((X/255.) - .5)*2


# The reason behind this is that gradient-based optimization is much more stable under these conditions,
# as discussed in Chapter 2. Note that we scaled the images on a pixel-by-pixel basis, which is different
# from the feature-scaling approach that we took in previous chapters.
# 
# Previously, we derived scaling parameters from the training dataset and used these to scale each
# column in the training dataset and test dataset. However, when working with image pixels, centering
# them at zero and rescaling them to a [–1, 1] range is also common and usually works well in practice.
# 
# 
# 

# To get an idea of how those images in MNIST look, let’s visualize examples of the digits 0-9 after re-
# shaping the 784-pixel vectors from our feature matrix into the original 28×28 image that we can plot
# via Matplotlib’s imshow function:

# In[31]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X[y == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
#plt.savefig('figures/11_4.png', dpi=300)
plt.show()


# Visualize 25 different versions of "7":

# In[32]:


fig,ax = plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
ax = ax.flatten()
for i in range(10):
    
    img = X[y==7][0].reshape(28,28)
    ax[i].imshow(img,cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


# Finally, let’s divide the dataset into training, validation, and test subsets. The following code will split the dataset such that 55,000 images are used for training, 5,000 images for validation, and 10,000
# images for testing:

# In[15]:


from sklearn.model_selection import train_test_split


X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=10000, random_state=123, stratify=y)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)


# optional to free up some memory by deleting non-used arrays:
del X_temp, y_temp, X, y


# ## Implementing a multilayer perceptron

# In[12]:


##########################
### MODEL
##########################

def sigmoid(z):                                        
    return 1. / (1. + np.exp(-z))


def int_to_onehot(y, num_labels):

    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1

    return ary


class NeuralNetMLP:

    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()
        
        self.num_classes = num_classes
        
        # hidden
        rng = np.random.RandomState(random_seed)
        
        self.weight_h = rng.normal(
            loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)
        
        # output
        self.weight_out = rng.normal(
            loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)
        
    def forward(self, x):
        # Hidden layer
        # input dim: [n_examples, n_features] dot [n_hidden, n_features].T
        # output dim: [n_examples, n_hidden]
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # Output layer
        # input dim: [n_examples, n_hidden] dot [n_classes, n_hidden].T
        # output dim: [n_examples, n_classes]
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):  
    
        #########################
        ### Output layer weights
        #########################
        
        # onehot encoding
        y_onehot = int_to_onehot(y, self.num_classes)

        # Part 1: dLoss/dOutWeights
        ## = dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeight
        ## where DeltaOut = dLoss/dOutAct * dOutAct/dOutNet
        ## for convenient re-use
        
        # input/output dim: [n_examples, n_classes]
        d_loss__d_a_out = 2.*(a_out - y_onehot) / y.shape[0]

        # input/output dim: [n_examples, n_classes]
        d_a_out__d_z_out = a_out * (1. - a_out) # sigmoid derivative

        # output dim: [n_examples, n_classes]
        delta_out = d_loss__d_a_out * d_a_out__d_z_out # "delta (rule) placeholder"

        # gradient for output weights
        
        # [n_examples, n_hidden]
        d_z_out__dw_out = a_h
        
        # input dim: [n_classes, n_examples] dot [n_examples, n_hidden]
        # output dim: [n_classes, n_hidden]
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)
        

        #################################        
        # Part 2: dLoss/dHiddenWeights
        ## = DeltaOut * dOutNet/dHiddenAct * dHiddenAct/dHiddenNet * dHiddenNet/dWeight
        
        # [n_classes, n_hidden]
        d_z_out__a_h = self.weight_out
        
        # output dim: [n_examples, n_hidden]
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)
        
        # [n_examples, n_hidden]
        d_a_h__d_z_h = a_h * (1. - a_h) # sigmoid derivative
        
        # [n_examples, n_features]
        d_z_h__d_w_h = x
        
        # output dim: [n_hidden, n_features]
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__dw_out, d_loss__db_out, 
                d_loss__d_w_h, d_loss__d_b_h)


# In[13]:


model = NeuralNetMLP(num_features=28*28,num_hidden=50,num_classes=10)


# ### Coding the neural network training loop
# 
# Defining data loaders:

# In[16]:


import numpy as np

num_epochs = 50
minibatch_size = 100


def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - minibatch_size 
                           + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        
        yield X[batch_idx], y[batch_idx]

        
# iterate over training epochs
for i in range(num_epochs):

    # iterate over minibatches
    minibatch_gen = minibatch_generator(
        X_train, y_train, minibatch_size)
    
    for X_train_mini, y_train_mini in minibatch_gen:

        break
        
    break
    
print(X_train_mini.shape)
print(y_train_mini.shape)


# ### Defining a function to compute the loss and accuracy

# In[17]:


def mse_loss(targets,probas,num_labels=10):
    onehot_targets = int_to_onehot(
        targets,num_labels=num_labels
    )
    return np.mean((onehot_targets - probas)**2)

def accuracy(targets,predicted_labels):
    return np.mean(predicted_labels == targets)

_,probas = model.forward(X_valid)
mse = mse_loss(y_valid,probas)
print(f'Initial validation MSE: {mse:.1f}')

predicted_labels = np.argmax(probas,axis=1)
acc = accuracy(y_valid,predicted_labels)
print(f'Initial validation accuracy: {acc*100:.1f}%')


# In this code example, note that model.forward() returns the hidden and output layer activations.
# Remember that we have 10 output nodes (one corresponding to each unique class label). Hence,
# when computing the MSE, we first converted the class labels into one-hot encoded class labels in
# the mse_loss() function. In practice, it does not make a difference whether we average over the row
# or the columns of the squared-difference matrix first, so we simply call np.mean() without any axis
# specification so that it returns a scalar.

# In[18]:


def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
        
    for i, (features, targets) in enumerate(minibatch_gen):

        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas)**2)
        correct_pred += (predicted_labels == targets).sum()
        
        num_examples += targets.shape[0]
        mse += loss

    mse = mse/i
    acc = correct_pred/num_examples
    return mse, acc

mse, acc = compute_mse_and_acc(model, X_valid, y_valid)
print(f'Initial valid MSE: {mse:.1f}')
print(f'Initial valid accuracy: {acc*100:.1f}%')


# In[19]:


def train(model, X_train, y_train, X_valid, y_valid, num_epochs,
          learning_rate=0.1):
    
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    
    for e in range(num_epochs):

        # iterate over minibatches
        minibatch_gen = minibatch_generator(
            X_train, y_train, minibatch_size)

        for X_train_mini, y_train_mini in minibatch_gen:
            
            #### Compute outputs ####
            a_h, a_out = model.forward(X_train_mini)

            #### Compute gradients ####
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = \
                model.backward(X_train_mini, a_h, a_out, y_train_mini)

            #### Update weights ####
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out
        
        #### Epoch Logging ####        
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
        train_acc, valid_acc = train_acc*100, valid_acc*100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        print(f'Epoch: {e+1:03d}/{num_epochs:03d} '
              f'| Train MSE: {train_mse:.2f} '
              f'| Train Acc: {train_acc:.2f}% '
              f'| Valid Acc: {valid_acc:.2f}%')

    return epoch_loss, epoch_train_acc, epoch_valid_acc


# In[20]:


np.random.seed(123) # for the training set shuffling

epoch_loss, epoch_train_acc, epoch_valid_acc = train(
    model, X_train, y_train, X_valid, y_valid,
    
    num_epochs=50, learning_rate=0.1)


# ### Evaluating the neural network performance

# In[23]:


import matplotlib.pyplot as plt 
plt.plot(range(len(epoch_loss)), epoch_loss)
plt.ylabel('Mean squared error')
plt.xlabel('Epoch')
#plt.savefig('figures/11_07.png', dpi=300)
plt.show()


# In[24]:


plt.plot(range(len(epoch_train_acc)), epoch_train_acc,
         label='Training')
plt.plot(range(len(epoch_valid_acc)), epoch_valid_acc,
         label='Validation')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc='lower right')
#plt.savefig('figures/11_08.png', dpi=300)
plt.show()


# In[25]:


test_mse, test_acc = compute_mse_and_acc(model, X_test, y_test)
print(f'Test accuracy: {test_acc*100:.2f}%')


# Plot failure cases:

# In[26]:


X_test_subset = X_test[:1000, :]
y_test_subset = y_test[:1000]

_, probas = model.forward(X_test_subset)
test_pred = np.argmax(probas, axis=1)

misclassified_images = X_test_subset[y_test_subset != test_pred][:25]
misclassified_labels = test_pred[y_test_subset != test_pred][:25]
correct_labels = y_test_subset[y_test_subset != test_pred][:25]


# In[27]:


fig, ax = plt.subplots(nrows=5, ncols=5, 
                       sharex=True, sharey=True, figsize=(8, 8))
ax = ax.flatten()
for i in range(25):
    img = misclassified_images[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title(f'{i+1}) '
                    f'True: {correct_labels[i]}\n'
                    f' Predicted: {misclassified_labels[i]}')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
#plt.savefig('figures/11_09.png', dpi=300)
plt.show()


# Training an artificial neural network
# ...
# 
# Computing the loss function

# In[29]:


Image(filename='figures/11_10.png', width=600) 


# Developing your intuition for backpropagation
# ...
# 
# Training neural networks via backpropagation

# In[31]:


Image(filename='./figures/11_11.png', width=700) 


# In[36]:


Image(filename='figures/11_12.png', width=800,height=500) 


# In[37]:


Image(filename='figures/11_13.png', width=900) 


# # Convergence in neural networks

# In[39]:


Image(filename='figures/11_14.png', width=700) 


# In[ ]:


get_ipython().system('jupyter nbconvert --to script  chapter_11.ipynb --output ch10')

