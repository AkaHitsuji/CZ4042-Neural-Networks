#
# Project 1, Question 1A, Part 1
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt


# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

# input attributes of 4 spectral bands x 9 pixels in each neighbourhood
NUM_FEATURES = 36
# 6 class labels
NUM_CLASSES = 6

learning_rate = 0.01
epochs = 1000
batch_size = 32
num_neurons = 10
seed = 10
np.random.seed(seed)

#read train data
train_input = np.loadtxt('../provided files/sat_train.txt',delimiter=' ')
trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))
train_Y[train_Y == 7] = 6

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix


# experiment with small datasets
trainX = trainX[:1000]
trainY = trainY[:1000]

n = trainX.shape[0]


# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])


# Build the graph for the deep net
# input layer has NUM_FEATURES nodes, hidden layer has num_neurons nodes, output softmax layer has NUM_CLASSES nodes
# first set of variables for weights and bias btwn input layer and hidden layer
weights_h = tf.Variable(tf.truncated_normal([NUM_FEATURES,num_neurons], stddev=0.001))
biases_h = tf.Variable(tf.zeros([num_neurons]))

# 2nd set of variables for weights and bias btwn hidden layer and output layer
weights = tf.Variable(tf.truncated_normal([num_neurons, NUM_CLASSES], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weights')
biases  = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')

# creating the neural net graph
# tf.matmul links 2 tensors to create a matrix multiplication tensor.
# logits are the vector of raw (non-normalized) predictions that a classification model generates, which is passed to a normalization function, in this case the softmax function
h = tf.nn.relu(tf.matmul(x, weights_h) + biases_h)
logits = tf.matmul(h, weights) + biases

ridge_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
ridge_param = tf.constant(0.000001)
regularization = tf.nn.l2_loss(weights) + tf.nn.l2_loss(weights_h)
loss = tf.reduce_mean(ridge_loss + ridge_param*regularization)

# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

# input and output layers are fed to the tf.placeholder tensors and weights are represented as tf.Variable as their value changes for each iteration
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_acc = []
    for i in range(epochs):
        train_op.run(feed_dict={x: trainX, y_: trainY})
        train_acc.append(accuracy.eval(feed_dict={x: trainX, y_: trainY}))

        if i % 100 == 0:
            print('iter %d: accuracy %g'%(i, train_acc[i]))


# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train accuracy')
plt.show()

# questions:
# - how is epoch determined?
# - what is seed?
# - why use tf.nn.relu for first layer
