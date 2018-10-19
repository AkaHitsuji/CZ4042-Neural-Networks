#
# Project 1, starter code part b
#

import tensorflow as tf
import numpy as np
import pylab as plt


NUM_FEATURES = 8

learning_rate = 10**-7
ridge_param = 10**-3
epochs = 500
batch_size = 32
num_neurons = 30
seed = 10
np.random.seed(seed)

#read and divide data into test and train sets 
cal_housing = np.loadtxt('../provided files/cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]

m = 3* X_data.shape[0] // 10
trainX, trainY = X_data[m:], Y_data[m:]

trainX = (trainX- np.mean(trainX, axis=0))/ np.std(trainX, axis=0)

# experiment with small datasets
testX = trainX[700:1000]
testY = trainY[700:1000]
trainX = trainX[:700]
trainY = trainY[:700]

n = trainX.shape[0]

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, 1])

# Build the graph for the deep net
weights_h = tf.Variable(tf.truncated_normal([NUM_FEATURES,num_neurons], stddev=0.001)) 
biases_h = tf.Variable(tf.zeros([num_neurons]))
h = tf.nn.relu(tf.matmul(x, weights_h) + biases_h)

weights = tf.Variable(tf.truncated_normal([num_neurons, 1], stddev=1.0 / np.sqrt(NUM_FEATURES), dtype=tf.float32), name='weights')
biases = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases')
y = tf.matmul(h, weights) + biases

ridge_loss = tf.square(y_ - y)
regularization = tf.nn.l2_loss(weights) + tf.nn.l2_loss(weights_h)
loss = tf.reduce_mean(ridge_loss + ridge_param*regularization)

#Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)
error = tf.reduce_mean(tf.square(y_ - y))

test_errs = []
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(epochs):
		# implementing mini-batch GD
		for s in range(0, n-batch_size, batch_size):
			train_op.run(feed_dict={x: trainX[s:s+batch_size], y_: trainY[s:s+batch_size]})
		test_err = error.eval(feed_dict={x: testX, y_:testY})
		test_errs.append(test_err)
		if i % 100 == 0:
			print('iter %d: test error %g'%(i, test_errs[i]))

# plot learning curves
fig = plt.figure(1)
plt.xlabel('number of iterations')
plt.ylabel('Test Error')
plt.plot(range(epochs), test_errs)
plt.show()