#
# Project 1, starter code part b
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt

# initialization routines for bias and weights
def init_bias(test_size = 1):
    return(tf.Variable(np.zeros(test_size), dtype=tf.float32))

def init_weights(n_in=1, n_out=1):
    return (tf.Variable(tf.truncated_normal([n_in, n_out], stddev=1.0/math.sqrt(float(n_in))), name='weights'))

# scale data
def scale(X):
    return (X - np.mean(X, axis=0))/np.std(X, axis=0)

NUM_FEATURES = 8
learning_rate = 0.5*(10**-6)
beta = 10**-3
epochs = 1000
batch_size = 32
num_neurons = 30
seed = 10
np.random.seed(seed)

#read and divide data into test and train sets
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
X_data = scale(X_data)
Y_data = (np.asmatrix(Y_data)).transpose()

idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]

test_size = 3* X_data.shape[0] // 10
train_size = X_data.shape[0]-test_size
trainX, trainY = X_data[test_size:], Y_data[test_size:]
testX, testY = X_data[:test_size], Y_data[:test_size]

# # experiment with small datasets
# trainX = trainX[:1000]
# trainY = trainY[:1000]
# train_size = trainX.shape[0]

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, 1])

# Build the graph for the deep net
V = init_weights(NUM_FEATURES, num_neurons)
c = init_bias(num_neurons)
W = init_weights(num_neurons)
b = init_bias()
h = tf.nn.relu(tf.matmul(x, V) + c)
y = tf.matmul(h, W) + b

square_error = tf.reduce_mean(tf.square(y_ - y))
regularization = tf.nn.l2_loss(V) + tf.nn.l2_loss(W)
loss = tf.reduce_mean(square_error + beta *regularization)

#Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)
error = tf.reduce_mean(tf.square(y_ - y))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	test_err = []
	for i in range(epochs):
		rand_array = np.arange(train_size)
		np.random.shuffle(rand_array)
		trainX = trainX[rand_array]
		trainY = trainY[rand_array]
		# implementing mini-batch GD
		for start in range(0, train_size-batch_size, batch_size):
			train_op.run(feed_dict={x: trainX[start:start+batch_size], y_: trainY[start:start+batch_size]})
		test_err.append(error.eval(feed_dict={x: testX, y_: testY}))
		
		if i % 100 == 0:
			print('iter %d: validation error %g'%(i, test_err[i]))

# plot learning curves

plt.figure(1)
plt.plot(range(epochs), test_err)
plt.xlabel('Epochs')
plt.ylabel('Test Data Error')
plt.savefig('./figures/B2b_Fig1.png')

plt.show()
