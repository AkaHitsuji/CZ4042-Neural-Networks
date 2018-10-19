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
num_neurons = 100
num_neurons_fixed = 20
seed = 10
np.random.seed(seed)
dropout = 0.9

#read and divide data into test and train sets
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()
X_data = scale(X_data)
Y_data = Y_data/1000

idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]

test_size = 3* X_data.shape[0] // 10
train_size = X_data.shape[0]-test_size
trainX, trainY = X_data[test_size:], Y_data[test_size:]
testX, testY = X_data[:test_size], Y_data[:test_size]

# experiment with small datasets
# trainX = trainX[:1000]
# trainY = trainY[:1000]
# train_size = trainX.shape[0]

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, 1])

# Build the graph for the deep net
V = init_weights(NUM_FEATURES, num_neurons)
c = init_bias(num_neurons)
V2 = init_weights(num_neurons, num_neurons_fixed)
c2 = init_bias(num_neurons_fixed)
V3 = init_weights(num_neurons_fixed, num_neurons_fixed)
c3 = init_bias(num_neurons_fixed)
W = init_weights(num_neurons_fixed)
b = init_bias()
h = tf.nn.relu(tf.matmul(x, V) + c)
h = tf.nn.dropout(h, dropout)
h2 = tf.nn.relu(tf.matmul(h, V2) + c2)
h2 = tf.nn.dropout(h2, dropout)
h3 = tf.nn.relu(tf.matmul(h2, V3) + c3)
h3 = tf.nn.dropout(h3, dropout)
y = tf.matmul(h3, W) + b

square_error = (tf.square(y_ - y))
regularization = tf.nn.l2_loss(V) + tf.nn.l2_loss(W) + tf.nn.l2_loss(V2) + tf.nn.l2_loss(V3)
loss = tf.reduce_mean(square_error + beta *regularization)

#Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)
error = tf.reduce_mean(tf.square(y_ - y))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	test_errs = []
	for i in range(epochs):
		rand_array = np.arange(train_size)
		np.random.shuffle(rand_array)
		trainX = trainX[rand_array]
		trainY = trainY[rand_array]
		# implementing mini-batch GD
		for start in range(0, train_size-batch_size, batch_size):
			train_op.run(feed_dict={x: trainX[start:start+batch_size], y_: trainY[start:start+batch_size]})
		
		test_err = error.eval(feed_dict={x: testX, y_:testY})
		test_errs.append(test_err)
		if i % 100 == 0:
			print('iter %d: validation error %g'%(i, test_errs[i]))
# plot learning curves
fig = plt.figure(1)
plt.title("5 layers With Dropout")
plt.xlabel('number of iterations')
plt.ylabel('Test Error')
# plt.yscale('log')
plt.plot(range(epochs), test_errs)
plt.show()

