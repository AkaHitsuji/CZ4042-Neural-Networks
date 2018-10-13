#
# Project 1, starter code part b
#

import tensorflow as tf
import numpy as np
import pylab as plt

NUM_FEATURES = 8

learning_rate = 10**-9
ridge_param = 10**-3
epochs = 500
batch_size = 32
num_neuron = 80
seed = 10
np.random.seed(seed)
validation_split = 0.7
dropout = 0.9

#read and divide data into test and train sets 
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]

m = 3* X_data.shape[0] // 10
trainX, trainY = X_data[m:], Y_data[m:]

trainX = (trainX- np.mean(trainX, axis=0))/ np.std(trainX, axis=0)

# create validation and test sets randomly
trainX = trainX[:5000]
trainY = trainY[:5000]
n = trainX.shape[0]
test_size = (1-validation_split)*n
random = np.random.randint(0, test_size)
start, end = int(random), int(random+test_size)
testX = trainX[start:end]
testY = trainY[start:end]
trainX = np.append(trainX[:start], trainX[end:], axis=0)
trainY = np.append(trainY[:start], trainY[end:], axis=0)

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, 1])

# Build the graph for the deep net
weights_h1 = tf.Variable(tf.truncated_normal([NUM_FEATURES,num_neuron], stddev=0.001)) 
biases_h1 = tf.Variable(tf.zeros([num_neuron]))
h1 = tf.nn.relu(tf.matmul(x, weights_h1) + biases_h1)
h1 = tf.nn.dropout(h1, dropout)

weights_h2 = tf.Variable(tf.truncated_normal([num_neuron,20], stddev=0.001)) 
biases_h2 = tf.Variable(tf.zeros([20]))
h2 = tf.nn.relu(tf.matmul(h1, weights_h2) + biases_h2)
h2 = tf.nn.dropout(h2, dropout)

weights_h3 = tf.Variable(tf.truncated_normal([20,20], stddev=0.001)) 
biases_h3 = tf.Variable(tf.zeros([20]))
h3 = tf.nn.relu(tf.matmul(h2, weights_h3) + biases_h3)
h3 = tf.nn.dropout(h3, dropout)

weights = tf.Variable(tf.truncated_normal([20, 1], stddev=1.0 / np.sqrt(NUM_FEATURES), dtype=tf.float32), name='weights')
biases = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases')

# # 3 layer
# y3 = tf.matmul(h1, weights) + biases
# regularization3 = tf.nn.l2_loss(weights) + tf.nn.l2_loss(weights_h1)
# 4 layer
# y4 = tf.matmul(h2, weights) + biases
# regularization4 = tf.nn.l2_loss(weights) + tf.nn.l2_loss(weights_h1) + tf.nn.l2_loss(weights_h2)
# 5 layer
y5 = tf.matmul(h3, weights) + biases
regularization5 = tf.nn.l2_loss(weights) + tf.nn.l2_loss(weights_h1) + tf.nn.l2_loss(weights_h2) + tf.nn.l2_loss(weights_h3)

ridge_loss = tf.square(y_ - y5)
loss = tf.reduce_mean(ridge_loss + ridge_param*regularization5)

#Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)
error = tf.reduce_mean(tf.square(y_ - y5))

test_errs = []
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(epochs):
		# implementing mini-batch GD
		for s in range(0, n-batch_size, batch_size):
			train_op.run(feed_dict={x: trainX[s:s+batch_size], y_: trainY[s:s+batch_size]})
		test_err = error.eval(feed_dict={x: testX, y_:testY})
		test_errs.append(test_err)
		print("training iteration %d" %i, end="\r")

# plot learning curves
fig = plt.figure(1)
plt.title("5 layers With Dropout")
plt.xlabel('number of iterations')
plt.ylabel('Test Error (LOG)')
plt.plot(range(epochs), np.log10(test_errs))
plt.show()