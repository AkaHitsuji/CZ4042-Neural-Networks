#
# Project 1, starter code part b
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt

# initialization routines for bias and weights
def init_bias(n = 1):
    return(tf.Variable(np.zeros(n), dtype=tf.float32))

def init_weights(n_in=1, n_out=1):
    return (tf.Variable(tf.truncated_normal([n_in, n_out], stddev=1.0/math.sqrt(float(n_in))), name='weights'))

# scale data
def scale(X):
    return (X - np.mean(X, axis=0))/np.std(X, axis=0)


NUM_FEATURES = 8
learning_rates = [0.5*(10**-6), 10**-7, 0.5*(10**-8), 10**-9, 10**-10]
ridge_param = 10**-3
epochs = 500
batch_size = 32
num_neurons = 30
seed = 10
np.random.seed(seed)
num_folds = 5

#read and divide data into test and train sets
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]

m = 3* X_data.shape[0] // 10
trainX, trainY = X_data[m:], Y_data[m:]
trainX = scale(trainX)

# experiment with small datasets
trainX = trainX[:1000]
trainY = trainY[:1000]
n = trainX.shape[0]

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

ridge_loss = tf.square(y_ - y)
regularization = tf.nn.l2_loss(V) + tf.nn.l2_loss(W)
loss = tf.reduce_mean(ridge_loss + ridge_param*regularization)

errors = []
sizeof_fold = n//num_folds
for learning_rate in learning_rates:
	#Create the gradient descent optimizer with the given learning rate.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = optimizer.minimize(loss)
	error = tf.reduce_mean(tf.square(y_ - y))

	errs=[]
	for fold in range(num_folds):
		start, end = fold * sizeof_fold, (fold+1)* sizeof_fold
		testX = trainX[start:end]
		testY = trainY[start:end]
		validX = np.append(trainX[:start],trainX[end:],axis=0)
		validY = np.append(trainY[:start],trainY[end:],axis=0)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(epochs):
				# implementing mini-batch GD
				for s in range(0, n-batch_size, batch_size):
					train_op.run(feed_dict={x: validX[s:s+batch_size], y_: validY[s:s+batch_size]})
				if i % 100 == 0:
					print('finished training iter %d'%i)
			test_err = error.eval(feed_dict={x: testX, y_: testY})
			print("start and end are %d and %d" %(start, end))
			print("%d fold test error is: %g" %(fold, test_err))
			errs.append(test_err)
	print('mean error = %g'% np.mean(errs))
	errors.append(np.mean(errs))
print(errors)
# plot learning curves
fig = plt.figure(1)
plt.xlabel('Learning Rates (LOG)')
plt.ylabel('Error')
plt.plot(np.log10(learning_rates), errors)
plt.show()
