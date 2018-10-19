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
learning_rate = 0.5*(10**-6)
beta = 10**-3
epochs = 1000
batch_size = 32
num_neurons = [20,40,60,80,100]
seed = 10
np.random.seed(seed)
num_folds = 5

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
train_X, train_Y = X_data[test_size:], Y_data[test_size:]
testX, testY = X_data[:test_size], Y_data[:test_size]

# experiment with small datasets
train_X = train_X[:1000]
train_Y = train_Y[:1000]
train_size = train_X.shape[0]

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, 1])

# Build the graph for the deep net


errors = []
sizeof_fold = train_size//num_folds
for number in num_neurons:
	print('Starting training for number of neurons: %d'%number)
	#Create the gradient descent optimizer with the given learning rate.
	V = init_weights(NUM_FEATURES, number)
	c = init_bias(number)
	W = init_weights(number)
	b = init_bias()
	h = tf.nn.relu(tf.matmul(x, V) + c)
	y = tf.matmul(h, W) + b

	square_error = tf.square(y_ - y)
	regularization = tf.nn.l2_loss(V) + tf.nn.l2_loss(W)
	loss = tf.reduce_mean(square_error + beta *regularization)

	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = optimizer.minimize(loss)
	error = tf.reduce_mean(tf.square(y_ - y))

	errs=[]
	for fold in range(num_folds):
		start, end = fold * sizeof_fold, (fold+1)* sizeof_fold
		validX = train_X[start:end]
		validY = train_Y[start:end]
		trainX = np.append(train_X[:start],train_X[end:],axis=0)
		trainY = np.append(train_Y[:start],train_Y[end:],axis=0)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(epochs):
				rand_array = np.arange(trainX.shape[0])
				np.random.shuffle(rand_array)
				trainX = trainX[rand_array]
				trainY = trainY[rand_array]
				# implementing mini-batch GD
				for s in range(0, trainX.shape[0]-batch_size, batch_size):
					train_op.run(feed_dict={x: trainX[s:s+batch_size], y_: trainY[s:s+batch_size]})
				if i % 100 == 0:
					print('finished training iter %d'%i)
			fold_err = error.eval(feed_dict={x: validX, y_: validY})
			print("start and end are %d and %d" %(start, end))
			print("%d fold test error is: %g" %(fold, fold_err))
			errs.append(fold_err)
	print('mean error = %g'% np.mean(errs))
	errors.append(np.mean(errs))
print(errors)
# plot learning curves
fig = plt.figure(1)
plt.xlabel('Learning Rates')
plt.xscale("log")
plt.ylabel('Test Data Error')
plt.plot(learning_rate, errors)
plt.savefig('./figures/B3_Fig1.png')
plt.show()
