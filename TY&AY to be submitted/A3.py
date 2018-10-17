#
# Project 1, starter code part a
#
import time
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
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

NUM_FEATURES = 36
NUM_CLASSES = 6
num_neurons = [5,10,15,20,25]

learning_rate = 0.01
epochs = 1000
batch_size = 64

seed = 10
np.random.seed(seed)

#read train data
train_input = np.loadtxt('sat_train.txt',delimiter=' ')
trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))
train_Y[train_Y == 7] = 6 #changes all 7s to 6

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix

#read test data
test_input = np.loadtxt('sat_test.txt',delimiter=' ')
testX, test_Y = test_input[:,:36], test_input[:,-1].astype(int)
testX = scale(testX, np.min(testX, axis=0), np.max(testX, axis=0))
test_Y[test_Y == 7] = 6

testY = np.zeros((test_Y.shape[0], NUM_CLASSES))
testY[np.arange(test_Y.shape[0]), test_Y-1] = 1 #one hot matrix


# experiment with small datasets
# trainX = trainX[:100]
# trainY = trainY[:100]
# testX = testX[:20]
# testY = testY[:20]


n = trainX.shape[0] #n=1000,number of datasets
m = testX.shape[0] #n=1000,number of datasets

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])


# Build the graph for the deep net


# final_train_acc = []
final_test_acc = []
final_train_err = []
time_taken = []
for neuron_num in num_neurons:
    V = init_weights(NUM_FEATURES,neuron_num)
    c = init_bias(neuron_num)
    W = init_weights(neuron_num, NUM_CLASSES)
    b = init_bias(NUM_CLASSES)

    h = tf.nn.sigmoid(tf.matmul(x, V) + c)
    logits  = tf.matmul(h, W) + b

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    beta = tf.constant(0.000001)
    L2_regularization = tf.nn.l2_loss(V) + tf.nn.l2_loss(W)
    loss = tf.reduce_mean(cross_entropy + beta*L2_regularization)

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    err = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.int32))

    start_time = time.time()
    print('running with batch size: %d'%batch_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_err = []
        test_acc = []
        for i in range(epochs):
            for starting_idex in range(0, n-batch_size, batch_size):
                train_op.run(feed_dict={x: trainX[starting_idex:starting_idex+batch_size], y_: trainY[starting_idex:starting_idex+batch_size]})
            
            train_err.append(err.eval(feed_dict={x: trainX, y_: trainY}))
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))

            if i % 100 == 0:
                print('iter %d: accuracy %g, errors %g'%(i, test_acc[i],train_err[i]))
        print('final test accuracy %g'%test_acc[-1])
        print('final train error %g'%train_err[-1])
        sess.close()

    final_train_err.append(train_err)
    final_test_acc.append(test_acc)
    time_taken.append(time.time() - start_time)
    print('Time taken: %g'%time_taken[-1])

# plot learning curves
plt.figure(1)
for test_acc in final_test_acc:
    plt.plot(range(epochs), test_acc)
plt.xlabel('Number of iterations')
legend = []
for i in num_neurons:
    legend.append(str(i) + " Neurons")
plt.legend(legend)
plt.ylabel('Accuracy in %')

plt.figure(2)
for train_err in final_train_err:
    plt.plot(range(epochs), train_err)
plt.xlabel('Number of iterations')
legend = []
for i in num_neurons:
    legend.append(str(i) + " Neurons")
plt.legend(legend)
plt.ylabel('Classification errors')

plt.figure(3)
plt.plot(num_neurons, time_taken)
plt.xlabel('Batch Size')
plt.ylabel('Time Taken in seconds')

plt.show()

