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
num_neurons = 25

learning_rate = 0.01
epochs = 1000
batch_size = 4

seed = 10
np.random.seed(seed)

#read train data
train_input = np.loadtxt('sat_train.txt',delimiter=' ')
train_X, train_Y = train_input[:,:36], train_input[:,-1].astype(int)
trainX = scale(train_X, np.min(train_X, axis=0), np.max(train_X, axis=0))
train_Y[train_Y == 7] = 6 #changes all 7s to 6

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix

#read test data
test_input = np.loadtxt('sat_test.txt',delimiter=' ')
test_X, test_Y = test_input[:,:36], test_input[:,-1].astype(int)
testX = scale(test_X, np.min(train_X, axis=0), np.max(train_X, axis=0))
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

V = init_weights(NUM_FEATURES,num_neurons)
c = init_bias(num_neurons)
W = init_weights(num_neurons, NUM_CLASSES)
b = init_bias(NUM_CLASSES)

h = tf.nn.sigmoid(tf.matmul(x, V) + c)
logits  = tf.matmul(h, W) + b

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
beta = [10**-3,10**-6,10**-9,10**-12,0]
L2_regularization = tf.nn.l2_loss(V) + tf.nn.l2_loss(W)

# Create the gradient descent optimizer with the given learning rate.


correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

for current_beta in beta:
    loss = tf.reduce_mean(cross_entropy + current_beta*L2_regularization)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    start_time = time.time()
    print('running with beta: %d'%current_beta)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_err = []
        test_acc = []
        for i in range(epochs):
            rand_array = np.arange(n)
            np.random.shuffle(rand_array)
            trainX = trainX[rand_array]
            trainY = trainY[rand_array]
            for starting_idex in range(0, n-batch_size, batch_size):
                train_op.run(feed_dict={x: trainX[starting_idex:starting_idex+batch_size], y_: trainY[starting_idex:starting_idex+batch_size]})
            
            train_err.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))

            if i % 100 == 0:
                print('iter %d: accuracy %g, errors %g'%(i, test_acc[i],train_err[i]))
        print('final test accuracy %g'%test_acc[-1])
        print('final train error %g'%train_err[-1])
        sess.close()

    final_train_err.append(train_err)
    final_test_acc.append(test_acc[-1])
    time_taken.append(time.time() - start_time)
    print('Time taken: %g'%time_taken[-1])

# plot learning curves

plt.figure(1)
for train_err in final_train_err:
    plt.plot(range(epochs), train_err)
plt.xlabel('Number of iterations')
legend = []
for i in beta:
    legend.append("Beta: "+str(i))
plt.legend(legend)
plt.ylabel('Training Data Error')
plt.savefig('./figures/A4_Fig1.png')

plt.figure(2)
plt.plot(beta,final_test_acc)
plt.xlabel("Beta")
plt.xscale("log")
plt.ylabel('Test Data Accuracy')
plt.savefig('./figures/A4_Fig2.png')

plt.show()

