#
# Project 2, starter code Part a
#

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle



NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 1000
batch_size = 128
FEATURE_MAP_1 = 90
FEATURE_MAP_2 = 100
DROPOUT = 0.9


seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    
    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels-1] = 1

    return data, labels_




def cnn(images):

    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
    
    #Conv 1 - maps one RGB image (3x32x32) to 50 feature maps (50x24x24), pool to (50x12x12)
    W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, FEATURE_MAP_1], stddev=1.0/np.sqrt(NUM_CHANNELS*9*9)), name='weights_1')
    b1 = tf.Variable(tf.zeros([FEATURE_MAP_1]), name='biases_1')
    conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)
    conv_1 = tf.nn.dropout(conv_1, DROPOUT)
    pool_1 = tf.nn.max_pool(conv_1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')

	#Conv 2 -- maps 50 feature maps (50x12x12) to 60 (60x8x8), pool to (60x4x4)
    W2 = tf.Variable(tf.truncated_normal([5, 5, FEATURE_MAP_1, FEATURE_MAP_2], stddev=1.0/np.sqrt(FEATURE_MAP_1*5*5)), name='weights_2')
    b2 = tf.Variable(tf.zeros([FEATURE_MAP_2]), name='biases_1')
    conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)
    conv_2 = tf.nn.dropout(conv_2, DROPOUT)
    pool_2 = tf.nn.max_pool(conv_2, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_2')

    # Fully connected layer 1 -- after 2 round of downsampling, our 32x32 image
    # is down to 60x4x4 feature maps -- maps this to 300 features
    W_fc1 = tf.Variable(tf.truncated_normal([4*4*FEATURE_MAP_2,300], stddev=1.0/np.sqrt(4*4*FEATURE_MAP_2)), name='weights_fc1')
    b_fc1 = tf.Variable(tf.zeros([300]), name='biases_fc1')
    # W_fc1 = weight_variable([4 * 4 * 60, 300])
    # b_fc1 = bias_variable([300])

    #what does this do?
    dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value 
    pool_2_flat = tf.reshape(pool_2, [-1, dim])
    h_fc1 = tf.nn.relu(tf.matmul(pool_2_flat, W_fc1) + b_fc1)
    h_fc1 = tf.nn.dropout(h_fc1, DROPOUT)
	
    #Softmax
    W_fc2 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0/np.sqrt(300)), name='weights_fc2')
    b_fc2 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_fc2')
    logits = tf.matmul(h_fc1, W_fc2) + b_fc2

    return conv_1, pool_1, conv_2, pool_2,logits


def main():

    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)
    
    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)

    #Scaling the data
    testX = (testX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)
    trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    c1,p1,c2,p2,logits = cnn(x)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    N = len(trainX)
    idx = np.arange(N)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        test_acc = []
        training_loss =[]
        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
            
            training_loss.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
                #_, loss_ = sess.run([train_step, loss], {x: trainX, y_: trainY})

            
            print('epoch', e, 'entropy', training_loss[e], 'test accuracy', test_acc[e])

        plt.figure(1)
        plt.plot(range(epochs), test_acc)
        plt.ylabel('Test Accuracy')
        plt.xlabel('Number of iterations')
        plt.savefig('./A3d-Test_accuracy.png')

        plt.figure(2)
        plt.plot(range(epochs), training_loss)
        plt.xlabel('Number of iterations')
        plt.ylabel('Training Loss')
        plt.savefig('./A3d-Training_loss.png')
 
        #plt.show()



if __name__ == '__main__':
  main()
