#
# Project 2, starter code Part a
#

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle

NUM_SAMPLES = 4
NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 2000
batch_size = 128

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
    
    #Conv 1
    W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, 50], stddev=1.0/np.sqrt(NUM_CHANNELS*9*9)), name='weights_1')
    b1 = tf.Variable(tf.zeros([50]), name='biases_1')

    conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)
    pool_1 = tf.nn.max_pool(conv_1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')
    
    #Conv 2
    W2 = tf.Variable(tf.truncated_normal([5, 5, 50, 60], stddev=1.0/np.sqrt(NUM_CHANNELS*5*5)), name='weights_2')
    b2 = tf.Variable(tf.zeros([60]), name='biases_2')

    conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)
    pool_2 = tf.nn.max_pool(conv_2, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_2')

    dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value 
    pool_2_flat = tf.reshape(pool_2, [-1, dim])
    
    # Fully Connected
    W3 = tf.Variable(tf.truncated_normal([dim, 300], stddev=1.0/np.sqrt(dim*300)), name='weights_3')
    b3 = tf.Variable(tf.zeros([300]), name='biases_3')
    
    u3 = tf.matmul(pool_2_flat, W3) + b3
    y3 = tf.nn.relu(u3)
  
    #Softmax
    W4 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0/np.sqrt(300*NUM_CLASSES)), name='weights_4')
    b4 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')
    
    logits = tf.matmul(y3, W4) + b4

    return conv_1, pool_1, conv_2, pool_2, logits


def main():

    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)
    
    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)
    
    # Scale data
    testX = (testX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)
    trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)
    
    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    conv_1, pool_1, conv_2, pool_2, logits = cnn(x)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    N = len(trainX)
    idx = np.arange(N)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_err = []
        test_acc = []
        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]
            
            for i in range(int(trainX.shape[0]/batch_size)):
                x_batch = trainX[i*batch_size:(i+1)*batch_size]  
                y_batch = trainY[i*batch_size:(i+1)*batch_size]
            
                _, loss_ = sess.run([train_step, loss], {x: x_batch, y_: y_batch})
            train_err.append(loss_)
            acc = accuracy.eval(feed_dict={x: testX, y_: testY})
            test_acc.append(acc)

print('epoch', e, 'entropy', loss_, 'accuracy', acc)
    
        # Plot training cost vs epochs    
        plt.figure()
        plt.plot(range(epochs), train_err, label = 'Training Cost')
        plt.xlabel(str(epochs) + ' Epochs')
        plt.ylabel('Training Cost')
        plt.title('Training Cost vs Epochs')
        plt.savefig('./images/p1a_1.png')
#        plt.legend()
#        plt.show()
        
        # Plot test accuracy vs epochs    
        plt.figure()
        plt.plot(range(epochs), test_acc, label = 'Test Accuracy')
        plt.xlabel(str(epochs) + ' Epochs')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy vs Epochs')
        plt.savefig('./images/p1a_2.png')
#        plt.legend()
#        plt.show()
    
        for sample in range(NUM_SAMPLES):
            
            ind = np.random.randint(low=0, high=N)
            X = trainX[ind,:]
            
            plt.figure()
            X_show = X.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)
            plt.axis('off')
            plt.imshow(X_show)
            plt.savefig('./images/p1b_'+ str(sample) + '_test_pattern.png')
            
            conv_1_, pool_1_, conv_2_, pool_2_ = sess.run([conv_1, pool_1, conv_2, pool_2], {x: X.reshape(1, NUM_CHANNELS*IMG_SIZE*IMG_SIZE)})
            
            plt.figure()
            conv_1_array = np.array(conv_1_)
            for i in range(50):
                plt.subplot(5, 10, i+1); plt.axis('off'); plt.imshow(conv_1_array[0,:,:,i])
            plt.savefig('./images/p1b_'+ str(sample) + '_conv_1.png')
    
            plt.figure()
            pool_1_array = np.array(pool_1_)
            for i in range(50):
                plt.subplot(5, 10, i+1); plt.axis('off'); plt.imshow(pool_1_array[0,:,:,i])
            plt.savefig('./images/p1b_'+ str(sample) + '_pool_1.png')
            
            plt.figure()
            conv_2_array = np.array(conv_2_)
            for i in range(60):
                plt.subplot(6, 10, i+1); plt.axis('off'); plt.imshow(conv_2_array[0,:,:,i])
            plt.savefig('./images/p1b_'+ str(sample) + '_conv_2.png')
            
            plt.figure()
            pool_2_array = np.array(pool_2_)
            for i in range(60):
                plt.subplot(6, 10, i+1); plt.axis('off'); plt.imshow(pool_2_array[0,:,:,i])
            plt.savefig('./images/p1b_'+ str(sample) + '_pool_2.png')
plt.close(fig='all')

if name == 'main':
  main()