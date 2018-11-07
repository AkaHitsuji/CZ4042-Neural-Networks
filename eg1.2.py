#
# Chapter 1, Example 2
#
print("importingg os")
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

print("device lib")
from tensorflow.python.client import device_lib
print("printing")
print(device_lib.list_local_devices())

print("importingg tf")
import numpy as np
import tensorflow as tf

w1 = tf.Variable([1.0, 1.0], tf.float32) # trainable parameters
b1 = tf.Variable(1.0, tf.float32)
w2 = tf.Variable([1.0, 2.0], tf.float32) # trainable parameters
b2 = tf.Variable(-1.0, tf.float32)
w3 = tf.Variable([-1.0, 1.0], tf.float32) # trainasble parameters
b3 = tf.Variable(2.0, tf.float32)
w = tf.Variable([1.0, -1.0, 1.0], tf.float32) # trainable parameters
b = tf.Variable(-1.5, tf.float32)

def thresh(u):
        shape = tf.shape(u)
        return tf.where(tf.greater(u, tf.zeros(shape)), tf.ones(shape), tf.zeros(shape))

x = tf.placeholder(tf.float32)
u1 = tf.tensordot(w1, x, axes=1) + b1
y1 = thresh(u1)
u2 = tf.tensordot(w2, x, axes=1) + b2
y2 = thresh(u2)
u3 = tf.tensordot(w3, x, axes=1) + b3
y3 = thresh(u3)
u = tf.tensordot(w, [y1, y2, y3], axes=1) + b
y = thresh(u)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

y1_, y2_, y3_, y_ = sess.run([y1, y2, y3, y], {x: [0.0, 0.0]})
print([0.0, 0.0])
print('y1:{}, y2:{}, y3:{}, y:{}'.format(y1_, y2_, y3_, y_))

y1_, y2_, y3_, y_ = sess.run([y1, y2, y3, y], {x: [2.0, 2.0]})
print([2.0, 2.0])
print('y1:{}, y2:{}, y3:{}, y:{}'.format(y1_, y2_, y3_, y_))

y1_, y2_, y3_, y_ = sess.run([y1, y2, y3, y], {x: [-1.0, 1.0]})
print([-1.0, 1.0])
print('y1:{}, y2:{}, y3:{}, y:{}'.format(y1_, y2_, y3_, y_))
