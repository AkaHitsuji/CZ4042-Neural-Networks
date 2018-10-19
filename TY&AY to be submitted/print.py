#
# Project 1, starter code part b
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt

errors = [3990596000.0, 4034025000.0, 4066203600.0, 3965050000.0, 3948454000.0]
num_neurons = [20,40,60,80,100]
print(errors)
# plot learning curves
fig = plt.figure(1)
plt.xlabel('Number of neurons')
plt.ylabel('Test Data Error')
plt.plot(num_neurons, errors)
plt.savefig('./figures/B3_Fig1.png')
plt.show()
