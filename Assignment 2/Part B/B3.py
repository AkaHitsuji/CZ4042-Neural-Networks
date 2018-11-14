import numpy as np
import pandas
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import datetime
import time

is_testing = False

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15

if is_testing:
    no_epochs = 5
else:
    no_epochs = 100

lr = 0.01
batch_size = 128
dropout = 0.9

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def char_rnn_model(x,with_dropout=False,cell_type='gru',num_layers=1):
  char_vectors = tf.one_hot(x, 256)

  char_list = tf.unstack(char_vectors, axis=1)

  def create_cell(cell_type):
      if cell_type == 'gru':
          cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
      elif cell_type == 'rnn':
          cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
      elif cell_type == 'lstm':
          cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
      return cell

  if num_layers>1:
      cell = tf.contrib.rnn.MultiRNNCell([create_cell(cell_type) for _ in range(num_layers)])
  else:
      cell = create_cell(cell_type)

  if with_dropout:
      cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout, output_keep_prob=dropout)

  _, encoding = tf.nn.static_rnn(cell, char_list, dtype=tf.float32)

  # encoding is a tuple (c,h), however dense layer only expects one input, h, therefore we only return h
  if isinstance(encoding, tf.nn.rnn_cell.LSTMStateTuple) or isinstance(encoding, tuple):
            encoding = encoding[-1]

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return logits, char_list

def read_data_chars():

  x_train, y_train, x_test, y_test = [], [], [], []

  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[1])
      y_train.append(int(row[0]))

  with open('test_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[1])
      y_test.append(int(row[0]))

  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)


  char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(char_processor.fit_transform(x_train)))
  x_test = np.array(list(char_processor.transform(x_test)))
  y_train = y_train.values
  y_test = y_test.values

  return x_train, y_train, x_test, y_test

def main(with_dropout=False,cell_type='gru',num_layers=1,gradient_clipping=False):

  x_train, y_train, x_test, y_test = read_data_chars()

  if is_testing:
    x_train = x_train[:500]
    y_train = y_train[:500]
    x_test = x_test[:10]
    y_test = y_test[:10]

  print(len(x_train))
  print(len(x_test))

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  logits, char_list = char_rnn_model(x,with_dropout,cell_type,num_layers)

  # Optimizer
  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))

  if gradient_clipping:
      optimizer = tf.train.AdamOptimizer(learning_rate=lr)
      gvs = optimizer.compute_gradients(entropy)
      capped_gvs = [(tf.clip_by_value(grad, -2., 2.), var) for grad, var in gvs]
      train_op = optimizer.apply_gradients(capped_gvs)

  else:
      train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

  # Accuracy
  correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), y_), tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  # training
  loss = []
  test_accuracy = []

  # shuffle data
  N = len(x_train)
  index = np.arange(N)

  # start timer
  timer = time.time()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(no_epochs):
        np.random.shuffle(index)
        x_train, y_train = x_train[index], y_train[index]

        # mini batch learning
        for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})

        _, loss_  = sess.run([train_op, entropy], {x: x_train, y_: y_train})
        test_acc_ = accuracy.eval(feed_dict={x:x_test, y_: y_test})

        loss.append(loss_)
        test_accuracy.append(test_acc_)

        if e%1 == 0:
            print('iter: %d, entropy: %g, test_accuracy:%g'%(e, loss[e],test_accuracy[e]))

  sess.close()

  # time taken for graph
  elapsed_time = (time.time() - timer)
  elapsed_time_str = 'Elapsed time: ' + str(datetime.timedelta(seconds=elapsed_time)).split(".")[0]

  # plot graph
  if with_dropout:
      title = 'Accuracy/Loss of Char RNN Classifier with Dropout'
      filename = 'graphs/B5-char-rnn-with-dropout'+str(datetime.datetime.now())+'.png'
  elif cell_type == 'lstm':
      title = 'Accuracy/Loss of Char RNN Classifier using LSTM'
      filename = 'graphs/B6a-char-rnn-lstm'+str(datetime.datetime.now())+'.png'
  elif cell_type == 'rnn':
      title = 'Accuracy/Loss of Char RNN Classifier using Vanilla RNN'
      filename = 'graphs/B6a-char-rnn-v_rnn'+str(datetime.datetime.now())+'.png'
  elif num_layers > 1:
      title = 'Accuracy/Loss of Char RNN Classifier with multiple layers'
      filename = 'graphs/B6b-char-rnn-2_layers'+str(datetime.datetime.now())+'.png'
  elif gradient_clipping:
      title = 'Accuracy/Loss of Char RNN Classifier with Gradient Clipping'
      filename = 'graphs/B6c-char-rnn-grad_clipping'+str(datetime.datetime.now())+'.png'
  else:
      title = 'Accuracy/Loss of Char RNN Classifier'
      filename = 'graphs/B3-char-rnn-'+str(datetime.datetime.now())+'.png'

  plt.plot(range(len(loss)), loss, label='trng_loss')
  plt.plot(range(len(test_accuracy)), test_accuracy, label='test_acc')
  plt.legend()
  plt.suptitle(title)
  subtitle_str = elapsed_time_str + '   Max Test Accuracy: ' + str(max(test_accuracy))
  plt.title(subtitle_str, fontsize=9)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy/Loss')
  plt.savefig(filename.replace(' ','-').replace(':','.'))
  plt.close()

if __name__ == '__main__':
  main()
