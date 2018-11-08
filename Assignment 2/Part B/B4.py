import numpy as np
import pandas
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
from datetime import datetime

is_testing = True

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 20

if is_testing:
    no_epochs = 5
else:
    no_epochs = 1000

lr = 0.01
batch_size = 128
dropout = 0.9

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def word_rnn_model(x,with_dropout):
  word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
  word_list = tf.unstack(word_vectors, axis=1)

  cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)

  if with_dropout:
      cell = tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=dropout, output_keep_prob=dropout)

  _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return logits, word_list

def data_read_words():

  x_train, y_train, x_test, y_test = [], [], [], []

  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[2])
      y_train.append(int(row[0]))

  with open("test_medium.csv", encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[2])
      y_test.append(int(row[0]))

  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  y_train = y_train.values
  y_test = y_test.values

  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH)

  x_transform_train = vocab_processor.fit_transform(x_train)
  x_transform_test = vocab_processor.transform(x_test)

  x_train = np.array(list(x_transform_train))
  x_test = np.array(list(x_transform_test))

  no_words = len(vocab_processor.vocabulary_)
  print('Total words: %d' % no_words)

  return x_train, y_train, x_test, y_test, no_words

def main(with_dropout=False):
  global n_words
  tf.reset_default_graph()

  x_train, y_train, x_test, y_test, n_words = data_read_words()

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

  logits, word_list = word_rnn_model(x,with_dropout)

  # Optimizer
  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
  train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

  # Accuracy
  correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), y_), tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  # sess = tf.Session()
  # sess.run(tf.global_variables_initializer())

  # training
  loss = []
  test_accuracy = []

  # shuffle data
  N = len(x_train)
  index = np.arange(N)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(no_epochs):
        np.random.shuffle(index)
        x_train, y_train = x_train[index], y_train[index]

        _, loss_  = sess.run([train_op, entropy], {x: x_train, y_: y_train})
        test_acc_ = accuracy.eval(feed_dict={x:x_test, y_: y_test})

        loss.append(loss_)
        test_accuracy.append(test_acc_)

        if e%1 == 0:
            print('iter: %d, entropy: %g, test_accuracy:%g'%(e, loss[e],test_accuracy[e]))

  sess.close()

  # plot graph
  plt.plot(range(len(loss)), loss, label='trng_loss')
  plt.plot(range(len(test_accuracy)), test_accuracy, label='test_acc')
  plt.legend()
  plt.title('Accuracy/Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy/Loss')
  if with_dropout:
      filename = 'graphs/B4-word-rnn-with-dropout'+str(datetime.now())+'.png'
  else:
      filename = 'graphs/B4-word-rnn-'+str(datetime.now())+'.png'
  plt.savefig(filename.replace(' ','-').replace(':','.'))
  plt.close()

if __name__ == '__main__':
  main()
