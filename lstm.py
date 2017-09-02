#!/usr/bin/env python3
import numpy as np
from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, Dropout,Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.utils import np_utils
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop, Adam
import keras.callbacks
from keras.backend.tensorflow_backend import set_session
import os, argparse, itertools
import sys
from operator import itemgetter
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
import keras.backend as K

class ReportMetric(keras.callbacks.Callback):

  def __init__(self, label_dim, bs, file):
    self.label_dim = label_dim
    self.bs = bs
    self.file = file

  def on_epoch_end(self, batch, logs={}):
    if self.label_dim != 1:
      predict = self.model.predict(self.validation_data[0], batch_size=self.bs, verbose=1)
      predict = np.asarray(predict)
      Y_pred = np.argmax(predict, axis=2).reshape(-1)
      Y_true = np.argmax(self.validation_data[1], axis=2).reshape(-1)
    else:
      predict = model.predict_classes(self.validation_data[0], batch_size=self.bs, verbose=1)
      Y_pred = predict.reshape(-1)
      Y_true = self.validation_data[1].reshape(-1)

    report = classification_report(Y_true, Y_pred,digits=4)
    conf_matrix  = confusion_matrix(Y_true, Y_pred)
    self.file.write('\n' + report + '\n')
    print(report)
    print(conf_matrix)

def make_sequences(Xs, Ys, seqlen, step = 1):
  Xseq, Yseq = [], []
  for i in range(0, Xs.shape[0] - seqlen, step):
    Xseq.append(Xs[i: i+seqlen])
    Yseq.append(Ys[i: i+seqlen])
  return np.array(Xseq), np.array(Yseq)

class WeightedBinaryCrossEntropy(object):

  def __init__(self, weights):
    pos_ratio = weights[0] / (weights[0] + weights[1])
    self.pos_ratio = tf.constant(pos_ratio, tf.float32)
    self.weights = tf.constant(weights[1], tf.float32)
    self.__name__ = 'w_binary_crossentropy({0})'.format(pos_ratio)

  def __call__(self, y_true, y_pred):
    return self.weighted_binary_crossentropy(y_true, y_pred)

  def weighted_binary_crossentropy(self, y_true, y_pred):
    # Transform to logits
    epsilon = tf.convert_to_tensor(K.common._EPSILON, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    y_pred = tf.log(y_pred / (1 - y_pred))

    cost = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.weights)
    return K.mean(cost * self.pos_ratio, axis=-1)

class WeightedCategoricalCrossEntropy(object):

  def __init__(self, weights):
    nb_cl = len(weights)
    self.weights = np.ones((nb_cl, nb_cl))
    for class_idx, class_weight in weights.items():
      self.weights[0][class_idx] = class_weight
      self.weights[class_idx][0] = class_weight
    self.__name__ = 'w_categorical_crossentropy'

  def __call__(self, y_true, y_pred):
    return self.w_categorical_crossentropy(y_true, y_pred)

  def w_categorical_crossentropy(self, y_true, y_pred):
    nb_cl = len(self.weights)
    final_mask = K.zeros_like(y_pred[..., 0])       # 2D -> [?],     3D -> [?, seq]
    y_pred_max = K.max(y_pred, axis=-1)             # 2D -> [?],     3D -> [?, seq]
    y_pred_max = K.expand_dims(y_pred_max, axis=-1) # 2D -> [?, 1],  3D -> [?, seq, 1]
    y_pred_max_mat = K.equal(y_pred, y_pred_max)    # 2D -> [?, nc], 3D -> [?, seq, nc]
    for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
        w = K.cast(self.weights[c_t, c_p], K.floatx())     # scalar
        y_p = K.cast(y_pred_max_mat[..., c_p], K.floatx()) # 2D -> [?], 3D -> [?, seq]
        y_t = K.cast(y_pred_max_mat[..., c_t], K.floatx()) # 2D -> [?], 3D -> [?, seq]
        final_mask += w * y_p * y_t
    return K.categorical_crossentropy(y_pred, y_true) * final_mask

def lstm_model(input_dim, output_dim, seq_len, two_layers, hidden=128, dropout=0.0, lr=0.1):
  model = Sequential()
  layers = {'input': input_dim, 'hidden': hidden, 'output': output_dim}

  model.add(LSTM(layers['hidden'], return_sequences=True,
    input_shape=(seq_len, layers['input'])
  ))
  model.add(Dropout(dropout))

  activation = 'softmax' if output_dim > 1 else 'sigmoid'
  loss = 'categorical_crossentropy' if output_dim > 1 else 'binary_crossentropy'

  model.add(TimeDistributed(Dense(layers['output'], activation=activation)))

  model.compile(loss=loss, optimizer=Adam(lr=lr), metrics=['acc'])
  model.summary()
  return model

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run LSTM on a preprocessed dataset')
  parser.add_argument('-d', '--dataset', required=True, type=str, help='Dataset directory')
  parser.add_argument('-i', '--iters', default=20, type=int, help='Number of random search samples')
  flags = parser.parse_args()

  dataset_dir = flags.dataset
  dataset_filenames = [
    'Xs_train.npy', 'Xs_val.npy', 'Xs_test.npy',
    'Ys_train.npy', 'Ys_val.npy', 'Ys_test.npy'
  ]
  dataset_filenames = map(lambda x: os.path.join(dataset_dir, x), dataset_filenames)
  X_train, X_valid, X_test, Y_train, Y_valid, Y_test = map(np.load, dataset_filenames)

  X = np.concatenate((X_train[2:],X_valid,X_test[:-2]))
  Y = np.concatenate((Y_train[2:],Y_valid,Y_test[:-2]))

  X = np.stack(np.split(X, 68656))
  Y = np.stack(np.split(Y, 68656))

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.6, test_size=0.4, random_state=42)
  X_valid, X_test, Y_valid, Y_test = train_test_split(X_test, Y_test, train_size = 0.5, test_size=0.5, random_state=42)

  X_train = X_train.reshape((-1,) + X_train.shape[2:])
  Y_train = Y_train.reshape((-1,) + Y_train.shape[2:])
  X_valid = X_valid.reshape((-1,) + X_valid.shape[2:])
  Y_valid = Y_valid.reshape((-1,) + Y_valid.shape[2:])
  X_test  = X_test.reshape((-1,) + X_test.shape[2:])
  Y_test  = Y_test.reshape((-1,) + Y_test.shape[2:])

  print(X_train.shape, Y_train.shape)

  n_label, label_dim = Y_train.shape
  sums = np.sum(Y_train, axis=0)
  
  print('Label dimension is: {}'.format(label_dim))
  if not label_dim in [1, 8, 36]:
    raise Exception('Unknown label dimension! Was {}'.format(label_dim))

  input_dim = X_train.shape[1]
  print('Input dimension is: {}'.format(input_dim))

  iters = flags.iters
  print('Number of iterations is: {}'.format(iters))

  learning_rate = lambda: 10 ** np.random.uniform(-2, -1)
  sequence_length = lambda: int(1 * np.random.uniform(4, 4))
  hidden_layer_size = lambda: int(2 ** np.random.uniform(6, 8))
  batch_size = lambda: int(2 ** np.random.uniform(6, 7))
  dropout = lambda: np.random.uniform(0.0, 0.4)
  step = lambda: 1 + int(2 * np.random.uniform(0,0))
  n_epochs = lambda: 10
  #two_layers = lambda: False
  two_layers = lambda: bool(np.random.choice(2))
  
  ranges = [
    learning_rate,
    batch_size,
    n_epochs,
    sequence_length,
    dropout,
    hidden_layer_size,
    step,
    two_layers
  ]

  with open(os.path.join(dataset_dir, 'aggregate.txt'), 'w') as f:
    f.write('Learning Rate,Batch size,Number of epochs,Sequence length,Dropout rate,Hidden layer size\n')

  for iterations in range(iters):
    #p() is calling all functions in the list
    hyperparams = [p() for p in ranges]
    (lr, bs, ne, sl, dr, hls, step, two_layers) = hyperparams
    hparams = ','.join(map(str, hyperparams))

    print('New hyperparameters setting:')
    print(hparams)

    output_dir = os.path.join(dataset_dir, 'results')
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    model = lstm_model(input_dim, label_dim, sl, \
      two_layers, hidden=hls, dropout=dr, lr=lr)

    print('Preparing sequences... Length = {}'.format(sl))

    #Shuffle and Split data
    X_train_seq, Y_train_seq = make_sequences(X_train, Y_train, sl, sl)
    X_valid_seq, Y_valid_seq = make_sequences(X_valid, Y_valid, sl, sl)
    X_test_seq,  Y_test_seq  = make_sequences(X_test,  Y_test,  sl, sl)

    output_file = os.path.join(output_dir, '-'.join(map(str, hyperparams)) + '.txt')
    with open(output_file, 'w') as f:

      log_dir = os.path.join(output_dir, '_'.join(map(str, hyperparams)))
      if not os.path.exists(log_dir):
        os.makedirs(log_dir)
      
      chpt_filepath = os.path.join(log_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')

      smcb = keras.callbacks.ModelCheckpoint(chpt_filepath)
      tbcb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)
      rmcb = ReportMetric(label_dim, bs, f)      

      h = model.fit(X_train_seq, Y_train_seq, \
        validation_data=(X_valid_seq, Y_valid_seq), batch_size=bs, epochs=ne, \
        callbacks=[tbcb, smcb, rmcb]
      )

      # h = model.fit(X_train_seq, Y_train_seq, \
      #   validation_data=(X_test_seq, Y_test_seq), batch_size=bs, epochs=ne, \
      #   callbacks=[tbcb, smcb, rmcb]
      # )
      for k,v in h.history.items():
        f.write(k + ':' + ','.join(map(str, v)) + '\n')

