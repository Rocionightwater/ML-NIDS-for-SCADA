
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import os, argparse, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score
import time
import scipy

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Run SVM on a preprocessed dataset')
  parser.add_argument('-d', '--dataset', required=True, type=str, help='Dataset directory')
  parser.add_argument('-i', '--iters', required=True, type=int, help='Number of iters (times to sample hyperparams)')
  parser.add_argument('-f', '--fraction-per', default= 1.0, type=float, help='Fraction of training data to use')

  flags = parser.parse_args()

  dataset_dir = flags.dataset
  dataset_filenames = [
    'Xs_train.npy', 'Xs_val.npy', 'Xs_test.npy',
    'Ys_train.npy', 'Ys_val.npy', 'Ys_test.npy'
  ]

  dataset_filenames = map(lambda x: os.path.join(dataset_dir, x), dataset_filenames)
  X_train, X_valid, X_test, Y_train, Y_valid, Y_test = map(np.load, dataset_filenames)

  X = np.concatenate((X_train,X_valid,X_test))
  y = np.concatenate((Y_train,Y_valid,Y_test))

  X, y = shuffle(X, y, random_state=42)

  X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size = 0.8, test_size=0.2,random_state=42)

  #Shuffle and Split data
  train_val_split = StratifiedShuffleSplit(n_splits=1, train_size = 0.75, test_size=0.25,random_state=42)
  train_val_split.get_n_splits(X_train, Y_train)

  for train_index, val_index in train_val_split.split(X_train, Y_train):

    X_train, X_valid = X[train_index], X[val_index]
    Y_train, Y_valid = y[train_index], y[val_index]

  if flags.fraction_per > 1.0:
    print('Cannot choose bigger fraction of train data than 1.0')
    sys.exit(1)
  elif flags.fraction_per < 1.0:
    idx_until = int(flags.fraction_per * len(X_train))
    X_train = X_train[:idx_until]
    Y_train = Y_train[:idx_until]
 
  label_dim = len(np.unique(Y_train))
  print(label_dim)

  output_dir = os.path.join(dataset_dir, 'results')
  if not os.path.exists(output_dir):
          os.makedirs(output_dir)

  print(Y_train.shape)
  print(Y_valid.shape)
  print(Y_test.shape)

  print('Label dimension is: {}'.format(label_dim))
  
  if not label_dim in [1, 2, 8, 36]:
    raise Exception('Unknown label dimension! Was {}'.format(label_dim))

  input_dim = X_train.shape[0]
  print('Input dimension is: {}'.format(input_dim))

  sample_C = lambda: 10 ** np.random.uniform(1,3)
  sample_gamma = lambda: 10 ** np.random.uniform(-4, 0)

  for i in range(flags.iters):

    C = sample_C()
    gamma = sample_gamma()

    print('Selected C: {:.3f}'.format(C))
    print('Selected gamma: {:.4f}'.format(gamma))

    filepath = '{:.3f}_C--{:.4f}_gamma.txt'.format(C, gamma)
    filepath = os.path.join(output_dir, filepath)

    with open(filepath, 'w') as f:

      f.write('Label dimension is: {}'.format(label_dim)+"\n")
      f.write('Input dimension is: {}'.format(input_dim)+"\n")
      
      if label_dim == 2:
        svr = svm.SVC(kernel="rbf", gamma=gamma, C=C, class_weight = 'balanced')
      else:
        svr = OneVsRestClassifier(svm.SVC(kernel="rbf", gamma=gamma, C=C, class_weight = 'balanced'))
             
      start_time = time.time()
      svr.fit(X_train, Y_train.ravel())
      end_time = time.time()

      predictions_train = svr.predict(X_train)  
      acc_train = accuracy_score(Y_train,predictions_train)
      predictions_val = svr.predict(X_valid)  
      acc_val = accuracy_score(Y_valid,predictions_val)
      classiReport = classification_report(Y_valid, predictions_val)

      print("Accuracy training data = "+str(acc_train))
      f.write("Accuracy training data = "+str(acc_train)+"\n")
      print("Accuracy validation data = "+str(acc_val))
      f.write("Accuracy validation data = "+str(acc_val)+"\n")
      print(classiReport)
      f.write(classiReport+"\n")

      predictions_test = svr.predict(X_test)  
      acc_test         = accuracy_score(Y_test,predictions_test)
      print("Accuracy test data = "+str(acc_test))
      f.write("Accuracy test data = "+str(acc_test)+"\n")
      classiReport_test = classification_report(Y_test, predictions_test)
      print(classiReport_test)
      f.write(classiReport_test+"\n")
      duration = end_time - start_time
      print("--- Training took %s seconds ---" % duration)
      f.write("--- Training took %s seconds ---" % duration)
