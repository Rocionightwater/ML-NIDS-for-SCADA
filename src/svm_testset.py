
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

  flags = parser.parse_args()

  dataset_dir = flags.dataset
  dataset_filenames = [
    'Xs_train.npy', 'Xs_val.npy', 'Xs_test.npy', 'Xs_train_val.npy', 'Xs_train_test.npy',
    'Ys_train.npy', 'Ys_val.npy', 'Ys_test.npy', 'Ys_train_val.npy', 'Ys_train_test.npy'
  ]

  dataset_filenames = map(lambda x: os.path.join(dataset_dir, x), dataset_filenames)
  X_train, X_valid, X_test, Xs_train_val, Xs_train_test, Y_train, Y_valid, Y_test, Ys_train_val,Ys_train_test = map(np.load, dataset_filenames)
 
  label_dim = len(np.unique(Ys_train_val))
  print(label_dim)

  output_dir = os.path.join(dataset_dir, 'results_testset')
  if not os.path.exists(output_dir):
          os.makedirs(output_dir)

  print('Label dimension is: {}'.format(label_dim))
  
  if not label_dim in [1, 2, 8, 36]:
    raise Exception('Unknown label dimension! Was {}'.format(label_dim))

  input_dim = Xs_train_val.shape[0]
  print('Input dimension is: {}'.format(input_dim))

  sample_C = lambda: 10 ** np.random.uniform(1,3)
  sample_gamma = lambda: 10 ** np.random.uniform(-4, 0)

  for i in range(flags.iters):

    print(Xs_train_val.shape,Ys_train_val.shape)
    print(Xs_train_test.shape,Ys_train_test.shape)

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
      svr.fit(Xs_train_val, Ys_train_val.ravel())
      end_time = time.time()

      predictions_train = svr.predict(Xs_train_val)  
      acc_train = accuracy_score(Ys_train_val,predictions_train)

      predictions_test = svr.predict(Xs_train_test)  
      acc_test         = accuracy_score(Ys_train_test,predictions_test)
      print("Accuracy test data = "+str(acc_test))
      f.write("Accuracy test data = "+str(acc_test)+"\n")
      classiReport_test = classification_report(Ys_train_test, predictions_test, digits=4)
      print(classiReport_test)
      f.write(classiReport_test+"\n")
      duration = end_time - start_time
      print("--- Training took %s seconds ---" % duration)
      f.write("--- Training took %s seconds ---" % duration)