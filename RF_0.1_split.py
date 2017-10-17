
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import os, argparse, itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
import time
import scipy

''' Data split in 10'%' training set and 90'%' testing set
    in order to compare with the PART algorithm from a related work '''

if __name__ == '__main__':

  start_time = time.time()
  parser = argparse.ArgumentParser(description='Run RF for 0.1 on a preprocessed dataset')
  parser.add_argument('-d', '--dataset', required=True, type=str, help='Dataset directory')
  parser.add_argument('-i', '--iters', default=10, type=int, help='Number of random search samples')

  flags = parser.parse_args()

  dataset_dir = flags.dataset
  dataset_filenames = [
    'Xs_train.npy', 'Xs_val.npy', 'Xs_test.npy',
    'Ys_train.npy', 'Ys_val.npy', 'Ys_test.npy'
  ]

  dataset_filenames = map(lambda x: os.path.join(dataset_dir, x), dataset_filenames)
  X_train, X_valid, X_test, Y_train, Y_valid, Y_test = map(np.load, dataset_filenames)

  print(X_train.shape)
  print(X_valid.shape)
  print(X_test.shape)

  X = np.concatenate((X_train,X_valid,X_test))
  y = np.concatenate((Y_train,Y_valid,Y_test))
  
  n_label, label_dim = Y_train.shape
  print np.unique(Y_train)
  print(label_dim)

  output_dir = os.path.join(dataset_dir, 'results_comparison_PART')
  if not os.path.exists(output_dir):
        os.makedirs(output_dir)

  X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size = 0.1, test_size=0.9)
  
  if len(np.unique(Y_train)) > 8:
    print("More than 8 labels")
    category = {0:0, 1:4, 2:4, 3:4, 4:4, 5:4, 6:4, 7:4, 8:4, 9:4, 10:4, 11:4, 12:4,
                13:3, 14:3, 15:3, 16:3, 17:3, 18:6, 19:5, 20:7, 21:5, 22:5, 23:7, 24:7,
                25:2, 26:2, 27:2, 28:2, 29:1, 30:1, 31:1, 32:1, 33:2, 34:2, 35:2}

    for i in range(Y_train.shape[0]):
      key = Y_train[i]
      Y_train[i] = category.get(key[0])
    for j in range(Y_valid.shape[0]):
      key = Y_valid[j]
      Y_valid[j] = category.get(key[0])
    for k in range(Y_test.shape[0]):
      key = Y_test[k]
      Y_test[k] = category.get(key[0])

  print('Label dimension is: {}'.format(label_dim))
  
  if not label_dim in [1, 2, 8, 36]:
    raise Exception('Unknown label dimension! Was {}'.format(label_dim))

  input_dim = X_train.shape[0]
  input_test= X_test.shape[0]
  print('Input dimension training data is: {}'.format(input_dim))
  print('Input dimension test data is: {}'.format(input_test))
  
  for iterations in range(flags.iters):
    
      ne =  np.random.random_integers(5,100)
      md   = np.random.random_integers(5,100)

      cr = 'entropy'
      cw = 'balanced'
      bo = True
      ws = False

      print('Selected n_estimators: '+str(ne))
      print('Selected criterion: '+str(cr))
      print('Selected max_depth: '+str(md))
      

      hyperparams = str(ne)+'--'+str(cr)+'--'+str(md)+'--'+str(bo)+'.txt'
      output_file = os.path.join(output_dir, hyperparams)
      with open(output_file, 'w') as f:
        f.write('Label dimension is: {}'.format(label_dim)+"\n")
        f.write('Input dimension is: {}'.format(input_dim)+"\n")


        clf = RandomForestClassifier(bootstrap=bo, class_weight= cw, criterion=cr,
              max_depth=md, max_features='auto', max_leaf_nodes=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=ne, n_jobs=-1,
              oob_score=True, random_state=False, verbose=0,
              warm_start=ws)

        clf.fit(X_train, Y_train.ravel())

        predictions_train = clf.predict(X_train)  
        acc_train = accuracy_score(Y_train,predictions_train)
        print("Accuracy trainning data = "+str(acc_train))
        f.write("Accuray training data = "+str(acc_train)+"\n")

        predictions_test = clf.predict(X_test)  
        acc_test = accuracy_score(Y_test,predictions_test)
        print("Accuray test data = "+str(acc_test))
        f.write("Accuray test data = "+str(acc_test)+"\n")
        classiReport_test = classification_report(Y_test, predictions_test,digits=3)
        conf_matrix  = confusion_matrix(Y_test, predictions_test)
        print(classiReport_test)
        print(conf_matrix)
        f.write(classiReport_test+"\n")
        f.write(conf_matrix)
        print("--- %s seconds ---" % (time.time() - start_time))
        f.write("--- %s seconds ---" % (time.time() - start_time)+"\n")
