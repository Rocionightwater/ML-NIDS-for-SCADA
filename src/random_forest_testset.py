
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


def main():
  start_time = time.time()
  parser = argparse.ArgumentParser(description='Run RF on a preprocessed dataset')
  parser.add_argument('-d', '--dataset', required=True, type=str, help='Dataset directory')
  parser.add_argument('-i', '--iters', default=10, type=int, help='Number of random search samples')

  flags = parser.parse_args()

  dataset_dir = flags.dataset
  dataset_filenames = [
    'Xs_train.npy', 'Xs_val.npy', 'Xs_test.npy', 'Xs_train_val.npy', 'Xs_train_test.npy',
    'Ys_train.npy', 'Ys_val.npy', 'Ys_test.npy', 'Ys_train_val.npy', 'Ys_train_test.npy'
  ]

  dataset_filenames = map(lambda x: os.path.join(dataset_dir, x), dataset_filenames)
  X_train, X_valid, X_test, Xs_train_val,Xs_train_test, Y_train, Y_valid, Y_test, Ys_train_val,Ys_train_test = map(np.load, dataset_filenames)



  print(X_train.shape)
  print(X_valid.shape)
  print(X_test.shape)
  print(Xs_train_val.shape, Ys_train_val.shape)
  print(Xs_train_test.shape, Ys_train_test.shape)
  print(Xs_train_val[0])
  print(Xs_train_val[1])
  print(Xs_train_test[2])

  seed = np.random.randint(100)
  
  n_label, label_dim = Ys_train_val.shape
  print(np.unique(Ys_train_val))
  print(label_dim)

  output_dir = os.path.join(dataset_dir, 'total_testset')
  if not os.path.exists(output_dir):
        os.makedirs(output_dir)
  
  '''
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
  '''

  print('Label dimension is: {}'.format(label_dim))
  
  if not label_dim in [1, 2, 8, 36]:
    raise Exception('Unknown label dimension! Was {}'.format(label_dim))



  for iterations in range(flags.iters):
    
    input_dim = Xs_train_val.shape[0]
    input_test= Xs_train_test.shape[0]
    print('Input dimension training data is: {}'.format(input_dim))
    print('Input dimension test data is: {}'.format(input_test))
    
    #ne and md equal to the best coming from hyperparameter search
    ne =  np.random.randint(10,100)
    md   = np.random.randint(10,100)

    cr = np.random.choice(['gini'])
    cw = 'balanced'
    bo = np.random.choice([False])
    mf = 0.5
    ws = False

    print('Selected n_estimators: '+str(ne))
    print('Selected criterion: '+str(cr))
    print('Selected max_depth: '+str(md))
    print('Selected bootstrap: '+str(bo))
    print('Selected max max_features: '+str(mf))

    hyperparams = str(ne)+'--'+str(cr)+'--'+str(md)+'--'+str(bo)+'--balanced.txt'
    output_file = os.path.join(output_dir, hyperparams)
    with open(output_file, 'w') as f:
      f.write('Label dimension is: {}'.format(label_dim)+"\n")
      f.write('Input dimension is: {}'.format(input_dim)+"\n")


      clf = RandomForestClassifier(bootstrap=bo, class_weight= cw, criterion=cr,
            max_depth=md, max_features=mf, max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=ne, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=ws)

      clf.fit(Xs_train_val, Ys_train_val.ravel())

      predictions_train = clf.predict(Xs_train_val)  
      acc_train = accuracy_score(Ys_train_val,predictions_train)

      f.write('max_features: {}'.format(mf)+"\n")
      print("Accuracy trainning data = "+str(acc_train))
      f.write("Accuray training data = "+str(acc_train)+"\n")

      predictions_test = clf.predict(Xs_train_test)  
      acc_test = accuracy_score(Ys_train_test,predictions_test)
      print("Accuray test data = "+str(acc_test))
      f.write("Accuray test data = "+str(acc_test)+"\n")
      classiReport_test = classification_report(Ys_train_test, predictions_test,digits=4)
      conf_matrix  = confusion_matrix(Ys_train_test, predictions_test)
      print(classiReport_test)
      print(conf_matrix)
      f.write(classiReport_test+"\n")
      print("--- %s seconds ---" % (time.time() - start_time))
      f.write("--- %s seconds ---" % (time.time() - start_time)+"\n")

if __name__ == '__main__':
  main()
