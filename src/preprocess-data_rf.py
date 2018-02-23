import scipy, argparse, itertools, os
from scipy.io import arff
import numpy as np
import sklearn.utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from keras.utils import np_utils


class MaskedStandardScaler:

  def fit_transform(self, Xs):
    self.mean = Xs.mean(axis=0)
    self.std = Xs.std(axis=0)
    return self.transform(Xs)

  def transform(self, Xs):
    return (Xs - self.mean) / self.std

class MaskedMinMaxScaler:

  def fit_transform(self, Xs):
    self.min = Xs.min(axis=0)
    self.max = Xs.max(axis=0)
    return self.transform(Xs)

  def transform(self, Xs):
    return (Xs - self.min) / (self.max - self.min)

def to_categorical_with_nans(Xs, n_cats):
  categorized = np.zeros((len(Xs), n_cats))
  for n, v in enumerate(np.array(Xs)):
    if not np.isnan(v):
      categorized[n, int(v)] = 1 
  return categorized

def pairwise(iterable):
  a, b = itertools.tee(iterable)
  next(b, None)
  return zip(a, b)

def split_dataset(data, labels, train_per_split, val_per_split, test_per_split):
  assert (train_per_split + val_per_split + test_per_split) <= 1.0
  
  train_end_index = int(data.shape[0] * train_per_split)
  valid_end_index = int(data.shape[0] * val_per_split) + train_end_index

  X_train, Y_train = data[:train_end_index], labels[:train_end_index]
  X_valid, Y_valid = data[train_end_index:valid_end_index], labels[train_end_index:valid_end_index]
  X_test,  Y_test  = data[valid_end_index:], labels[valid_end_index:]
  
  return (X_train, X_valid, X_test), (Y_train, Y_valid, Y_test)

def normalize(Xs, op, train_data_model):

  ops = { 
    'mean'     : preprocessing.StandardScaler,
    'minmax'   : preprocessing.MinMaxScaler,
    'm-mean'   : MaskedStandardScaler,
    'm-minmax' : MaskedMinMaxScaler
  }

  if not train_data_model:
    model = ops[op]()
    Xs = model.fit_transform(Xs)
    return Xs, model
  else:
    return train_data_model.transform(Xs), None

def get_cluster_model_k(model, flags):
  if flags.payload_kmeans_imputed:
    return model.cluster_centers_.shape[0]
  elif flags.payload_gmm_imputed:
    return model.weights_.shape[0]

def cluster_payload_features(payloads, flags, train_data_model):
  groups = group_payloads(payloads)
  groups_ids = [np.where(groups == i)[0] for i in range(3)]

  only_nans = np.zeros_like(groups_ids[0])
  only_pres = payloads[groups_ids[1]][:,-1].reshape((-1, 1))
  wo_pres   = payloads[groups_ids[2]][:,:-1]

  if not train_data_model:
    
    if flags.payload_kmeans_imputed:
      print('Imputing payload features with kmeans')
      k1, k2 = flags.payload_kmeans_imputed
      cluster_func = cluster_kmeans
    elif flags.payload_gmm_imputed:
      print('Imputing payload features with GMM')
      k1, k2 = flags.payload_gmm_imputed
      cluster_func = cluster_gmm

    only_pres, only_pressure_cluster_model = cluster_func(only_pres, \
      'pressure', elbow=flags.elbow, k=k1)
    wo_pres, wo_pressure_cluster_model = cluster_func(wo_pres, \
      'remaining payload features', elbow=flags.elbow, k=k2)

  else:
    only_pressure_cluster_model = train_data_model['only_pressure']
    wo_pressure_cluster_model = train_data_model['wo_pressure']
    only_pres = only_pressure_cluster_model.predict(only_pres)
    wo_pres = wo_pressure_cluster_model.predict(wo_pres)

  only_pres += 1
  wo_pres += (get_cluster_model_k(only_pressure_cluster_model, flags) + 1)

  combined_cat = np.concatenate((only_nans, only_pres, wo_pres))
  combined_ind = np.concatenate(groups_ids)
  combined = sorted(zip(combined_ind, combined_cat), key=lambda x: x[0])
  combined = np.array([x[1] for x in combined])

  models = {
    'only_pressure' : only_pressure_cluster_model,
    'wo_pressure' : wo_pressure_cluster_model
  }

  return np_utils.to_categorical(combined), models

def preprocess_payload_features_fulldataset(payloads, flags):


  print('Imputing payload features by keeping old value')
  
  print('Processing pressure measurement values')
  pressures_f = payloads[:,-1]
  pressure_not_nans = np.where(~np.isnan(pressures_f))[0]
  impute_by_keeping_last_value(pressures_f, pressure_not_nans)

  print('Processing binary payload values')
  binary_f = payloads[:,-4:-1]
  binary_f_not_nans = np.where(~np.isnan(binary_f)[:,0])[0]
  impute_by_keeping_last_value(binary_f, binary_f_not_nans)

  print('Processing system mode payload values')
  system_f = payloads[:,-5]
  system_f_not_nans = np.where(~np.isnan(system_f))[0]
  impute_by_keeping_last_value(system_f, system_f_not_nans)

  print('Processing real valued payload values')
  real_val_payloads_f = payloads[:,:-5]
  real_val_payloads_f_not_nans = np.where(~np.isnan(real_val_payloads_f)[:,0])[0]
  impute_by_keeping_last_value(real_val_payloads_f, real_val_payloads_f_not_nans)

  '''restack'''
  payloads = np.column_stack((
    real_val_payloads_f,
    system_f,
    binary_f,
    pressures_f
  ))
  return payloads


def preprocess_data_fulldataset(Xs, flags):
  
  addresses = Xs[:, 0].reshape((-1, 1))
  responses = Xs[:, 2].reshape((-1, 1))

  functions = Xs[:, 1].reshape((-1, 1))
  payloads = Xs[:, 3:14]
  
  models = None

  payloads  = preprocess_payload_features_fulldataset(payloads, flags)

  stacked = np.column_stack((
    addresses, 
    functions, 
    responses,
    payloads
  ))

  if Xs.shape[1] >= 14:
    print('Additional features detected!')
    remaining = Xs[:, 14:]
    stacked = np.column_stack((stacked, remaining))

  return stacked

def preprocess_payload_features(payloads, flags, train_data_model):

  if flags.payload_indicator_imputed:
    print('Imputing payload features using indicators')
    indicators = np.isnan(payloads).astype(int)


    '''split'''
    realval_payload_f = payloads[:,:-5]
    pressure_f = payloads[:,-1].reshape((-1, 1))

    system_f = payloads[:,-5].reshape((-1, 1))
    binary_payload_f = payloads[:,-4:-1]

    '''categorize system_f feature'''
    system_f = np.ma.array(system_f, mask=np.isnan(system_f)) 
    system_f = to_categorical_with_nans(system_f, 3)

    '''masked normalization of realval payload features'''
    op = 'm-' + flags.normalize
    realval_payload_f = np.ma.array(realval_payload_f, mask=np.isnan(realval_payload_f)) 
    realval_payload_f, model = normalize(realval_payload_f, op, train_data_model)
    realval_payload_f = np.array(realval_payload_f)

    '''re-stack'''
    payloads = np.column_stack((
      realval_payload_f,
      system_f,
      binary_payload_f,
      pressure_f
    ))

    '''replace remaining NaNs with 0'''
    payloads[np.isnan(payloads)] = 0
    return np.column_stack((payloads, indicators)), model

  elif flags.payload_keep_value_imputed:
    print('Imputing payload features by keeping old value')
    
    print('Processing pressure measurement values')
    pressures_f = payloads[:,-1]

    print('Processing binary payload values')
    binary_f = payloads[:,-4:-1]

    print('Processing system mode payload values')
    system_f = payloads[:,-5]

    print('Processing real valued payload values')
    real_val_payloads_f = payloads[:,:-5]

    op = flags.normalize
    real_val_payloads_f, model = normalize(real_val_payloads_f, op, train_data_model)

    system_f = to_categorical_with_nans(system_f, 3)

    '''restack'''
    payloads = np.column_stack((
      real_val_payloads_f,
      system_f,
      binary_f,
      pressures_f
    ))
    return payloads, model

  else:
    return cluster_payload_features(payloads, flags, train_data_model)

def impute_by_keeping_last_value(features, not_nans):
  first_not_nan = not_nans[0]
  features[:first_not_nan] = features[first_not_nan]

  for begin, end in pairwise(not_nans):
    features[begin:end] = features[begin]

  '''
  handle the case if we have to keep the value 
  until the end of the data set
  '''
  last = len(features)
  last_not_nan = not_nans[-1]
  if last != last_not_nan:
    features[last_not_nan+1:] = features[last_not_nan]

def group_payloads(payloads):
  '''
  Take the last two columns which uniquely identify a category of payload
  Either:
    both columns are NaNs -> 0
    first is a NaN and second is not an NaN -> 1
    first is not a NaN and second is a NaN -> 2
  Returns indicies
  '''
  ids = payloads[:,-2:]
  ids = np.packbits(~np.isnan(ids), axis=1) // 64
  return ids.reshape(-1)

def preprocess_function_codes(functions, flags, train_data_model):

  if flags.encode_function:    
    '''
    Encode function codes is the same for training and non-training data
    '''
    print('Encoding function...')
    encoder = LabelEncoder()
    functions = encoder.fit_transform(functions.reshape(-1))
    return np_utils.to_categorical(functions), None

  if not train_data_model:

    if flags.cluster_function_kmeans != None:
      k = flags.cluster_function_kmeans
      predicted, model = cluster_kmeans(functions, 'function', elbow=flags.elbow, k=k)
      return np_utils.to_categorical(predicted), model

    elif flags.cluster_function_gmm:
      k = flags.cluster_function_gmm
      predicted, model = cluster_gmm(functions, 'function', elbow=flags.elbow, k=k)
      return np_utils.to_categorical(predicted), model

  else:
    return np_utils.to_categorical(train_data_model.predict(functions)), None

def preprocess_data(Xs, flags, train_data_models = None):
  
  addresses = Xs[:, 0].reshape((-1, 1))
  responses = Xs[:, 2].reshape((-1, 1))

  functions = Xs[:, 1].reshape((-1, 1))
  payloads = Xs[:, 3:14]
  
  models = None

  if not train_data_models:
    functions, f_model = preprocess_function_codes(functions, flags, None)
    payloads, p_model  = preprocess_payload_features(payloads, flags, None)
    models = {'function': f_model, 'payload': p_model}

  else:
    f_model = train_data_models['function']
    p_model = train_data_models['payload']

    functions, _ = preprocess_function_codes(functions, flags, f_model)
    payloads, _  = preprocess_payload_features(payloads, flags, p_model)  

  stacked = np.column_stack((
    addresses, 
    functions, 
    responses,
    payloads
  ))

  if Xs.shape[1] >= 14:
    print('Additional features detected!')
    remaining = Xs[:, 14:]

    if not train_data_models:
      remaining, r_model = normalize(remaining, flags.normalize, None)

      models['remaining'] = r_model
    else:
      r_model = train_data_models['remaining']
      remaining, _ = normalize(remaining, flags.normalize, r_model)

    stacked = np.column_stack((stacked, remaining))

  return stacked, models

def cluster(model_class, score_func, Xs, feature_name, k, elbow, max_k):
  
  if elbow:
    models = [model_class(i).fit(Xs) for i in range(1, max_k+1)]
    scores = [score_func(model, Xs) for model in models]
    k = prompt_elbow_method(scores, feature_name)

  m = model_class(k).fit(Xs)
  return m.predict(Xs), m

def cluster_gmm(Xs, feature_name, k=None, elbow=True):
  
  print('Clustering {} with GMM'.format(feature_name))
  GMM = lambda k: GaussianMixture(n_components=k, init_params='random')
  score_func = lambda gmm, Xs: gmm.aic(Xs)
  return cluster(GMM, score_func, Xs, feature_name, k, elbow, max_k=20)

def cluster_kmeans(Xs, feature_name, k=None, elbow=True):
  
  print('Clustering {} with Kmeans'.format(feature_name))
  KM = lambda k: KMeans(n_clusters=k)
  score_func = lambda km, Xs: km.score(Xs)
  return cluster(KM, score_func, Xs, feature_name, k, elbow, max_k=10)

def prompt_elbow_method(scores, feature_name):
  print(scores)
  xs = range(1, len(scores)+1)
  ys = np.array(scores)
  is_log = ''
  if np.all(ys > 0) or np.all(ys < 0):
    ys = np.log(np.abs(ys))
    is_log = 'log'
  print(ys)
  plt.plot(xs, ys)
  plt.xticks(xs)
  plt.xlabel('Number of clusters')
  plt.ylabel('{} Score'.format(is_log))
  plt.title('Elbow method for {}'.format(feature_name))
  plt.show()
  print('Please enter number of clusters for {}:'.format(feature_name))
  k = int(input('-->'))
  print('\nselected k: {}'.format(k))
  return k

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Preprocess a Mississippi State SCADA Lab dataset.')
  parser.add_argument('-d', '--dataset', type=str, help='Dataset filename')
  parser.add_argument('--label', type=str, choices=['binary', 'category', 'dcategory'],
    default='binary', help='Type of labels to output in the preprocessed dataset')
  parser.add_argument('-t', '--time-series', action='store_true', 
    help='Keep temporal structure of the data')
  parser.add_argument('-n', '--normalize', type=str, choices=['mean', 'minmax'],
    default='mean', help='Type of normalization to preform')
  parser.add_argument('--discard-crc', action='store_true', 
    help='Remove CRC feature from the dataset')
  parser.add_argument('--discard-length', action='store_true', 
    help='Remove Length feature from the dataset')

  function_encoding_group = parser.add_mutually_exclusive_group(required=True)
  function_encoding_group.add_argument('--encode-function', action='store_true',
    help='Encode function codes with One-Hot encoding')
  function_encoding_group.add_argument('--cluster-function-kmeans', type=int,
    help='Cluster function codes with kmeans')
  function_encoding_group.add_argument('--cluster-function-gmm', type=int,
    help='Cluster function codes with GMM')

  payload_features_impution_group = parser.add_mutually_exclusive_group(required=True)
  payload_features_impution_group.add_argument('--payload-keep-value-imputed', action='store_true',
    help='Impute payload features by keeping the oldest value')
  payload_features_impution_group.add_argument('--payload-indicator-imputed', action='store_true',
    help='Impute payload features with 0\'s and add indicators')    
  payload_features_impution_group.add_argument('--payload-kmeans-imputed', type=int, nargs=2,
    help='Impute payload features by clustering with kmeans, first for pressure second for remaining')
  payload_features_impution_group.add_argument('--payload-gmm-imputed', type=int, nargs=2,
    help='Impute payload features by clustering with GMM, first for pressure second for remaining')    

  parser.add_argument('--elbow', action='store_true', 
    help='When using clustering, use elbow method to get best number of clusters')

  parser.add_argument('-o', '--output', type=str, help='Output .npy files')
  parser.add_argument('--split', type=float, nargs=3, help='train/val/test split ratio')

  flags = parser.parse_args()

  dataset, meta = arff.loadarff(flags.dataset)

  label_types = {
    'binary'    : 'binary result',
    'category'  : 'categorized result',
    'dcategory' : 'specific result'
  }

  print('Encoding labels...')
  label_name = label_types[flags.label]
  labels = dataset[label_name].astype(np.float)
  labels = labels.reshape((-1, 1))
  
  
  print('Encoding addressess...')
  #pre-process address before splitting, simple change of values and reshape into column vec
  addresses = preprocessing.label_binarize(dataset['address'], classes=[4])

  print('Encoding functions...')
  functions = dataset['function'].astype(np.float).reshape((-1, 1))

  print('Encoding command responses...')
  #pre-process address before splitting, parse string and reshape into column vec
  responses = dataset['command response'].astype(np.float).reshape((-1, 1))

  print('Extracting payload features...')
  payload_feature_names = meta.names()[3:14]
  payload_features = dataset[payload_feature_names]
  payload_features = payload_features \
    .view(np.float64) \
    .reshape(payload_features.shape + (-1,))

  Xs = np.column_stack((
    addresses, 
    functions, 
    responses,
    payload_features
  ))

  if not flags.discard_length:
    print('\tAdding length...')
    lengths = dataset['length'].astype(np.float).reshape((-1, 1))
    Xs = np.column_stack((Xs, lengths))

  if not flags.discard_crc:
    print('\tAdding crc rate...')
    crcs = dataset['crc rate'].astype(np.float).reshape((-1, 1))
    Xs = np.column_stack((Xs, crcs))

  if flags.time_series:
    print('\tAdding timestamp differences...')
    Xs = np.column_stack((Xs, dataset['time']))
    

  if flags.payload_keep_value_imputed:

    Xs = preprocess_data_fulldataset(Xs,flags)
    Xs, labels = sklearn.utils.shuffle(Xs, labels)

    print('\n\tSplitting dataset...')
    Xs, Ys = split_dataset(Xs, labels, *flags.split)
    Xs_train, Xs_val, Xs_test = Xs
    Ys_train, Ys_val, Ys_test = Ys

    Xs_train_val = np.concatenate((Xs_train,Xs_val))
    Ys_train_val = np.concatenate((Ys_train,Ys_val))

    Xs_train_test = Xs_test
    Ys_train_test = Ys_test

    Xs_train, train_models = preprocess_data(Xs_train, flags)
    Xs_train_val, train_models_val = preprocess_data(Xs_train_val, flags)
    Xs_val,  _ = preprocess_data(Xs_val,  flags, train_data_models=train_models)
    Xs_test, _ = preprocess_data(Xs_test, flags, train_data_models=train_models)
    Xs_train_test, _ = preprocess_data(Xs_train_test, flags, train_data_models=train_models_val)

  else:
    print('\n\tSplitting dataset...')
    Xs, labels = sklearn.utils.shuffle(Xs, labels)
    Xs, Ys = split_dataset(Xs, labels, *flags.split)
    Xs_train, Xs_val, Xs_test = Xs
    Ys_train, Ys_val, Ys_test = Ys

    Xs_train_val = np.concatenate((Xs_train,Xs_val))
    Ys_train_val = np.concatenate((Ys_train,Ys_val))
    Xs_train_test = Xs_test
    Ys_train_test = Ys_test

    Xs_train, train_models = preprocess_data(Xs_train, flags)
    Xs_val,  _ = preprocess_data(Xs_val,  flags, train_data_models=train_models)
    Xs_test, _ = preprocess_data(Xs_test, flags, train_data_models=train_models)

    Xs_train_val, train_models_val = preprocess_data(Xs_train_val, flags)
    Xs_train_test, _ = preprocess_data(Xs_train_test, flags, train_data_models=train_models_val)

    print(Xs_train_val.shape, Ys_train_val.shape)
    print(Xs_train_test.shape, Ys_train_test.shape)


  if not flags.time_series:
    Xs_train, Ys_train = sklearn.utils.shuffle(Xs_train, Ys_train, random_state=0)
 
  output_dir = flags.output

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  np.save(os.path.join(output_dir, 'Xs_train'), Xs_train)
  np.save(os.path.join(output_dir, 'Xs_val'),   Xs_val)
  np.save(os.path.join(output_dir, 'Xs_test'),  Xs_test)

  np.save(os.path.join(output_dir, 'Ys_train'), Ys_train)
  np.save(os.path.join(output_dir, 'Ys_val'),   Ys_val)
  np.save(os.path.join(output_dir, 'Ys_test'),  Ys_test)

  np.save(os.path.join(output_dir, 'Xs_train_val'), Xs_train_val)
  np.save(os.path.join(output_dir, 'Xs_train_test'), Xs_train_test)
  np.save(os.path.join(output_dir, 'Ys_train_val'), Ys_train_val)
  np.save(os.path.join(output_dir, 'Ys_train_test'), Ys_train_test)