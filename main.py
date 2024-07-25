## base package
import numpy as np
import random as ran
import time
import math
from scipy.special import comb
import os
import pickle

## model package
# XGBoost
import xgboost as xgb
# MLP
from sklearn.neural_network import MLPClassifier
# SVM
from sklearn import svm
# Logistic Regression
from sklearn.linear_model import LogisticRegression
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
# Random Forest
from sklearn.ensemble import RandomForestClassifier
# KNN
from sklearn.neighbors import KNeighborsClassifier
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
# LASSO
from sklearn.linear_model import Lasso
# evaluation measures
from sklearn.metrics import average_precision_score, roc_curve, auc

## parallel package
from functools import partial
from pathos.pools import ProcessPool, ThreadPool
from tqdm import tqdm

## feature engineering
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, DictionaryLearning
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap


MAX_TRAIN = 20
FEATURE_SAMPLING = 10
method_names = ['optimal', 'XGBoost+square', 'XGBoost+logistic', 'XGBoost+hinge', 'XGBoost+softmax', 'MLP', 'SVM', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN', 'Naive Bayes', 'LASSO']
all_artificial_type = ['poisson', 'normal', 'power']
parameter = {'poisson': [1, 4, 10], 'normal': [1, 10], 'power': [2, 4]}
new_output = './other_output'
document = './new_cleaned_data'
feature_output = './other_feature_output'


def parallel(func, *args, show=False, thread=False, **kwargs):

    p_func = partial(func, **kwargs)
    pool = ThreadPool() if thread else ProcessPool()
    try:
        if show:
            start = time.time()
            with tqdm(total=len(args[0]), desc="computing process") as t:
                r = []
                for i in pool.imap(p_func, *args):
                    r.append(i)
                    t.set_postfix({'parallel function': func.__name__, "computing cost": "%ds" % (time.time() - start)})
                    t.update()
        else:
            r = pool.map(p_func, *args)
        return r
    except Exception as e:
        print(e)
    finally:
        pool.close()  # close the pool to any new jobs
        pool.join()  # cleanup the closed worker processes
        pool.clear()  # Remove server with matching state


def data_input(document, file):
  datas, labels = [], []
  with open(document + '/' + file, 'r') as f:
    lines = f.readlines()
    if ',' in lines[0]:
      dot = ','
    else:
      dot = ' '

    for line in lines:
      new_line = line.strip('\n').split(dot)
      while '' in new_line:
        new_line.remove('')
      data = [int(x) for x in new_line[:-2]]
      label = int(new_line[-2])
      datas.append(data)
      labels.append(label)
  feature_num = len(datas[0])

  def discrete_list(l):
    l1 = list(set(l))
    c = len(l1)
    e2e = {}
    for i, x in enumerate(l1):
      e2e[x] = i
    new_l = [e2e[x] for x in l]
    return c, new_l

  class_num, new_labels = discrete_list(labels)
  return feature_num, class_num, datas, new_labels


def artificial_data(data_type, parameter, division_prob, data_size=100000):
  # generate three kinds of artificial data: possion, normal and power
  if data_type == 'poisson':
    datas = [[x] for x in np.random.poisson(parameter, data_size)]
  elif data_type == 'normal':
    datas = [[int(x)] for x in np.random.normal(10, parameter, data_size) if x > 0]
  elif data_type == 'power':
    datas = [[x] for x in np.random.zipf(parameter, data_size)]
  
  new_labels = np.random.binomial(1, division_prob, data_size)

  return datas, new_labels
  

def data_division(datas, labels, p):
  x_train, x_test, y_train, y_test = [[] for i in range(4)]
  for i, data in enumerate(datas):
    if ran.random() < p:
      x_train.append(data)
      y_train.append(labels[i])
    else:
      x_test.append(data)
      y_test.append(labels[i])
  return x_train, y_train, x_test, y_test


def min_hinge(distribution):
  return sum([min(value[0], value[1]) for value in distribution.values()])


def training(datas, labels, p):
  ### determine out of sample or in sample
  if p < 1 and p > 0:
    # out of sample
    while True:
      x_train, y_train, x_test, y_test = data_division(datas, labels, p)
      if len(x_train) > 0 and len(x_test) > 0:
        break
  elif p == 1:
    # in sample
    x_train, y_train, x_test, y_test = datas, labels, datas, labels
  else:
    return 0
  
  ### training process
  all_predictions = {}
  ## optimal model
  # data constructing (calculating the frequency of every feature vector)
  all_different_datas = []
  for x in datas:
    if x not in all_different_datas:
      all_different_datas.append(x)

  data_distribution = {}
  for i, x in enumerate(x_train):
    j = all_different_datas.index(x)
    if j not in data_distribution:
      data_distribution[j] = [0, 0, 0, 0]
    data_distribution[j][y_train[i]] += 1

  for i, x in enumerate(x_test):
    j = all_different_datas.index(x)
    if j not in data_distribution:
      data_distribution[j] = [0, 0, 0, 0]
    data_distribution[j][y_test[i] + 2] += 1
  # calculate JSD (overlapping) between positive and negative instances
  delta = 0
  for value in data_distribution.values():
    a1, a2, a3, a4 = value
    delta += max(a1, a3) + max(a2, a4) - max(a1 + a2, a3 + a4)

  basic = [js_entropy(data_distribution), len(datas), len(x_train), len(x_test), delta]
  hinges = {}
  # optimal score for each feature vector based on its frequency
  opt_pred = []
  for i, x in enumerate(x_test):
    value = data_distribution[all_different_datas.index(x)]
    opt_pred.append(value[3] / (value[2] + value[3]))
  
  all_predictions['optimal'] = opt_pred
  hinges['optimal'] = min_hinge(data_distribution)
  
  ## XGBoost model
  # construct training set and test set
  dtrain = xgb.DMatrix(np.array(x_train), label=np.array(y_train))
  dtest = xgb.DMatrix(x_test)
  # square loss
  model = xgb.train({'booster': 'gbtree', 'objective': 'reg:squarederror'}, dtrain, MAX_TRAIN)
  all_predictions['square'] = model.predict(dtest)
  hinges['square'] = continuous_classifier_hinge_accuracy(model.predict(dtrain), y_train)
  # logistic loss
  model = xgb.train({'booster': 'gbtree', 'objective': 'reg:logistic'}, dtrain, MAX_TRAIN)
  all_predictions['logistic'] = model.predict(dtest)
  hinges['logistic'] = continuous_classifier_hinge_accuracy(model.predict(dtrain), y_train)
  # hinge loss
  model = xgb.train({'booster': 'gbtree', 'objective': 'binary:hinge'}, dtrain, MAX_TRAIN)
  all_predictions['hinge'] = model.predict(dtest)
  hinges['hinge'] = discrete_classifier_hinge_accuracy(model.predict(dtrain), y_train)
  # softmax loss
  model = xgb.train({'booster': 'gbtree', 'objective': 'multi:softmax', 'num_class': 2}, dtrain, MAX_TRAIN)
  all_predictions['softmax'] = model.predict(dtest)
  hinges['softmax'] = discrete_classifier_hinge_accuracy(model.predict(dtrain), y_train)

  ## MLP
  clf = MLPClassifier()
  clf.fit(x_train, y_train)
  all_predictions['MLP'] = clf.predict(x_test)
  hinges['MLP'] = discrete_classifier_hinge_accuracy(clf.predict(x_train), y_train)

  ## SVM
  clf = svm.SVC()
  clf.fit(x_train, y_train)
  all_predictions['SVM'] = clf.predict(x_test)
  hinges['SVM'] = discrete_classifier_hinge_accuracy(clf.predict(x_train), y_train)

  ## Logistic Regression
  clf = LogisticRegression()
  clf.fit(x_train, y_train)
  all_predictions['LR'] = clf.predict(x_test)
  hinges['LR'] = continuous_classifier_hinge_accuracy(clf.predict(x_train), y_train)

  ## Decision Tree
  clf = DecisionTreeClassifier()
  clf.fit(x_train, y_train)
  all_predictions['DT'] = clf.predict(x_test)
  hinges['DT'] = discrete_classifier_hinge_accuracy(clf.predict(x_train), y_train)

  ## Random Forest
  clf = RandomForestClassifier()
  clf.fit(x_train, y_train)
  all_predictions['RF'] = clf.predict(x_test)
  hinges['RF'] = discrete_classifier_hinge_accuracy(clf.predict(x_train), y_train)

  ## KNN
  clf = KNeighborsClassifier()
  clf.fit(x_train, y_train)
  all_predictions['KNN'] = clf.predict(x_test)
  hinges['KNN'] = discrete_classifier_hinge_accuracy(clf.predict(x_train), y_train)

  ## Naive Bayes
  clf = GaussianNB()
  clf.fit(x_train, y_train)
  all_predictions['NB'] = clf.predict(x_test)
  hinges['NB'] = discrete_classifier_hinge_accuracy(clf.predict(x_train), y_train)

  ## LASSO
  clf = Lasso()
  clf.fit(x_train, y_train)
  all_predictions['Lasso'] = clf.predict(x_test)
  hinges['Lasso'] = discrete_classifier_hinge_accuracy(clf.predict(x_train), y_train)
  print(hinges)
  print(basic)

  return basic, hinges, all_predictions, y_test


def artificial_training_testing(x_train, y_train, x_test, y_test):
  datas = x_train + x_test
  
  ### training process
  all_predictions = {}
  ## optimal model
  # data constructing (calculating the frequency of every feature vector)
  all_different_datas = []
  for x in datas:
    if x not in all_different_datas:
      all_different_datas.append(x)

  data_distribution = {}
  for i, x in enumerate(x_train):
    j = all_different_datas.index(x)
    if j not in data_distribution:
      data_distribution[j] = [0, 0, 0, 0]
    data_distribution[j][y_train[i]] += 1

  for i, x in enumerate(x_test):
    j = all_different_datas.index(x)
    if j not in data_distribution:
      data_distribution[j] = [0, 0, 0, 0]
    data_distribution[j][y_test[i] + 2] += 1
  # calculate JSD (overlapping) between positive and negative instances
  js = js_entropy(data_distribution)
  basic = [js, len(datas), len(x_train), len(x_test)]
  hinges = {}
  # optimal score for each feature vector based on its frequency
  opt_pred = []
  for i, x in enumerate(x_test):
    value = data_distribution[all_different_datas.index(x)]
    opt_pred.append(value[3] / (value[2] + value[3]))
  
  all_predictions['optimal'] = opt_pred
  hinges['optimal'] = min_hinge(data_distribution)
  
  ## XGBoost model
  # construct training set and test set
  dtrain = xgb.DMatrix(np.array(x_train), label=np.array(y_train))
  dtest = xgb.DMatrix(x_test)
  # square loss
  model = xgb.train({'booster': 'gbtree', 'objective': 'reg:squarederror'}, dtrain, MAX_TRAIN)
  all_predictions['square'] = model.predict(dtest)
  hinges['square'] = continuous_classifier_hinge_accuracy(model.predict(dtrain), y_train)
  # logistic loss
  model = xgb.train({'booster': 'gbtree', 'objective': 'reg:logistic'}, dtrain, MAX_TRAIN)
  all_predictions['logistic'] = model.predict(dtest)
  hinges['logistic'] = continuous_classifier_hinge_accuracy(model.predict(dtrain), y_train)
  # hinge loss
  model = xgb.train({'booster': 'gbtree', 'objective': 'binary:hinge'}, dtrain, MAX_TRAIN)
  all_predictions['hinge'] = model.predict(dtest)
  hinges['hinge'] = discrete_classifier_hinge_accuracy(model.predict(dtrain), y_train)
  # softmax loss
  model = xgb.train({'booster': 'gbtree', 'objective': 'multi:softmax', 'num_class': 2}, dtrain, MAX_TRAIN)
  all_predictions['softmax'] = model.predict(dtest)
  hinges['softmax'] = discrete_classifier_hinge_accuracy(model.predict(dtrain), y_train)

  ## MLP
  clf = MLPClassifier()
  clf.fit(x_train, y_train)
  all_predictions['MLP'] = clf.predict(x_test)
  hinges['MLP'] = discrete_classifier_hinge_accuracy(clf.predict(x_train), y_train)

  ## SVM
  clf = svm.SVC()
  clf.fit(x_train, y_train)
  all_predictions['SVM'] = clf.predict(x_test)
  hinges['SVM'] = discrete_classifier_hinge_accuracy(clf.predict(x_train), y_train)

  ## Logistic Regression
  clf = LogisticRegression()
  clf.fit(x_train, y_train)
  all_predictions['LR'] = clf.predict(x_test)
  hinges['LR'] = continuous_classifier_hinge_accuracy(clf.predict(x_train), y_train)

  ## Decision Tree
  clf = DecisionTreeClassifier()
  clf.fit(x_train, y_train)
  all_predictions['DT'] = clf.predict(x_test)
  hinges['DT'] = discrete_classifier_hinge_accuracy(clf.predict(x_train), y_train)

  ## Random Forest
  clf = RandomForestClassifier()
  clf.fit(x_train, y_train)
  all_predictions['RF'] = clf.predict(x_test)
  hinges['RF'] = discrete_classifier_hinge_accuracy(clf.predict(x_train), y_train)

  ## KNN
  clf = KNeighborsClassifier()
  clf.fit(x_train, y_train)
  all_predictions['KNN'] = clf.predict(x_test)
  hinges['KNN'] = discrete_classifier_hinge_accuracy(clf.predict(x_train), y_train)

  ## Naive Bayes
  clf = GaussianNB()
  clf.fit(x_train, y_train)
  all_predictions['NB'] = clf.predict(x_test)
  hinges['NB'] = discrete_classifier_hinge_accuracy(clf.predict(x_train), y_train)

  ## LASSO
  clf = Lasso()
  clf.fit(x_train, y_train)
  all_predictions['Lasso'] = clf.predict(x_test)
  hinges['Lasso'] = discrete_classifier_hinge_accuracy(clf.predict(x_train), y_train)
  print(hinges)
  print(basic)

  return basic, hinges, all_predictions, y_test


def simple_training(datas, labels, p):
  # ### determine out of sample or in sample
  # if p < 1 and p > 0:
  #   # out of sample
  #   while True:
  #     x_train, y_train, x_test, y_test = data_division(datas, labels, p)
  #     if len(x_train) > 0 and len(x_test) > 0:
  #       break

  x_train, y_train, x_test, y_test = datas, labels, datas, labels

  ### training process
  all_predictions = {}
  ## optimal model
  # data constructing (calculating the frequency of every feature vector)
  all_different_datas = []
  for x in x_train:
    if x not in all_different_datas:
      all_different_datas.append(x)

  data_distribution = {}
  for i, x in enumerate(x_train):
    j = all_different_datas.index(x)
    if j not in data_distribution:
      data_distribution[j] = [0, 0]
    data_distribution[j][y_train[i]] += 1
  # calculate JSD (overlapping) between positive and negative instances
  js = js_entropy(data_distribution)
  # optimal score for each feature vector based on its frequency
  opt_pred = []
  for i, x in enumerate(x_test):
    value = data_distribution[all_different_datas.index(x)]
    opt_pred.append(value[1] / sum(value))

  all_predictions['optimal'] = opt_pred
  
  # ## XGBoost model
  # # construct training set and test set
  # dtrain = xgb.DMatrix(np.array(x_train), label=np.array(y_train))
  # dtest = xgb.DMatrix(x_test)
  # # square loss
  # model = xgb.train({'booster': 'gbtree', 'objective': 'reg:squarederror'}, dtrain, MAX_TRAIN)
  # all_predictions['square'] = model.predict(dtest)

  # # logistic loss
  # model = xgb.train({'booster': 'gbtree', 'objective': 'reg:logistic'}, dtrain, MAX_TRAIN)
  # all_predictions['logistic'] = model.predict(dtest)

  # # hinge loss
  # model = xgb.train({'booster': 'gbtree', 'objective': 'binary:hinge'}, dtrain, MAX_TRAIN)
  # all_predictions['hinge'] = model.predict(dtest)

  # # softmax loss
  # model = xgb.train({'booster': 'gbtree', 'objective': 'multi:softmax', 'num_class': 2}, dtrain, MAX_TRAIN)
  # all_predictions['softmax'] = model.predict(dtest)

  # ## MLP
  # clf = MLPClassifier()
  # clf.fit(x_train, y_train)
  # all_predictions['MLP'] = clf.predict(x_test)

  # ## SVM
  # clf = svm.SVC()
  # clf.fit(x_train, y_train)
  # all_predictions['SVM'] = clf.predict(x_test)

  # ## Logistic Regression
  # clf = LogisticRegression()
  # clf.fit(x_train, y_train)
  # all_predictions['LR'] = clf.predict(x_test)

  # ## Decision Tree
  # clf = DecisionTreeClassifier()
  # clf.fit(x_train, y_train)
  # all_predictions['DT'] = clf.predict(x_test)

  # ## Random Forest
  # clf = RandomForestClassifier()
  # clf.fit(x_train, y_train)
  # all_predictions['RF'] = clf.predict(x_test)

  # ## KNN
  # clf = KNeighborsClassifier()
  # clf.fit(x_train, y_train)
  # all_predictions['KNN'] = clf.predict(x_test)

  # ## Naive Bayes
  # clf = GaussianNB()
  # clf.fit(x_train, y_train)
  # all_predictions['NB'] = clf.predict(x_test)

  # ## LASSO
  # clf = Lasso()
  # clf.fit(x_train, y_train)
  # all_predictions['Lasso'] = clf.predict(x_test)
  
  ### evaluation process
  all_evaluation_measures = []
  for key, value in all_predictions.items():
    fpr1, tpr1, thresholds1 = roc_curve(y_test, value, pos_label=1)
    roc_auc = auc(fpr1, tpr1)
    all_evaluation_measures.append(roc_auc)
    print(key + ': ' + str(roc_auc))
  return js, all_evaluation_measures


def js_entropy(data_distribution):
  positive_sum, negative_sum = 0, 0
  for x in data_distribution.values():
    positive_sum += x[1]
    negative_sum += x[0]
  
  js = 0
  for value in data_distribution.values():
    p0 = value[0] / negative_sum
    p1 = value[1] / positive_sum
    if p0 > 0:
      js += 0.5 * p0 * math.log2(2 * p0 / (p0 + p1))
    if p1 > 0:
      js += 0.5 * p1 * math.log2(2 * p1 / (p0 + p1))

  return js


def feature2data(all_data, features):
  new_datas1 = [[data1[i] for i in features] for data1 in all_data]
  return new_datas1


def feature_engineering(datas, labels, original_js, p=1):
  feature_num = len(datas[0])
  all_nums = [i for i in range(feature_num)]
  selected_features = []
  feature_based_results = []

  for k in range(1, feature_num + 1):

    if k > 50:
      break
    
    results = []
    jss = []
    print('*' * 40)
    print('# sampled feature num:', k)

    feature_sampling_num = min(FEATURE_SAMPLING, len(all_nums))
    potential_nums = ran.sample(all_nums, feature_sampling_num)
    for turn in range(feature_sampling_num):
      selected_datas = feature2data(datas, selected_features+[potential_nums[turn]])
      js, aucs = simple_training(selected_datas, labels, p)
      jss.append(js)
      results.append(aucs)
    
    max_index = jss.index(max(jss))
    result = results[max_index]
    all_nums.remove(potential_nums[max_index])
    selected_features.append(potential_nums[max_index])
    opt_datas = feature2data(datas, selected_features)
    print('JSD=', round(max(jss), 4))
    for i, x in enumerate(result):
      print(method_names[i] + '=', round(x, 4))
    feature_based_results.append([jss[max_index]]+result)

    if max(jss) == original_js:
      break

    '''
    ## feafeature extraction
    # LDA is supervised (with labels) and its n_components is one for binary classification
    # print('LDA begining...')
    lda = LinearDiscriminantAnalysis(n_components=1)
    extracted_datas = lda.fit_transform(opt_datas, labels)
    
    new_datas = []
    for i, x in enumerate(extracted_datas):
      new_datas.append(opt_datas[i]+list(x))

    js, aucs = simple_training(selected_datas, labels, p)
    feature_based_results.append([js]+aucs)
    print('JSD=', round(js, 4))
    for i, x in enumerate(aucs):
      print(method_names[i] + '=', round(x, 4))

    # PCA is unsupervised (without labels) and its n_components is no more than min(n_samples, n_features)
    print('ICA:')
    for j in range(1, k+1):
      ica = FastICA(n_components=j)
      extracted_datas = ica.fit_transform(opt_datas)
      print('# extracted features:', j)
      
      new_datas = []
      for ix, x in enumerate(extracted_datas):
        new_datas.append(opt_datas[ix]+list(x))
      js, aucs = simple_training(selected_datas, labels, p)
      feature_based_results.append([js]+aucs)
      print('JSD=', round(js, 4))
      for ix, x in enumerate(aucs):
        print(method_names[ix] + '=', round(x, 4))
    '''
  return feature_based_results


def feature_based_experiment(file, ix):
  print('file name:', file.split('.')[0])
  feature_num, class_num, datas, labels = data_input(document, file)
  print('# data:', len(datas))
  print('# total feature:', feature_num)
  print('# total class:', class_num)

  for i1 in range(10):
    try:
      with open('./other_output/' + file + '_0.1_' + str(i1) + '.txt', 'rb') as f:
        js = pickle.load(f)[0][0]
      break
    except:
      continue
  print('# original js:', js)

  feature_results = feature_engineering(datas, labels, js)
  with open(feature_output + '/' + file + '_' + str(ix) + '.txt', 'wb') as f:
    pickle.dump(feature_results, f)


def experiment(file, p, ix):
  feature_num, class_num, datas, labels = data_input(document, file)
  results = training(datas, labels, p)
  js = results[0][0]
  print('JSD:', round(js, 4))
  with open(new_output + '/' + file + '_' + str(p) + '_' + str(ix) + '.txt', 'wb') as f:
    pickle.dump(results, f)
      

def bound_of_artificial_data(input_datas, input_labels, output_datas, output_labels):
  # data constructing (calculating the frequency of every feature vector)
  all_different_datas = []
  for x in input_datas + output_datas:
    if x not in all_different_datas:
      all_different_datas.append(x)

  data_distribution = {}
  for i, x in enumerate(input_datas):
    j = all_different_datas.index(x)
    if j not in data_distribution:
      data_distribution[j] = [0, 0, 0, 0]
    data_distribution[j][input_labels[i]] += 1
  for i, x in enumerate(output_datas):
    j = all_different_datas.index(x)
    if j not in data_distribution:
      data_distribution[j] = [0, 0, 0, 0]
    data_distribution[j][output_labels[i] + 2] += 1
  
  min_hinge, max_accuracy, delta = 0, 0, 0
  for key, value in data_distribution.items():
    a, b, c, d = value
    min_hinge += min(a, b)
    max_accuracy += max(c, d)
    if (a-b)*(c-d) < 0:
      delta += min(abs(a-b), abs(c-d))
  
  return min_hinge, max_accuracy, delta


def artificial_experiment(input_type, input_prob, input_parameter, output_type, output_prob, output_parameter):
  input_datas, input_labels = artificial_data(input_type, input_parameter, input_prob)
  output_datas, output_labels = artificial_data(output_type, output_parameter, output_prob)

  # results = artificial_training_testing(input_datas, input_labels, output_datas, output_labels)
  # with open('./results_'+input_type+'_'+str(input_parameter)+'_'+str(input_prob)+'_'+output_type+'_'+str(output_parameter)+'_'+str(output_prob)+'.txt', 'wb') as f:
  # pickle.dump(results, f)
  
  bounds = bound_of_artificial_data(input_datas, input_labels, output_datas, output_labels)

  with open('./artificial_bounds.txt', 'wb') as f:
    f.write('\t'.join([input_type, str(input_parameter), str(input_prob), output_type, str(output_parameter), str(output_prob)]) + '\n')

  # with open('./bounds_'+input_type+'_'+str(input_parameter)+'_'+str(input_prob)+'_'+output_type+'_'+str(output_parameter)+'_'+str(output_prob)+'.txt', 'wb') as f:
  #   pickle.dump(bounds, f)


def bound_calculation(file):
  feature_num, class_num, datas, labels = data_input(document, file)
  m = len(labels)
  print(file, m)
  if m > 10000:
    bound_sampling = 100000
  else:
    bound_sampling = 1000000
  
  # data constructing (calculating the frequency of every feature vector)
  all_different_datas = []
  for x in datas:
    if x not in all_different_datas:
      all_different_datas.append(x)

  data_distribution = {}
  for i, x in enumerate(datas):
    j = all_different_datas.index(x)
    if j not in data_distribution:
      data_distribution[j] = [0, 0]
    data_distribution[j][labels[i]] += 1
  
  results = {}
  for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    min_hinge_loss, max_accuracy, min_delta = 0, 0, 0
    train_num, test_num = 0, 0
    extreme_a, extreme_b = 0, 0

    for value in data_distribution.values():
      a, b = value
      if a == 0:
        extreme_b += b
      elif b == 0:
        extreme_a += a
      else:
        print(a, b)
        for turn in range(bound_sampling):
          a1 = np.random.binomial(a, p)
          b1 = np.random.binomial(b, p)

          train_num += a1 + b1
          test_num += a + b - a1 - b1

          min_hinge_loss += min(a1, b1)
          max_accuracy += max(a - a1, b - b1)
          min_delta += max(a1, b1) + max(a - a1, b - b1) - max(a, b)
    
    for turn in range(bound_sampling):
      extreme_b1 = np.random.binomial(extreme_b, p)
      extreme_a1 = np.random.binomial(extreme_a, p)

      train_num += extreme_a1 + extreme_b1
      test_num += extreme_a - extreme_a1 + extreme_b - extreme_b1

      max_accuracy += extreme_a - extreme_a1 + extreme_b - extreme_b1

    results[p] = [min_hinge_loss / train_num, max_accuracy / test_num, min_delta / bound_sampling / m]
  
  print(results)

  with open('./bounds_'+file, 'wb') as f:
    pickle.dump(results, f)


def continuous_classifier_hinge_accuracy(train_pred, y_train):
  score_distribution = {}
  for i, x in enumerate(train_pred):
    if x not in score_distribution:
      score_distribution[x] = [0,0]
    score_distribution[x][y_train[i]] += 1
  
  all_scores = sorted(list(score_distribution.keys()), reverse=True)
  hinge_loss = sum([value[1]  for value in score_distribution.values()])
  all_hinges = [hinge_loss]
  for score in all_scores:
    value = score_distribution[score]
    hinge_loss += value[0] - value[1]
    all_hinges.append(hinge_loss)

  return min(all_hinges)


def discrete_classifier_hinge_accuracy(y_train, train_pred):
  hinge_loss = 0
  for i, x in enumerate(train_pred):
    if x != y_train[i]:
      hinge_loss += 1
  
  return hinge_loss


if __name__ == '__main__':
  all_file = list(os.listdir(document))
  all_p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

  # in-sample experiment
  fs, ppp, ixs = [], [], []
  for file in all_file:
    fs.append(file)
    ppp.append(1)
    ixs.append(0)
  # parallel(experiment, fs, ppp, ixs)

  # out-of-sample experiment
  fs, ppp, ixs = [], [], []
  for file in all_file:
    for i in range(5):
      for p in all_p:
        fs.append(file)
        ppp.append(p)
        ixs.append(i)
  # parallel(experiment, fs, ppp, ixs)

  # phase 2
  # parallel(bound_calculation, all_file)
  # special_files = ['Android Malware Detection.csv', 'Covid Dataset.csv', 'TUANDROMD.csv', 'WineQuality.csv']
  # parallel(bound_calculation, special_files)
  # bound_calculation('vehicle_stolen_dataset_with_headers.csv')

  # phase 3
  # feature_based_experiment('CSA-Data.csv', 0)
  parallel(feature_based_experiment, all_file, [0 for x in all_file])
  # special_files = ['BranchPrediction.csv', 'diabetes_012_health_indicators_BRFSS2015.csv', 'Heart Disease.csv', 'heart_disease_health_indicators_BRFSS2015.csv']
  # parallel(feature_based_experiment, special_files, [0 for x in special_files])
  
  ## synthetic data experiment
  # poisson(5), normal(10), power(2)
  all_artificial_prob = [0.1, 0.5, 0.9]

  # artificial_experiment('poisson', 0.2, 'power', 0.7)
  
  input_types, output_types = [], []
  ips, ops = [], []
  input_paras, output_paras = [], []

  for tp in all_artificial_type:
    for tp1 in all_artificial_type:
      for pp in all_artificial_prob:
        for pp1 in all_artificial_prob:
          for para1 in parameter[tp]:
            for para2 in parameter[tp1]:
              input_types.append(tp)
              output_types.append(tp1)
              ips.append(pp)
              ops.append(pp1)
              input_paras.append(para1)
              output_paras.append(para2)

  # parallel(artificial_experiment, input_types, ips, input_paras, output_types, ops, output_paras)

