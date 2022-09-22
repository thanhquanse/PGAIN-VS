# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from audioop import rms
from re import I
import sys
from unittest import result
import matplotlib.pyplot as plt
import argparse
import yaml
import os
import pandas as pd
import numpy as np

from pgain.pgain import pgain
from libs.pgainutils import rmse_loss, mae_loss, adjusted_r2, std_loss
from libs.utils import load_full_data, rotate_sensors, randomize_missing, rotate_single_sensor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from scipy.optimize import minimize

from openbox import Optimizer, sp
from openbox.benchmark.objective_functions.synthetic import CONSTR
import pybobyqa

ROOT_PATH = os.getcwd()
params = yaml.load(open(ROOT_PATH + '/config.yaml'), yaml.Loader)

parameters_gain = None
dataframe = None
dataframe_dim = None
result_df = None

def f(x):
  print("x[0]: {}, x[1]: {}, x[2]: {}".format(x[0], x[1], x[2]))
  ori_data_x, miss_data_x, data_m = rotate_sensors(dataframe, int(x[0]), int(x[1]), x[2])
  imputed_data_x = pgain(miss_data_x, parameters_gain)
  rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)

  new_row = pd.DataFrame()
  new_row['miss_number'] = [int(x[1])]
  new_row['miss_rate'] = [x[2]]
  new_row['start_idx'] = [int(x[0])]
  new_row['value'] = [rmse]
  new_row.to_csv('optimization_result.csv', mode='a', index=False, header=False)

  return rmse

def minimize_f():
  x0 = [1, 1, 0.01]
  cons = (
    {
    'type': 'ineq',
    'fun': lambda x: x[1] - 2
    },
    {
    'type': 'ineq',
    'fun': lambda x: x[2] - 0.0
    }
  )

  bnds = ((0, 11), (2, 5), (0.02, 0.5))

  res = minimize(f, x0, constraints=cons, bounds=bnds)
  print(res)

def get_configspace():
  space = sp.Space()
  miss_num = dataframe.shape[1] // 2
  miss_number = sp.Int('miss_number', 2, miss_num, None, None)
  miss_rate = sp.Float('miss_rate', 0.01, 0.1, None, None)
  start_idx = sp.Int('start_idx', 0, miss_num, None, None)
  space.add_variables([start_idx, miss_number, miss_rate])

  return space

def objective(config: sp.Configuration):
  parameters = config.get_dictionary()
  round_miss_rate = np.round(parameters['miss_rate'], 2)
  ori_data_x, miss_data_x, data_m = rotate_single_sensor(dataframe, parameters['start_idx'], parameters['miss_number'], round_miss_rate)
  imputed_data_x = pgain(miss_data_x, parameters_gain)
  rmse = np.round(rmse_loss(ori_data_x, imputed_data_x, data_m), 3)
  r2 = np.round(adjusted_r2(ori_data_x, imputed_data_x, data_m), 3)
  std = np.round(std_loss(ori_data_x, imputed_data_x, data_m), 3)
  interval = (dataframe_dim * round_miss_rate * params['adjacent_obs']) / 60 #hours

  if parameters_gain['data_type'] == 'raspihat':
    interval = (dataframe_dim * round_miss_rate * params['raspihat_adjacent_obs']) / 3600 #10s

  result = dict()
  result['objs'] = [rmse,]
  result['constraints'] = [(parameters['miss_number'] * (-1)) + 2, (round_miss_rate * (-1)) + 0.01]

  new_row = pd.DataFrame()
  new_row['miss_number'] = [parameters['miss_number']]
  new_row['miss_rate'] = [round_miss_rate]
  new_row['interval'] = [interval]
  new_row['start_idx'] = [parameters['start_idx']]
  new_row['rmse'] = [rmse]
  new_row['std'] = [std]
  new_row['r2'] = [r2]
  new_row.to_csv(parameters_gain['data_type'] + '_optimization_result_' + parameters_gain['on_subset'] + '.csv', mode='a', index=False, header=False)

  return result

def objective_scipy_minmax(x):
  ori_data_x, miss_data_x, data_m = rotate_sensors(dataframe, 1, x)
  imputed_data_x = pgain(miss_data_x, parameters_gain)
  rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)

  return rmse

def run(args):
  data_type = args.data_type
  miss_rate = args.miss_rate
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations,
                     'data_type': args.data_type,
                     'on_subset': args.on_subset
                    }

  # Load data and introduce missingness
  data_frame = load_full_data(data_type, True)
  data_frame.rename(columns={'response': 'predictor0'}, inplace=True)

  data_frame.replace(0, 0.01, inplace=True)

  if args.on_subset == 'yes':
    if data_type == 'solar':
      subset_str = params['solar_subset']
    if data_type == 'traffic':
      subset_str = params['traffic_subset']
    if data_type == 'raspihat':
      subset_str = params['raspihat_subset']

    subset_list = subset_str.split(',')
    data_frame = data_frame.reindex(columns=subset_list)
  
  print(data_frame)

  global dataframe
  dataframe = data_frame
  global parameters_gain
  parameters_gain = gain_parameters
  global dataframe_dim
  dataframe_dim = data_frame.shape[0]


  # minimize_f()
  # sys.exit()
  result_df =  pd.DataFrame(columns=['miss_number', 'miss_rate', 'duration', 'start_idx', 'rmse', 'std', 'r2'])
  result_df.to_csv(data_type + "_optimization_result_" + args.on_subset + '.csv', index=False)

  # Run
  prob = CONSTR()
  opt = Optimizer(
    objective,
    get_configspace(),
    num_objs=1,
    num_constraints=2,
    acq_optimizer_type='random_scipy',
    max_runs=100,
    surrogate_type='prf',
    time_limit_per_trial=180,
    task_id='so_hpo',
  )
  history = opt.run()

  history = opt.get_history()
  print(history)

  # result_df.to_csv("optimization_result.csv", index=None)

def run_scipy_minmax(args):
  data_type = args.data_type
  miss_rate = args.miss_rate
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}

  # Load data and introduce missingness
  data_frame = load_full_data(data_type, True)

  if args.on_subset == 'yes':
    subset_str = params['subset']
    subset_list = subset_str.split(',')
    data_frame = data_frame.reindex(columns=subset_list)
  
  global dataframe
  dataframe = data_frame
  global parameters_gain
  parameters_gain = gain_parameters

  # Define the starting point
  x0 = 0.01

  # Define bounds (required for global optimization)
  lower = 0.05
  upper = 0.5

  soln = pybobyqa.solve(objective_scipy_minmax, x0, maxfun=10,
                      bounds=(lower, upper),
                      seek_global_minimum=True)
  print(soln)

def run_radomization(args):
  data_type = args.data_type
  miss_rate = args.miss_rate
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}

  # Load data and introduce missingness
  data_frame = load_full_data(data_type, True)

  data_frame.replace(0, 0.01, inplace=True)

  # print(data_frame.to_string())
  # sys.exit()

  if args.on_subset == 'yes':
    subset_str = params['subset']
    subset_list = subset_str.split(',')
    data_frame = data_frame.reindex(columns=subset_list)

  ori_data_x, miss_data_x, data_m = randomize_missing(data_frame, miss_rate)

  print(miss_data_x)

  # Impute missing data
  imputed_data_x = pgain(miss_data_x, gain_parameters)
  
  # Report the RMSE performance
  rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
  
  print()
  print('RMSE Performance: ' + str(np.round(rmse, 4)))

  print()
  # print('R2 Performance: ' + str(np.round(adjusted_r2(ori_data_x, imputed_data_x, data_m), 4)))
  
  return imputed_data_x, rmse

def visualize(args):
  data_type = args.data_type
  miss_rate = args.miss_rate
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}

  # Load data and introduce missingness
  data_frame = load_full_data(data_type, True)

  if args.on_subset == 'yes':
    subset_str = params['subset']
    subset_list = subset_str.split(',')
    data_frame = data_frame.reindex(columns=subset_list)

  time_axis = data_frame.index

  ori_data_x, miss_data_x, data_m = rotate_sensors(data_frame, 0, params['miss_sensor_num'], miss_rate)

  # Impute missing data
  imputed_data_x = pgain(miss_data_x, gain_parameters)

  rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
  print()
  print('RMSE Performance: ' + str(np.round(rmse, 4)))

  for pos in range(0, params['miss_sensor_num'] + 1):
      plt.rcParams["figure.figsize"] = [7.50, 3.50]
      plt.rcParams["figure.autolayout"] = True
      ori_line, = plt.plot(time_axis, ori_data_x[:, pos])
      ori_line.set_label("Original")
      
      imputed_line, = plt.plot(time_axis, imputed_data_x[:, pos], color='red')
      imputed_line.set_label("Imputed")
      plt.xticks(rotation=90, fontsize='xx-small')
      plt.legend(loc="upper right")
      plt.savefig('images/rotation/' + data_type + '_' + str(miss_rate) + '_position_' + str(pos) + '_rmse_' + str(rmse) + '.png', bbox_inches='tight', pad_inches=0, dpi=150)
      # plt.show()
      plt.clf()

def main(args):
  '''Main function for virtual sensor data imputation.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_type = args.data_type
  miss_rate = args.miss_rate
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}

  # Load data and introduce missingness
  data_frame = load_full_data(data_type, True)

  if args.on_subset == 'yes':
    subset_str = params['subset']
    subset_list = subset_str.split(',')
    data_frame = data_frame.reindex(columns=subset_list)
  
  print(data_frame.shape)

  folds = list(KFold(n_splits = 5, shuffle = True, random_state = 42).split(data_frame))
  kfold_rmse_scores = []
  kfold_mae_scores = []
  kfold_mae_custom_scores = []
  kfold_r2_scores = []

  # print(folds)
  for fold_number, (train_idx, val_idx) in enumerate(folds):
    print('\n-----  Fold: %d  -----' % (fold_number))

    # Select train vs test
    train_df = data_frame.iloc[train_idx]
    val_df =  data_frame.iloc[val_idx]

    # Combine
    train_val_df = train_df.append(val_df)

    time_axis = train_val_df.index
  
    ori_data_x, miss_data_x, data_m = rotate_sensors(train_val_df, 0, params['miss_sensor_num'], miss_rate)

  #   np.set_printoptions(suppress=True)
  #   np.set_printoptions(threshold=sys.maxsize)
  #   print(miss_data_x)

    # Impute missing data
    imputed_data_x = pgain(miss_data_x, gain_parameters)

    # Report the RMSE performance
    rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
    kfold_rmse_scores.append(rmse)
    print()
    print('RMSE Performance: ' + str(np.round(rmse, 4)))

    rmse_sklearn = mean_squared_error(ori_data_x, imputed_data_x, squared=False)
    print()
    print('RMSE Sklearn Performance: {} in range min {} and max {}'.format(str(np.round(rmse_sklearn, 4)), np.amin(ori_data_x), np.amax(ori_data_x)))

    mae_sklearn = mean_absolute_error(ori_data_x, imputed_data_x)
    kfold_mae_scores.append(mae_sklearn)
    print()
    print('MAE Sklearn Performance: ' + str(np.round(mae_sklearn, 4)))

    mae = mae_loss(ori_data_x, imputed_data_x, data_m)
    kfold_mae_custom_scores.append(mae)
    print()
    print('MAE custom Performance: ' + str(np.round(mae, 4)))

    r2 = r2_score(ori_data_x, imputed_data_x)
    kfold_r2_scores.append(r2)
    print()
    print('R2 Performance: ' + str(np.round(r2, 4)))

    # for pos in range(params['miss_sensor_num']):
    #   plt.rcParams["figure.figsize"] = [7.50, 3.50]
    #   plt.rcParams["figure.autolayout"] = True
    #   ori_line, = plt.plot(time_axis, ori_data_x[:, pos])
    #   ori_line.set_label("Original")
      
    #   imputed_line, = plt.plot(time_axis, imputed_data_x[:, pos], color='red')
    #   imputed_line.set_label("Imputed")
    #   plt.xticks(rotation=90, fontsize='xx-small')
    #   plt.legend(loc="upper right")
    #   plt.savefig('images/rotation/' + data_type + '_' + str(miss_rate) + '_position_' + str(pos) + '_' + str(params['miss_sensor_num']) + '_missing_' + '_fold_' + str(fold_number) + '.png', bbox_inches='tight', pad_inches=0, dpi=150)
    #   # plt.show()
    #   plt.clf()

  print("----------------------------------------------------------")
  print("RMSE avg acc: %.5f (+/- %.5f)\n" % (np.mean(kfold_rmse_scores), np.std(kfold_rmse_scores)))
  print("MAE avg acc: %.2f (+/- %.2f)\n" % (np.mean(kfold_mae_scores), np.std(kfold_mae_scores)))
  print("MAE custom avg acc: %.2f (+/- %.2f)\n" % (np.mean(kfold_mae_custom_scores), np.std(kfold_mae_custom_scores)))
  print("R2 avg acc: %.2f (+/- %.2f)\n" % (np.mean(kfold_r2_scores), np.std(kfold_r2_scores)))
    
  return imputed_data_x, rmse

def visualize_result(args):
  '''Main function for virtual sensor data imputation.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_type = args.data_type
  miss_rate = args.miss_rate
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}

  # Load data and introduce missingness
  data_frame = load_full_data(data_type, True)
  data_frame.rename(columns={'response': 'predictor0'}, inplace=True)
  data_frame.replace(0, 0.01, inplace=True)

  if args.on_subset == 'yes':
    if data_type == 'solar':
      subset_str = params['solar_subset']
    if data_type == 'traffic':
      subset_str = params['traffic_subset']
    if data_type == 'raspihat':
      subset_str = params['raspihat_subset']

    subset_list = subset_str.split(',')
    data_frame = data_frame.reindex(columns=subset_list)
  
  # print(data_frame)
  time_axis = data_frame.index

  miss_num = args.miss_num
  start_from = args.start_from

  ori_data_x, miss_data_x, data_m, sensor_pos_map = rotate_single_sensor(data_frame, start_from, miss_num, miss_rate)
  # print(miss_data_x)
  np.savetxt(data_type + "_rotation_missing.csv", miss_data_x, delimiter=",", fmt="%f")

  # Impute missing data
  imputed_data_x = pgain(miss_data_x, gain_parameters)

  rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
  print()
  print('RMSE Performance: ' + str(np.round(rmse, 4)))

  for pos in range(start_from, (miss_num  + start_from)):
      plt.rcParams["figure.figsize"] = [7.50, 3.50]
      plt.rcParams["figure.autolayout"] = True

      sensor = data_frame.columns[pos]

      sensor_idx_arr = sensor_pos_map[sensor]
      idxs = sensor_idx_arr[1]

      # print(idxs)
      
      imputed_line, = plt.plot(time_axis[idxs[0]:idxs[1]], imputed_data_x[idxs[0]:idxs[1], pos], color='red')
      imputed_line.set_label("Imputed")

      ori_line, = plt.plot(time_axis[idxs[0]:idxs[1]], ori_data_x[idxs[0]:idxs[1], pos], alpha=0.6, color='grey')
      ori_line.set_label("Original")

      plt.xticks(rotation=90, fontsize='xx-small')
      plt.legend(loc="upper right")
      plt.savefig('images/rotation/' + data_type + '_' + str(miss_rate) + '_position_' + str(pos) + '_missing_' + str(miss_num)  + '_rmse_' + str(rmse) + '.png', bbox_inches='tight', pad_inches=0, dpi=150)
      # plt.show()
      plt.clf()

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_type',
      choices=['solar','temperature', 'raspihat', 'traffic'],
      default='solar',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability in one sensor',
      default=0.3,
      type=float)
  parser.add_argument(
      '--on_subset',
      help='Rotate on the optimal subset only',
      default='no',
      type=str)
  parser.add_argument(
      '--plot_result',
      help='Visualize the imputed result',
      default='yes',
      type=str)
  parser.add_argument(
      '--start_from',
      help='For visualization only, start rotation from index',
      default=0,
      type=int)
  parser.add_argument(
      '--miss_num',
      help='For visualization only, number of sensors to be reduced',
      default=2,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=10000,
      type=int)
  
  args = parser.parse_args() 
  
  # Calls main function
  if args.plot_result == 'yes':
    visualize_result(args)
  else:
    run(args)
  # run_radomization(args)