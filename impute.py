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
from cgi import test
import matplotlib.pyplot as plt
import argparse
import yaml
import os
import sys
import numpy as np

from pgain.pgain import pgain
from libs.pgainutils import rmse_loss, mae_loss
from libs.utils import transform_missing_data, load_and_align_data_with_broken_sensor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold

ROOT_PATH = os.getcwd()
params = yaml.load(open(ROOT_PATH + '/config.yaml'), yaml.Loader)

def main (args):
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
  pearson_hint = True if args.pearson_hint == 'yes' else False
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  
  # np.set_printoptions(threshold=sys.maxsize)

  # Load data and introduce missingness
  data_frame = load_and_align_data_with_broken_sensor(data_type, True, pearson_hint)

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

    # print(train_val_df)
  # sys.exit()
    # Rename to consolidate the name of the sensors
    train_val_df.rename(columns={'response': 'predictor0'}, inplace=True)
    time_axis = train_val_df.index
    ori_data_x, miss_data_x, data_m = transform_missing_data(train_val_df, miss_rate, "predictor0")
    pos = int(params['imputed_position'])

    imputed_pos = int(miss_data_x.shape[0] * (1 - miss_rate))

    # Impute missing data
    imputed_data_x = pgain(miss_data_x, gain_parameters)

    # Report the RMSE performance
    rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
    kfold_rmse_scores.append(rmse)
    print()
    print('RMSE Performance: ' + str(np.round(rmse, 4)))

    rmse_sklearn = mean_squared_error(ori_data_x[imputed_pos:], imputed_data_x[imputed_pos:], squared=False)
    print()
    print('RMSE Sklearn Performance: {} in range min {} and max {}'.format(str(np.round(rmse_sklearn, 4)), np.amin(ori_data_x[imputed_pos:]), np.amax(ori_data_x[imputed_pos:])))

    mae_sklearn = mean_absolute_error(ori_data_x[imputed_pos:, pos], imputed_data_x[imputed_pos:, pos])
    kfold_mae_scores.append(mae_sklearn)
    print()
    print('MAE Sklearn Performance: ' + str(np.round(mae_sklearn, 4)))

    mae_custom = mae_loss(ori_data_x, imputed_data_x, data_m)
    kfold_mae_custom_scores.append(np.round(mae_custom, 4))
    print()
    print('MAE Custom Performance: ' + str(np.round(mae_custom, 4)))

    r2 = r2_score(ori_data_x[imputed_pos:, pos], imputed_data_x[imputed_pos:, pos])
    kfold_r2_scores.append(r2)
    print()
    print('R2 Performance: ' + str(np.round(r2, 4)))

    # plt.rcParams["figure.figsize"] = [7.50, 3.50]
    # plt.rcParams["figure.autolayout"] = True
    # ori_line, = plt.plot(time_axis[imputed_pos:], ori_data_x[imputed_pos:, pos])
    # ori_line.set_label("Original")
    
    # imputed_line, = plt.plot(time_axis[imputed_pos:], imputed_data_x[imputed_pos:, pos], color='red')
    # imputed_line.set_label("Imputed")
    # plt.xticks(rotation=90, fontsize='xx-small')
    # plt.legend(loc="upper right")
    # plt.savefig('images/' + data_type + '_' + str(miss_rate) + '_fold_' + str(fold_number) + '.png', bbox_inches='tight', pad_inches=0, dpi=150)
    # # plt.show()
    # plt.clf()
  print("----------------------------------------------------------")
  print("RMSE avg acc: %.5f (+/- %.5f)\n" % (np.mean(kfold_rmse_scores), np.std(kfold_rmse_scores)))
  print("MAE avg acc: %.2f (+/- %.2f)\n" % (np.mean(kfold_mae_scores), np.std(kfold_mae_scores)))
  print("MAE custom avg acc: %.2f (+/- %.2f)\n" % (np.mean(kfold_mae_custom_scores), np.std(kfold_mae_custom_scores)))
  print("R2 avg acc: %.5f (+/- %.5f)\n" % (np.mean(kfold_r2_scores), np.std(kfold_r2_scores)))
    
  return imputed_data_x, rmse

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
      '--pearson_hint',
      help='pearson hint',
      default='no',
      type=str)
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
  imputed_data, rmse = main(args)
