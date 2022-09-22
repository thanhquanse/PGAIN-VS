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
import sys
from turtle import color
import matplotlib.pyplot as plt
import argparse
import yaml
import os
import numpy as np

from pgain.pgain import pgain
from libs.pgainutils import rmse_loss
from libs.utils import transform_missing_data, load_and_align_data_with_broken_sensor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
  _metrics = args.metric
  metrics = _metrics.split(',')

  miss_rate_arr = [10, 20, 30, 40, 50, 60, 70, 80, 90]
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  
  # np.set_printoptions(threshold=sys.maxsize)

  # Load data and introduce missingness
  data_frame = load_and_align_data_with_broken_sensor(data_type, True)
  
  if data_frame is None:
    print("Error: No correlative sensors found. Exit!")
    sys.exit()
  
  # Rename to consolidate the name of the sensors
  data_frame.rename(columns={'response': 'predictor0'}, inplace=True)

  rmse_result_arr = []
  mae_result_arr = []
  r2_result_arr = []

  # Respectively check on missing rate
  for miss_rate in miss_rate_arr:
    ori_data_x, miss_data_x, data_m = transform_missing_data(data_frame, round(miss_rate / 100.0, 2), "predictor0")
    pos = int(params['imputed_position'])

    imputed_pos = int(miss_data_x.shape[0] * (1 - round(miss_rate / 100.0, 2)))

    # Impute missing data
    imputed_data_x = pgain(miss_data_x, gain_parameters)

    for metric in metrics:
      if metric == 'rmse':
        # Report the RMSE performance
        rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
        print()
        print('Missing rate: {} - RMSE Performance: {}'.format(str(miss_rate), str(np.round(rmse, 4))))

        rmse_sklearn = mean_squared_error(ori_data_x[:, pos], imputed_data_x[:, pos], squared=False)
        rmse_result_arr.append(rmse)
        print()
        print('Missing rate: {} - RMSE sklearn Performance: {}'.format(str(miss_rate), str(np.round(rmse_sklearn, 4))))
      elif metric == 'mae':
        # Report the MAE performance
        mae_sklearn = mean_absolute_error(ori_data_x[:, pos], imputed_data_x[:, pos])
        mae_result_arr.append(mae_sklearn)
        print()
        print('Missing rate: {} - MAE Sklearn Performance: {}'.format(str(miss_rate), str(np.round(mae_sklearn, 4))))
      elif metric == 'r2':
        # Report the R2 performance
        r2 = r2_score(ori_data_x[:, pos], imputed_data_x[:, pos])
        r2_result_arr.append(r2)
        print()
        print('Missing rate: {} - R2 Performance: {}'.format(str(miss_rate), str(np.round(r2, 4))))
      else:
        print("Metric {} is not supported.".format(metric))

  if len(rmse_result_arr) != 0:
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    rmse_line, = plt.plot(miss_rate_arr, rmse_result_arr)
    rmse_line.set_label("NRMSE")

    plt.xticks(miss_rate_arr)
    plt.legend(loc="upper right")
    plt.savefig("images/" + data_type + "_rmse.png", dpi=150)
    
    plt.clf()

  if len(mae_result_arr) != 0:
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    mae_line, = plt.plot(miss_rate_arr, mae_result_arr, color='red')
    mae_line.set_label("MAE")

    plt.xticks(miss_rate_arr)
    plt.legend(loc="upper right")
    plt.savefig("images/" + data_type + "_mae.png", dpi=150)   

    plt.clf() 

  if len(r2_result_arr) != 0:
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    r2_line, = plt.plot(miss_rate_arr, r2_result_arr, color='black')
    r2_line.set_label("R2")

    plt.xticks(miss_rate_arr)
    plt.legend(loc="upper right")
    plt.savefig("images/" + data_type + "_r2.png", dpi=150)

    plt.clf()

  # plt.xticks(miss_rate_arr)
  # plt.legend(loc="upper right")
  # plt.show()


if __name__ == '__main__':  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_type',
      choices=['solar','temperature', 'raspihat', 'traffic'],
      default='solar',
      type=str)
  parser.add_argument(
      '--metric',
      help='Metric to evalute (RMSE/R2/MAE)',
      default='rmse',
      type=str)
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
  main(args)