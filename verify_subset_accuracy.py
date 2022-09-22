# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import yaml
import os

from pgain.pgain import pgain
from libs.pgainutils import rmse_loss
from libs.utils import transform_missing_data, load_full_data, verify_subset

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
params = yaml.load(open(ROOT_PATH + '/config.yaml'), yaml.Loader)

def main(args):
    gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
    
    data_type = args.data_type
    miss_rate = args.miss_rate

    # Get list of sensors
    data_frame = load_full_data(data_type, True)
    time_axis = data_frame.index

    # Rename to consolidate the name of the sensors
    data_frame.rename(columns={'response': 'predictor0'}, inplace=True)

    # Get all sensors
    sensors = list(data_frame)
    pos = params['imputed_position']
    rmse_arr = []
    mae_arr = []
    r2_arr = []
    subset_str = params['subset']
    subset_list = subset_str.split(',')
    result_df = pd.DataFrame(columns=['Sensor', 'RMSE', 'MAE', 'R2'])

    for sensor in sensors:
        # Find and align sub-subset of sensor
        best_subset_df = verify_subset(data_type, data_frame, subset_list, sensor)
        # Transform data
        ori_data_x, miss_data_x, data_m = transform_missing_data(best_subset_df, miss_rate, 'response')

        # Start imputing
        imputed_data_x = pgain(miss_data_x, gain_parameters)

        imputed_pos = int(imputed_data_x.shape[0] * (1 - miss_rate))

        # Normalized RMSE
        rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
        rmse_sklearn = mean_squared_error(ori_data_x[imputed_pos:], imputed_data_x[imputed_pos:], squared=False)
        rmse_arr.append(np.round(rmse_sklearn, 4))

        # MAE
        mae = mean_absolute_error(ori_data_x[imputed_pos:, pos], imputed_data_x[imputed_pos:, pos])
        mae_arr.append(np.round(mae, 4))

        # R2
        r2 = r2_score(ori_data_x[imputed_pos:, pos], imputed_data_x[imputed_pos:, pos])
        r2_arr.append(np.round(r2, 4))

        new_row = pd.DataFrame()
        new_row['Sensor'] = [sensor]
        new_row['RMSE'] = [rmse_sklearn]
        new_row['MAE'] = [mae]
        new_row['R2'] = [r2]
        result_df = result_df.append(new_row)

        plt.rcParams["figure.figsize"] = [7.5, 3.5]
        plt.rcParams["figure.autolayout"] = True

        ori_line, = plt.plot(time_axis[imputed_pos:], ori_data_x[imputed_pos:, pos])
        ori_line.set_label("Original")

        imputed_line, = plt.plot(time_axis[imputed_pos:], imputed_data_x[imputed_pos:, pos], color='red')
        imputed_line.set_label("Imputed")

        plt.xticks(time_axis)
        plt.legend(loc="upper right")
        plt.savefig('images/' + data_type + '_' + str(miss_rate) + '_' + sensor + '.png', bbox_inches='tight', pad_inches=0, dpi=150)
        plt.clf()

    print("RMSE list:")
    print(', '.join([str(x) for x in rmse_arr]))
    print()

    print("MAE list:")
    print(', '.join([str(x) for x in mae_arr]))
    print()

    print("R2 list:")
    print(', '.join([str(x) for x in r2_arr]))

    result_df.to_csv('images/subset_' + data_type + '_verification_result.csv', index=False)

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