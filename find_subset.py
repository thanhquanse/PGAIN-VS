# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from operator import index
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import sys

from pgain.pgain import pgain
from libs.pgainutils import normalization
from libs.utils import transform_missing_data, load_full_data, calulate_pearson_corrs
from kneed import KneeLocator

ROOT_PATH = os.path.dirname(sys.modules['__main__'].__file__)
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
    # data_frame.to_csv("knockoff_ds_" + data_type + ".csv", index=None)
    # print(data_frame)
    # Get faulty sensor dataset info
    # print("***The faulty sensor dataset info***")
    # print("\nMean: ", data_frame['response'].mean())
    # print("\nSTD: ", data_frame['response'].std())
    # print("\nMin: ", data_frame['response'].min())
    # print("\nMax: ", data_frame['response'].max())
    # print("\n25%: ", data_frame['response'].quantile(0.25))
    # print("\n50%: ", data_frame['response'].quantile(0.50))
    # print("\n75%: ", data_frame['response'].quantile(0.75))

    # Get the whole dataset info
    # print("***The whole dataset info***")
    # print("\nMean: ", data_frame.values().mean())
    # print("\nSTD: ", data_frame.values().std())
    # print("\nMin: ", data_frame.values().min())
    # print("\nMax: ", data_frame.values().max())
    # print("\n25%: ", data_frame.values().quantile(0.25))
    # print("\n50%: ", data_frame.values().quantile(0.50))
    # print("\n75%: ", data_frame.values().quantile(0.75))

    # sys.exit()

    # Rename to consolidate the name of the sensors
    data_frame.rename(columns={'response': 'predictor0'}, inplace=True)

    # Loop to calculate imputed data
    results_k_sens = pd.DataFrame(columns=['rank', 'predicted_sensor', 'n_discarded_sensors', 'ecdf_area_95th'])
    all_sensors = list(data_frame)
    sensor_map_pearson_ranks = calulate_pearson_corrs(data_frame)

    ranks = {
        'pearson':sensor_map_pearson_ranks
    }

    rank_labels={
        'pearson': 'Pearson'
    }

    less_corr_sensors = []

    # In loop, calculate error
    for idx, sens in enumerate(all_sensors):
        sensor_corrs = sensor_map_pearson_ranks[sens]
        sensor_corrs = sorted(sensor_corrs, key=lambda x: x[1], reverse=False) # in this way, we have the sensors from the worst to the best
        sensor_corrs_desc = sorted(sensor_corrs, key=lambda x: x[1], reverse=True)
        corr_sensors = [x[0] for x in sensor_corrs_desc]
        # list_corr_sensors = ','.join(corr_sensors)
        corr_values = [x[1] for x in sensor_corrs if x[1] > 0.6]

        df = data_frame.copy()
        df = df.reindex(columns=corr_sensors)

        # if len(corr_values) < 8:
        #     print("Less correlated sensors: {}, skipping...".format(idx))
        #     less_corr_sensors.append(idx)
        #     continue

        position = 0

        # Rearrange to predict
        predicted_column = data_frame[sens]
        df.insert(position, sens, predicted_column)

        for k in range(position, len(all_sensors) - position - 1):
            _data_frame = df.iloc[: , :-k]
            
            if k == 0:
                _data_frame = df.copy()

            # print(_data_frame)
            
            ori_data_x, miss_data_x, data_m = transform_missing_data(_data_frame, miss_rate, sens)
            imputed_pos = int(ori_data_x.shape[0] * (1 - miss_rate))

            if len(corr_values) >= params['less_sensors']:
                # Impute missing data
                imputed_data_x = pgain(miss_data_x, gain_parameters)

                ori_data, norm_parameters = normalization(ori_data_x)
                imputed_data, _ = normalization(imputed_data_x, norm_parameters)

                # Extract only predicted sensor
                ori_predicted_col = [col[position] for col in ori_data]
                imputed_data_col = [col[position] for col in imputed_data]

                print("Verifying errors...")
                print(np.abs(np.asarray(imputed_data_col[imputed_pos:]) - np.asarray(imputed_data_col[imputed_pos:])))

                # Calculate 95 error
                errors = np.abs(np.asarray(ori_predicted_col[imputed_pos:]) - np.asarray(imputed_data_col[imputed_pos:]))
            
            else:
                print("Less correlated sensors: {}, highest error by default...".format(sens))
                errors = np.ones((ori_data_x.shape[0] - imputed_pos))

            # print("Errors: {}".format(errors))
            # errors = rmse_loss(ori_data_x, imputed_data_x, data_m)
            ecdf_x = np.sort(errors)
            ecdf_y = np.arange(1, len(ecdf_x) ) / len(ecdf_x)
            perc_95 = int(len(ecdf_x)*0.95)

            ecdf_area_95 = np.trapz(ecdf_y[:perc_95], ecdf_x[:perc_95])

            new_row = pd.DataFrame()
            new_row['rank'] = ['pearson']
            new_row['predicted_sensor'] = [sens]
            new_row['n_discarded_sensors'] = [k]
            new_row['ecdf_area_95th'] = [ecdf_area_95]
            results_k_sens = results_k_sens.append(new_row)

    # Start drawing plot
    plt.figure(figsize=(20, 8))
    plot_string = "Area under the curve: \n"

    ranks_errors_list = []
    for key in ranks:
        subresults = results_k_sens.query("rank == @key")
        grouped = subresults.groupby(['n_discarded_sensors']).sum()
        auc = round(np.trapz(grouped['ecdf_area_95th'], list(range(0, len(grouped['ecdf_area_95th'])))), 2)
        ranks_errors_list += [(auc, key)]

    ranks_errors_list = sorted(ranks_errors_list, reverse=False)
    maxval = -1

    for key in [ k for (_,k) in ranks_errors_list ]:
        subresults = results_k_sens.query("rank == @key")
        grouped = subresults.groupby(['n_discarded_sensors']).sum()

        maxval = np.max(grouped['ecdf_area_95th']) if np.max(grouped['ecdf_area_95th']) > maxval else maxval

        plt.plot(range(1, len(grouped['ecdf_area_95th']) + 1), grouped['ecdf_area_95th'].tolist()[::-1], label=rank_labels[key])
        
        plot_string += "  > " + rank_labels[key] + ": " + str(round(np.trapz(grouped['ecdf_area_95th'], list(range(0, len(grouped['ecdf_area_95th'])))), 2)) + "\n"
    
    plt.xlabel("Number of selected sensors (from best to worst according to the rank)")
    plt.ylabel("Sum of 95th percentile error: {}".format(data_type))
    plt.legend(title="Ranking strategy")
    plt.text(x=7.2, y=maxval-1.79, s=plot_string)
    plt.xticks(np.arange(1, len(grouped['ecdf_area_95th']) + 1, 1.0))
    plt.savefig('Number_of_sensors_selected_' + data_type + '.pdf', bbox_inches='tight', pad_inches=0, dpi=1000)
    # plt.show()

    # Borda vorting
    results_best_sens = {}
    for s in all_sensors:
        results_best_sens[s] = 0

    n_refs = len(all_sensors)
    distance = 'pearson'
    rank = ranks[distance]
    epsilon = np.finfo('float').eps
    sensors_error = results_k_sens.query('rank==@distance').groupby('predicted_sensor')[['ecdf_area_95th']].sum().values
    sensors_weight = np.subtract(1, np.divide(sensors_error, np.max(sensors_error+epsilon)))

    for i, sens in enumerate(all_sensors):
        sensors_rank = rank[sens]
        sensors_rank = sorted(sensors_rank, key=lambda x: x[1], reverse=False) # in this way, we have the sensors from the best to the worst

        sensors_rank = [x for (x, _) in sensors_rank[0:n_refs]]

        # first j sensors in the rank
        for j in range(n_refs - 1):
            if j in less_corr_sensors:
                continue
            results_best_sens[sensors_rank[j]]+=(n_refs-j)*sensors_weight[i][0]

    plt.figure(figsize=(14,4))

    results_best_sens=sorted([x for x in zip(results_best_sens.keys(), results_best_sens.values())], key=lambda pair: pair[1], reverse=True)

    keys=[x[0] for x in results_best_sens]
    values=[x[1] for x in results_best_sens]


    kn = KneeLocator(range(len(values)), values, curve='convex', direction='decreasing')
    print('elbow:', kn.knee + 1)

    plt.bar(range(len(values)), height=values)
    plt.plot(range(len(values)), values, color='b')

    plt.xticks(range(len(keys)), keys, rotation=35, ha="right", rotation_mode="anchor")
    plt.ylabel('Weighted Borda count vote')# 'Borda count vote considering '+str(n_refs)+' positions'

    plt.savefig('Best_ranked_sensors_' + data_type + '.pdf', dpi=1000, bbox_inches='tight', pad_inches=0)
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
