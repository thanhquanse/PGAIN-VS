from collections import defaultdict
import dateutil
import glob
import json
import csv
import os
import pickle
import pandas as pd
import numpy as np
import yaml

from functools import reduce
from entities.replier import Replier
from entities.sensor import Sensor
from libs.pgainutils import binary_sampler

temperature_data_file = "resources/temperature.json"
solar_power_data_file = "resources/solar_power.json"
traffic_speed_data_file = "resources/traffic_speed.json"
raspihat_data_file = "resources/raspihat_temperature.json"
broken_temperature_sensor_id = "resources/broken_temperature_sensor.json"
broken_solar_sensor_id = "resources/broken_solar_sensor.json"
broken_traffic_sensor_id = "resources/broken_traffic_sensor.json"
broken_raspihat_sensor_id = "resources/broken_raspihat_sensor.json"

# ROOT_PATH = os.path.dirname(sys.modules['__main__'].__file__)
ROOT_PATH = os.getcwd()
params = yaml.load(open(ROOT_PATH + '/config.yaml'), yaml.Loader)

def load_temperature_data():
    with open(temperature_data_file, 'r') as file:
        data = file.read()
    obj_json = json.loads(data)

    return obj_json

def load_solar_power_data():
    with open(solar_power_data_file, 'r') as file:
        data = file.read()
    obj_json = json.loads(data)

    return obj_json

def load_traffic_data():
    with open(traffic_speed_data_file, 'r') as file:
        data = file.read()
    obj_json = json.loads(data)

    return obj_json

def load_raspihat_temperature_data():
    with open(raspihat_data_file, 'r') as file:
        data = file.read()
    obj_json = json.loads(data)

    return obj_json

def load_broken_temperature_id():
    with open(broken_temperature_sensor_id, 'r') as file:
        data = file.read()
    obj_json = json.loads(data)

    return obj_json

def load_broken_solar_id():
    with open(broken_solar_sensor_id, 'r') as file:
        data = file.read()
    obj_json = json.loads(data)

    return obj_json

def load_broken_traffic_id():
    with open(broken_traffic_sensor_id, 'r') as file:
        data = file.read()
    obj_json = json.loads(data)

    return obj_json

def load_broken_raspihat_id():
    with open(broken_raspihat_sensor_id, 'r') as file:
        data = file.read()
    obj_json = json.loads(data)

    return obj_json

def load_traing_data(sensorID, dropSeconds=False, limit_row=None):
    if not type(sensorID) == list:
        sensorID = [sensorID]
    sensorIDNames = list(map(lambda x: x.replace(':', '_'), sensorID))
    result = {}
    for i in range(0, len(sensorIDNames)):
        path = ROOT_PATH + "/data/*/" + sensorIDNames[i] + ".csv"
        fNames = glob.glob(path, recursive=True)

        if not fNames or len(fNames) == 0:
            continue

        data = []
        for fName in fNames:
            csvFile = csv.DictReader(open(fName, "r"))
            fieldnames = csvFile.fieldnames
            first = False
            for idx, line in enumerate(csvFile):
                if first:
                    first = False
                    continue
                dt = dateutil.parser.isoparse(line[fieldnames[0]])
                if dropSeconds:
                    dt = dt.replace(second=0)
                data.append({"timestampStr": dt.isoformat(), "timestamp": dt, "value": float(line[fieldnames[1]])})

                if limit_row is not None:
                    if idx >= limit_row:
                        break
            print("Loaded historic data from: {} with {} samples".format(os.path.basename(fName), len(data)))

        result[sensorID[i]] = data

    return result

def align_data(brokenSensorID, sensors, drop_nan=False):
    if not brokenSensorID in sensors:
        return Replier("error", description="no meta data for broken sensor")
    brokenSensor = sensors[brokenSensorID]

    td = brokenSensor.training_data
    indices = list(map(lambda x: x['timestamp'], td))
    df = pd.DataFrame(td, index=indices)

    df.rename(columns={'value': 'response'}, inplace=True)
    df.drop(columns=['timestamp', 'timestampStr'], inplace=True)

    x = 1
    predictorMap = {}
    df.index.name = 'Index'
    # print("DF shape: ", df.shape)
    # df = df[~df.index.duplicated(keep='last')]
    # idx = df.index
    # print(df[idx.isin(idx[idx.duplicated()])])
    # print("DF duplicate rows: ", (~df.index.duplicated(keep='last')).sum())
    try:
        for i in sensors:
            currentSensor = sensors[i]
            if currentSensor.get_id() == brokenSensorID or not currentSensor.training_data: continue
            # print(currentSensor.get_id(), "has", len(currentSensor.training_data), "samples of training data")
            indices = list(map(lambda x: x['timestamp'], currentSensor.training_data))

            preditorName = 'predictor' + str(x)

            predictorMap[preditorName] = currentSensor.get_id()
            df2 = pd.DataFrame(currentSensor.training_data, index=indices)

            df2.rename(columns={'value': preditorName}, inplace=True)
            df2.drop(columns=['timestamp', 'timestampStr'], inplace=True)
            df2.index.name = 'Index'
            df2.index.drop_duplicates(keep='last')

            if drop_nan:
                df2.dropna(how='any', inplace=True)

            l,w = df2.shape
            if l >= 100:
                df = df.join(df2, how='outer')
                df = df[~df.index.duplicated(keep='last')]
            x += 1

        if drop_nan:
            df.dropna(how='any', inplace=True)
    except MemoryError:
        return Replier("error", description="not enough memory to align training data")

    return Replier(data={'df': df, 'predictorMap': predictorMap})

def load_pickle(file):
    try:
        pickle_in= open(file, "rb")
        data = pickle.load(pickle_in)
        pickle_in.close()
        return data
    except IOError:
        print("Error loading pickle data from " + file)
        
    return None

def write_pickle(data, file):    
    try:
        pickle_out = open(file, "wb")
        pickle.dump(data, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_out.close()
        return True
    except IOError:
        print("Error writing data to " + file)
        
    return False

def convert_to_nparray(df, corr_ids):
    y = np.array(df.response)
    temporary = []
    for i in range(0,len(df.columns)):
        if df.columns[i] in corr_ids:
            temporary.append(np.array(df.loc[:,df.columns[i]]))
    x = np.stack(temporary).T
    return x,y

def create_corr_df(df):
    df_dest = pd.DataFrame(columns=['from', 'to', 'cost'])
    for col1 in list(df):
        for col2 in list(df):
            corr = abs(np.corrcoef(df[col1], df[col2])[0][1])
            row = {'from': col1, 'to': col2, 'cost': corr}
            df_dest = df_dest.append(row, ignore_index=True)
    return df_dest

def corr_df_to_csv(df):
    df.to_csv("data/distance.csv", index=False)

def filterByCorrelation(df):
    """
    input: df = panda DataFrame as returned by align data
    """
    faulty_sensor = df.response
    candidates = df.iloc[:, df.columns != 'response']
    corr_array, corr_ids = correlations(faulty_sensor, candidates)
    return corr_ids

def correlations(faulty_sensor, candidates):   #candidates is the list of potential sensors
    small_data = params['small_data']          #30000                #small/big variables to select a limited number of sensors
    less_sensors_less_data = params['less_sensors_less_data']
    more_sensors = params['more_sensors']
    less_sensors = params['less_sensors']
    high_corr_limit = params['high_corr_limit']       #0.85               # correlation partitions
    med_corr_limit = params['med_corr_limit']
    high_corr_sensors, med_corr_sensors, low_corr_sensors = [], [], []
    high_corr, med_corr, low_corr = [], [], []
    lim_corr_array = []                        #lim_corr_array contains the limited number of high correlation sensors

    sensor_and_corr = []

    faulty_samples = len(faulty_sensor)

    # Find correlations and divide in categories
    for i in candidates:
        sensor_corr = abs(np.corrcoef(faulty_sensor,candidates[i])[0][1])
        print("{}".format(str(sensor_corr)))
        if sensor_corr >= high_corr_limit:
            high_corr_sensors.append(candidates[i])
            high_corr.append(sensor_corr)

            sensor_and_corr.append((candidates[i].name, sensor_corr))
        elif sensor_corr < high_corr_limit and sensor_corr >= med_corr_limit:
            med_corr_sensors.append(candidates[i])
            med_corr.append(sensor_corr)

            # sensor_and_corr.append((candidates[i].name, sensor_corr))
        else:
            low_corr_sensors.append(candidates[i])
            low_corr.append(sensor_corr)
    
    # Use limited number of sensors in case of many available
    if faulty_samples <= small_data and candidates.shape[1] > more_sensors:
        if len(high_corr) > more_sensors:
            print("Case 1 - More high correlated sensors found")
            high_corr_sorted = sorted(high_corr)           #Debug - can be removed
            sorted_index = sorted(range(len(high_corr)), key=lambda k: high_corr[k], reverse=True)
            lim_corr_array = [high_corr_sensors[i] for i in sorted_index[0:more_sensors]]

    elif faulty_samples <= small_data and len(high_corr) < less_sensors_less_data: #increase number of sensors if less sensors with high_corr
        print("Case 2 - Combine high and medium correlated sensors")
        high_med_corr = high_corr_sensors + med_corr_sensors
        high_sorted_index = sorted(range(len(high_corr)), key=lambda k: high_corr[k], reverse=True)
        med_sorted_index = sorted(range(len(med_corr)), key=lambda k: med_corr[k], reverse=True)
        high_lim_corr_array = [high_corr_sensors[i] for i in high_sorted_index]
        med_lim_corr_array = [med_corr_sensors[i] for i in med_sorted_index]

        lim_corr_array = high_lim_corr_array + med_lim_corr_array
        # lim_corr_array = high_corr_sensors + med_corr_sensors
    
    elif faulty_samples > small_data and candidates.shape[1] > less_sensors:
        if len(high_corr) > less_sensors:
            print("Case 3 - Take high correlated sensors because faulty samples is high")
            sorted_index = sorted(range(len(high_corr)), key=lambda k: high_corr[k], reverse=True)
            lim_corr_array = [high_corr_sensors[i] for i in sorted_index[0:less_sensors]]
    else:
        print("Case 4 - The last case")
        sorted_index = sorted(range(len(high_corr)), key=lambda k: high_corr[k], reverse=True)
        lim_corr_array = [high_corr_sensors[i] for i in sorted_index]
        # return high_corr_sensors, [high_corr_sensors[i].name for i in range(0,len(high_corr_sensors))] #, high_corr_ids
    
    # lim_corr_array = high_corr_sensors

    return lim_corr_array, [lim_corr_array[i].name for i in range(0,len(lim_corr_array))]#, corr_ids

def calulate_pearson_corrs(data_frame):
  sensor_map_pearson_ranks = {}
  all_sensors = list(data_frame)
  for idx, sens in enumerate(all_sensors):
    sens_values = data_frame[sens]
    remaining_sensors = sorted(list(set(all_sensors) - set([sens])))

    pearson_corrs = []

    for remaining_sens in remaining_sensors:
      remaining_sens_values = data_frame[remaining_sens]
      # pearson_corrs.append(1)
    #   pearson_corrs.append(np.abs(np.corrcoef(sens_values, remaining_sens_values)[0][1]))
      pearson_corrs.append(1./ (1. + abs(np.corrcoef(sens_values, remaining_sens_values)[0, 1])))

    sensor_map_pearson_ranks[sens] = [(x,y) for y,x in sorted(zip(pearson_corrs, remaining_sensors), reverse=True)]

  return sensor_map_pearson_ranks

def transform_missing_data(data_frame, missing_rate=0.3, predicted="response"):
  print()
  print("{} is selected to predict with missing rate: {} ...".format(predicted, str(missing_rate)))

  _miss_data_x = data_frame.copy()
  data_x = data_frame.to_numpy(dtype=np.float)

  # Set most of the values of predicted sensors to be nan to impute again
  # Calculate none missing size
  none_missing_size = int(_miss_data_x.shape[0] * (1 - missing_rate))
  _miss_data_x.loc[none_missing_size:, predicted] = np.nan
  miss_data_x = _miss_data_x.to_numpy(dtype=np.float)

  _data_m = miss_data_x.copy()
  _data_m_temp = np.where(~np.isnan(_data_m), 1, _data_m)
  data_m = np.where(np.isnan(_data_m_temp), 0, _data_m_temp)

  print()
  print("Total: {}".format(data_x.shape))
  print("Number of missing values: {}".format(np.count_nonzero(np.isnan(miss_data_x))))

  return data_x, miss_data_x, data_m

def load_resources(type):
    # brokenSensorID = "urn:ngsi-ld:Sensor:SolarPowerAarhus:488:currentProduction"
    # brokenSensorID = "urn:ngsi-ld:Sensor:SolarPowerAarhus:487:currentProduction"
    # brokenSensorID = "urn:ngsi-ld:Sensor:SolarPowerAarhus:486:currentProduction"
    # brokenSensorID = "urn:ngsi-ld:Sensor:SolarPowerAarhus:485:currentProduction"
    # brokenSensorID = "urn:ngsi-ld:Sensor:SolarPowerAarhus:481:currentProduction"
    # brokenSensorID = "urn:ngsi-ld:Sensor:SolarPowerAarhus:478:currentProduction"
    # brokenSensorID = "urn:ngsi-ld:Sensor:SolarPowerAarhus:477:currentProduction"

    # brokenSensorID = "urn_ngsi-ld_Sensor_RomanianWeather_IASI_temperature"
    # brokenSensorID = "urn_ngsi-ld_Sensor_RomanianWeather_CRAIOVA_temperature" ###
    # brokenSensorID = "urn_ngsi-ld_Sensor_RomanianWeather_CLUJ-NAPOCA_temperature"
    # brokenSensorID = "urn_ngsi-ld_Sensor_RomanianWeather_BRASOV_temperature"
    # brokenSensorID = "urn_ngsi-ld_Sensor_RomanianWeather_BRAILA_temperature"
    # brokenSensorID = "urn_ngsi-ld_Sensor_RomanianWeather_BARLAD_temperature"

    data = brokenSensorID = None

    if type == 'solar':
        data = load_solar_power_data()
        brokenSensorID = load_broken_solar_id()
    elif type == 'traffic':
        data = load_traffic_data()
        brokenSensorID = load_broken_traffic_id()
    elif type == 'temperature':
        data = load_temperature_data()
        brokenSensorID = load_broken_temperature_id()
    elif type == 'raspihat':
        data = load_raspihat_temperature_data()
        brokenSensorID = load_broken_raspihat_id()
    else:
        print("Error: {} data is not supported yet".format(type))
    
    return data, brokenSensorID

def load_and_align_data_with_broken_sensor(type='solar', drop_nan=False, pearson_hint=True):
    data, brokenSensorID = load_resources(type)
    sensorMap = {}
    counter = 0

    for s in data[:50]:
        sensor = Sensor(s)
        id = sensor.get_id()
        training_data = load_traing_data(id, True, params['sample_num'])

        if training_data:
            sensor.training_data = training_data[id]
            sensorMap[sensor.get_id()] = sensor
            if sensor.get_id() != brokenSensorID['id']:
                counter += 1

    if counter < 2:
        print("not enough training data found")

    alignedDataReply = align_data(brokenSensorID['id'], sensorMap, drop_nan)
    if alignedDataReply.success:
        df = alignedDataReply.data['df']
        if not pearson_hint:
            return df
        try:
            corr_ids = filterByCorrelation(df)
        except MemoryError:
            print("Not enough memory")
            Replier("error", "not enough memory to find correlations. Try to limit the number of sensor sources (limitSourceSensors).").answer()
        if len(corr_ids) == 0:
            print("No correlating sensor")
            Replier("error", description="no correlating sensor found").answer()
        else:
            ### Arrange imputed sensor to be the first column ###
            corr_ids.insert(int(params['imputed_position']), 'response')
            # _corr_ids = ['predictor9', 'predictor2', 'predictor15', 'response', 'predictor12', 'predictor5', 'predictor4', 'predictor14']
            print("List of corr_ids: {}".format(corr_ids))
            df = df.reindex(columns=corr_ids)
    
            return df
    else:
        print("align data failed")
    
    return None

def load_full_data(type, drop_nan=False):
    data, brokenSensorID = load_resources(type)
    sensorMap = {}
    counter = 0

    for s in data[:50]:
        sensor = Sensor(s)
        id = sensor.get_id()
        training_data = load_traing_data(id, False, params['sample_num'])

        if training_data:
            sensor.training_data = training_data[id]
            sensorMap[sensor.get_id()] = sensor
            if sensor.get_id() != brokenSensorID['id']:
                counter += 1

    if counter < 2:
        print("not enough training data found")

    alignedDataReply = align_data(brokenSensorID['id'], sensorMap, drop_nan)
    if alignedDataReply.success:
        df = alignedDataReply.data['df']
        return df
    else:
        print("align data failed")
    
    return None

def verify_subset(type, aligned_data, best_subset_col=[], predicted='response'):
    if predicted not in best_subset_col:
         best_subset_col.insert(0, predicted)
    if aligned_data is not None:
        df = aligned_data
        df = df.reindex(columns=best_subset_col)
        try:
            df.rename(columns={predicted: 'response'}, inplace=True)
            corr_ids = filterByCorrelation(df)
        except MemoryError:
            print("Not enough memory")
            Replier("error", "not enough memory to find correlations. Try to limit the number of sensor sources (limitSourceSensors).").answer()
        if len(corr_ids) == 0:
            print("No correlating sensor")
            Replier("error", description="no correlating sensor found").answer()
        else:
            ### Arrange imputed sensor to be the first column ###
            corr_ids.insert(params['imputed_position'], 'response')
            # print("List of corr_ids: {}".format(corr_ids))
            df = df.reindex(columns=corr_ids)

            return df
    else:
        print("Error: Aligned data not found")
    
    return None

def randomize_missing(dataframe, miss_rate):
    data_x = dataframe.to_numpy(dtype=np.float)

    # Parameters
    no, dim = data_x.shape
    
    # Introduce missing data
    data_m = binary_sampler(1-miss_rate, no, dim)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan
        
    return data_x, miss_data_x, data_m

def rotate_sensors(dataframe, start_idx=0, miss_number=1, miss_rate=0.2):
    sensors = list(dataframe)
    _miss_data_x = dataframe.copy()
    data_x = dataframe.to_numpy(dtype=np.float)
    total_samples = _miss_data_x.shape[0]

    number_of_missing = int(total_samples * miss_rate)
    sub_sensors = []

    # start_idx = 0
    end_idx = start_idx + (2 * miss_number)

    # sensors_num = len(sensors[0:(2 * params['miss_sensor_num'])])
    # sensors_num = len(sensors[start_idx:end_idx])
    sensors_num = len(sensors)
    # sub_sensors_num = (sensors_num // params['miss_sensor_num'])
    start = start_idx
    end = 0
    pivot = 0

    while pivot < total_samples:
        pivot = pivot + number_of_missing
        if start == end_idx - 1:
            start = start_idx - 1
            # break
        if end == end_idx - 1:
            start = start_idx
            # break
        end = start + miss_number - 1

        sensor_arr = []
        for x in range(start, (end + 1)):
            # print("Sensor: {}".format(x))
            sensor_arr.append(sensors[x % sensors_num])
            
        sub_sensors.append(sensor_arr)
        start = end + 1

    # print("Pivot: {}".format(pivot))
    # for i in range((len(sensors) // params['miss_sensor_num']) + 1):
    #     start = i * params['miss_sensor_num']
    #     end = (i + 1) * params['miss_sensor_num']
    #     sub_sensors.append(sensors[start:end])

    # print(sub_sensors)
    segment = total_samples // (number_of_missing)
    # print("Segment: {}".format(segment))
    # print(sensors)
    print("Start IDX: {}".format(start_idx))
    print("Number of sensors missing: {}".format(miss_number))
    print("Number of row missing: {}".format(number_of_missing))

    i = 0
    start = 0
    end = (i + 1) * number_of_missing

    j = 0
    while end < total_samples:
        for sub_sen in sub_sensors:
            for sen in sub_sen:
                _miss_data_x.loc[start:end, sen] = np.nan
        
            start = end
            end = end + number_of_missing
    
    miss_data_x = _miss_data_x.to_numpy(dtype=np.float)

    # print(_miss_data_x.to_string())
    # _miss_data_x.to_csv("rotation_measurement.csv", index=None)

    _data_m = miss_data_x.copy()
    _data_m_temp = np.where(~np.isnan(_data_m), 1, _data_m)
    data_m = np.where(np.isnan(_data_m_temp), 0, _data_m_temp)

    return data_x, miss_data_x, data_m

def rotate_single_sensor(dataframe, start_idx=0, miss_number=1, miss_rate=0.2):
    sensors = list(dataframe)
    _miss_data_x = dataframe.copy()
    data_x = dataframe.to_numpy(dtype=np.float)
    total_samples = _miss_data_x.shape[0]

    number_of_missing = int(total_samples * miss_rate)
    sub_sensors = []

    sensors_num = len(sensors)

    start = start_idx
    end = 0
    pivot = 0

    while pivot < total_samples:
        pivot = pivot + number_of_missing
        end = start + miss_number - 1

        # sensor_arr = []
        for x in range(start, (end + 1)):
            # idx = x % sensors_num
            # print("Sensor: {} - {}".format(idx, sensors_num))
            # sensor_arr.append(sensors[x % sensors_num])
            sub_sensors.append(sensors[x % sensors_num])
            
        # sub_sensors.append(sensor_arr)
        start = start_idx

    # print(sub_sensors)
    print("Start from: {}".format(start_idx))
    print("Number of sensors missing: {}".format(miss_number))
    print("Number of row missing: {}".format(number_of_missing))
    print()

    j = 0
    start = 0
    end = number_of_missing
    sensor_pos_map = defaultdict(list)
    while end < total_samples:
        end = end + number_of_missing
        print("Rotation on {} - from {} - to {}".format(sub_sensors[j], start, end))
        sensor_pos_map[sub_sensors[j]].append([start, end])

        _miss_data_x.loc[start:end, sub_sensors[j]] = np.nan
        start = end
        j = j + 1
    
    miss_data_x = _miss_data_x.to_numpy(dtype=np.float)

    # print(_miss_data_x.to_string())
    # _miss_data_x.to_csv("rotation_measurement.csv", index=None)

    _data_m = miss_data_x.copy()
    _data_m_temp = np.where(~np.isnan(_data_m), 1, _data_m)
    data_m = np.where(np.isnan(_data_m_temp), 0, _data_m_temp)

    return data_x, miss_data_x, data_m, sensor_pos_map

def show_data_info(dataframe):
    print("Min: {}".format(np.amin(dataframe)))
    print("Max: {}".format(np.amax(dataframe)))
    print("Mean: {}".format(np.mean(dataframe)))
    print("STD: {}".format(np.std(dataframe)))
    print("25%: {}".format(np.percentile(dataframe, 25)))
    print("50%: {}".format(np.percentile(dataframe, 50)))
    print("75%: {}".format(np.percentile(dataframe, 75)))