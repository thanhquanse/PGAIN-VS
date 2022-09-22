import json
import pandas as pd

def read_pickle_data():
    df = pd.read_pickle("pickle/extended_dataframe.pickle")

    node1 = df.loc[df['node'] == 'raspihat01']
    node2 = df.loc[df['node'] == 'raspihat02']
    node3 = df.loc[df['node'] == 'raspihat03']
    node4 = df.loc[df['node'] == 'raspihat04']
    node5 = df.loc[df['node'] == 'raspihat05']
    node6 = df.loc[df['node'] == 'raspihat06']
    node7 = df.loc[df['node'] == 'raspihat07']
    node8 = df.loc[df['node'] == 'raspihat08']
    node9 = df.loc[df['node'] == 'raspihat09']
    node10 = df.loc[df['node'] == 'raspihat10']
    node11 = df.loc[df['node'] == 'raspihat11']
    node12 = df.loc[df['node'] == 'raspihat12']

    del df

    _node1 = node1[['datetime', 'temperature']]
    _node1.rename(columns={'temperature': 'raspihat01'}, inplace=True)
    del node1

    _node2 = node2[['datetime', 'temperature']]
    _node2.rename(columns={'temperature': 'raspihat02'}, inplace=True)
    del node2

    _node3 = node3[['datetime', 'temperature']]
    _node3.rename(columns={'temperature': 'raspihat03'}, inplace=True)
    del node3

    _node4 = node4[['datetime', 'temperature']]
    _node4.rename(columns={'temperature': 'raspihat04'}, inplace=True)
    del node4

    _node5 = node5[['datetime', 'temperature']]
    _node5.rename(columns={'temperature': 'raspihat05'}, inplace=True)
    del node5

    _node6 = node6[['datetime', 'temperature']]
    _node6.rename(columns={'temperature': 'raspihat06'}, inplace=True)
    del node6

    _node7 = node7[['datetime', 'temperature']]
    _node7.rename(columns={'temperature': 'raspihat07'}, inplace=True)
    del node7

    _node8 = node8[['datetime', 'temperature']]
    _node8.rename(columns={'temperature': 'raspihat08'}, inplace=True)
    del node8

    _node9 = node9[['datetime', 'temperature']]
    _node9.rename(columns={'temperature': 'raspihat09'}, inplace=True)
    del node9

    _node10 = node10[['datetime', 'temperature']]
    _node10.rename(columns={'temperature': 'raspihat10'}, inplace=True)
    del node10

    _node11 = node11[['datetime', 'temperature']]
    _node11.rename(columns={'temperature': 'raspihat11'}, inplace=True)
    del node11

    _node12 = node12[['datetime', 'temperature']]
    _node12.rename(columns={'temperature': 'raspihat12'}, inplace=True)
    del node12

    dataframe = _node1.merge(_node2, on='datetime')\
        .merge(_node3, on='datetime')\
        .merge(_node4, on='datetime')\
        .merge(_node5, on='datetime')\
        .merge(_node6, on='datetime')\
        .merge(_node7, on='datetime')\
        .merge(_node8, on='datetime')\
        .merge(_node9, on='datetime')\
        .merge(_node10, on='datetime')\
        .merge(_node11, on='datetime')\
        .merge(_node12, on='datetime')
    
    return dataframe

if __name__ == '__main__':
    dataframe = read_pickle_data()
    columns = list(dataframe.columns.values)
    sensor_list = []

    for i in range(len(columns)):
        if i == 0: continue
        col = columns[i]
        df = dataframe[['datetime', col]]
        df.rename(columns={'datetime' : 'timestamp'}, inplace=True)

        outFilename = "historic_data/temperature_silicon_austria/urn_ngsi-ld_Sensor_Raspihat_{}.csv".format(col)
        sensor_name = "urn_ngsi-ld_Sensor_Raspihat_{}".format(col)

        sensor = {
            'id': sensor_name
        }
        sensor_list.append(sensor)

        df.to_csv(outFilename, index=False)

    # print(traffic_sensor_list)
    with open("resources/raspihat_temperature.json", "w+", encoding='utf-8') as file:
        json.dump(sensor_list, file, ensure_ascii=False, indent=4)