import json
import pandas as pd

if __name__ == '__main__':
    dataframe = pd.read_csv('historic_data/traffic_speed/METR-LA.csv')
    columns = list(dataframe.columns.values)
    traffic_sensor_list = []

    for i in range(len(columns)):
        if i == 0: continue
        col = columns[i]
        df = dataframe[['Unnamed: 0', col]]
        df.rename(columns={'Unnamed: 0' : 'timestamp'}, inplace=True)

        outFilename = "historic_data/traffic_speed/urn_ngsi-ld_Sensor_TrafficLA_peed_{}.csv".format(col)
        sensor_name = "urn_ngsi-ld_Sensor_TrafficLA_peed_{}".format(col)

        sensor = {
            'id': sensor_name
        }
        traffic_sensor_list.append(sensor)

        # df.to_csv(outFilename, index=False)

    # print(traffic_sensor_list)
    with open("resources/traffic_speed.json", "w+", encoding='utf-8') as file:
        json.dump(traffic_sensor_list, file, ensure_ascii=False, indent=4)