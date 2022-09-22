import csv
import sys
import datetime

if __name__ == '__main__':

    inFilename = "romania_weather.csv"
    f = open(inFilename, "r")
    i = csv.reader(f)
    cities = {}
    for entry in i:
        cities[entry[2].upper()] = 1

    for city in cities:
        if city.startswith('/'):
            city = city[1:]
        outFilename = "urn_ngsi-ld_Sensor_RomanianWeather_{}_temperature.csv".format(city)
        f = open(inFilename, "r")
        f2 = open(outFilename, "w")
        i = csv.reader(f)
        o = csv.writer(f2)
        o.writerow(["timestamp", "temperature"])
        for entry in i:
            if entry[2].upper() == city:
                try:
                    dt = datetime.datetime.fromtimestamp(int(entry[5])/1000)
                    print(dt.isoformat(), entry[4])
                    o.writerow([dt.isoformat(), entry[4]])
                except ValueError:
                    pass
        f.close()
        f2.close()
