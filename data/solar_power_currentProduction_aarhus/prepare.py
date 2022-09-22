import csv
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("please provide sensor number")
        sys.exit(1)
    number = sys.argv[1]
    inFilename = "sensor_{}.csv".format(number)
    outFilename = "urn_ngsi-ld_Sensor_SolarPowerAarhus_CurrentProduction_{}.csv".format(number)
    f = open(inFilename, "r")
    f2 = open(outFilename, "w")
    i = csv.reader(f)
    o = csv.writer(f2)
    for entry in i:
        print(entry[7], entry[3])
        o.writerow([entry[7], entry[3]])
    f.close()
    f2.close()
