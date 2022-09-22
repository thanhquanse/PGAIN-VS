**PGAIN VIRTUAL SENSOR and SENSOR ROTATIONAL MEASUREMENT**
**Author: Nguyen Thanh Quan**

1/ Impute a single sensor with missing rate
    - Configure high/medium/low correlation to select best sensors in config.yaml as suggestion inline
    - Run the following command to impute
    `python impute.py --data_type <datatype> --miss_rate <missing_rate> --pearson_hint yes`

2/ Rank sensors
    `python find_subset.py --data_type <datatype> --miss_rate 0.2`

3/ Perform rotation after having sensor ranking
    `python find_subset.py --data_type <datatype> --miss_rate <miss_rate found> --miss_num <number of reduced sensors> --start_from <index to start apply rotation>`