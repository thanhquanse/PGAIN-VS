**PGAIN VIRTUAL SENSOR and SENSOR ROTATIONAL MEASUREMENT**
</br>
**Author: Nguyen Thanh Quan**
</br>
1/ Impute a single sensor with missing rate
</br>
    - Configure high/medium/low correlation to select best sensors in config.yaml as suggestion inline
</br>
    - Run the following command to impute
</br>
    `python impute.py --data_type <datatype> --miss_rate <missing_rate> --pearson_hint yes`

</br>
2/ Rank sensors
</br>
    `python find_subset.py --data_type <datatype> --miss_rate 0.2`
</br>
3/ Perform rotation after having sensor ranking
</br>
    `python find_subset.py --data_type <datatype> --miss_rate <miss_rate found> --miss_num <number of reduced sensors> --start_from <index to start apply rotation>`