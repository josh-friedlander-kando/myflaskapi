import time
import requests
from datetime import datetime, timedelta

all_points = [925, 888, 896, 906, 911, 920, 926, 927, 889, 891]

base_url = 'http://127.0.0.1:5000/'
start = (datetime.now() - timedelta(weeks=16)).replace(hour=0, minute=0, second=0, microsecond=0)
end = int((start + timedelta(weeks=16)).timestamp())
start = int(start.timestamp())

prediction_param = 'EC'

train_params = {'unit_id': '', 'start': start, 'end': end, 'prediction_param': prediction_param}
pred_params = {'pred_hours': 12, 'pred_only': True}

for p in all_points:
    train_params['point_id'] = p
    train_params['model_id'] = '_'.join([str(x) for x in [p, start, end]]) + '_' + prediction_param
    start_time = time.time()
    r = requests.post(base_url + 'create', json=train_params)
    assert r.ok
    pred_params['point_id'] = p
    pred_params['model_id'] = '_'.join([str(x) for x in [p, start, end]]) + '_' + prediction_param
    r = requests.post(base_url + 'predict', json=pred_params)
    print(f'Finished point {p}, took {time.time() - start_time} seconds')


[925, 888, 896, 906, 911, 920, 926, 927, 889, 891, 914, 916, 923, 932, 937, 939, 883, 940, 893, 900, 901, 907, 899, 903, 909, 918, 921, 922, 924, 931, 933, 936, 895, 898, 908, 910, 913, 919, 934, 935, 955, 956, 975, 981, 994, 995, 960, 967, 971, 979, 992, 993, 1000, 958, 959, 961, 970, 973, 980, 982, 983, 988, 996, 999, 1012, 998, 1013, 964, 972, 986, 969, 965, 968, 976, 984, 997, 985, 1023, 1033, 1038, 1043, 1046, 1048, 1049, 1050, 1051, 1056, 1057, 1059, 1060, 1061, 1062, 1069, 1071, 1073, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1086, 1090, 1091, 1092, 1093, 1094, 1095, 1098, 1099, 1103, 1104, 1106, 1107, 1111, 1114, 1115, 1125, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1143, 1144, 1145, 1146, 1151, 1161, 1163, 1166, 1191, 1192, 1208, 1232, 1238, 1263, 1276, 1323, 1332, 1333, 1335, 1336, 1338, 1339, 1340, 1342, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1372, 1373, 1374, 1375, 1378, 1461, 1485, 1490, 1510, 1530, 1531, 1533, 1542, 1545, 1546, 1551, 1560, 1561, 1567, 1568, 1584, 1660, 882, 1908, 1917, 1977, 1978, 1989, 1990, 1991, 1994, 1995, 2045, 2046, 2198, 2248, 2292, 2294, 2295, 2296, 2313, 2385, 2387, 2388, 2389, 2390, 2391, 2392, 3029, 3226, 3256, 3436, 3668, 4067, 4210, 4338, 4386, 5056, 5058]