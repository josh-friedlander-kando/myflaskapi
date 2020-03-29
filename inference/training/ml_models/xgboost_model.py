import os
import json
import numpy as np
from abc import ABC
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error

from model_template import ModelTemplate


def fetch_and_process_data(client, **kwargs):
    client_context = {k: v for k, v in kwargs.items() if k in ['point_id', 'start', 'end']}
    data = client.get_all(**client_context)
    if len(data['samplings']) == 0:
        print(f'No data found at point {kwargs["point_id"]}')
        return None
    # TODO if model not fit bc of missing data, pass this on to predict method
    df = pd.DataFrame(data['samplings']).T[['EC', 'PH', 'COD', 'TSS', 'FLOW', 'TEMPERATURE']].fillna(method='ffill')
    x, y = df[['EC', 'PH', 'TSS', 'FLOW', 'TEMPERATURE']].copy(), df.COD.copy()
    print('finished processing data successfully')
    return x, y


class XgboostTemplate(ModelTemplate, ABC):
    def __init__(self):
        super().__init__()
        self.xgbr = xgb.XGBRegressor()
        self.metadata = {}
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_pred = None, None, None, None, None

    def do_train(self, client, **kwargs):
        x, y = fetch_and_process_data(client, **kwargs)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.15)
        _ = self.xgbr.fit(self.X_train, self.y_train)
        print('finished fitting model')

    def do_save_metadata(self):
        kfold = KFold(n_splits=10, shuffle=True)
        self.metadata['scores'] = cross_val_score(self.xgbr, self.X_train, self.y_train, cv=5).tolist()
        self.metadata['kf_cv_scores'] = cross_val_score(self.xgbr, self.X_train, self.y_train, cv=kfold).tolist()
        self.y_pred = self.xgbr.predict(self.X_test)
        self.metadata['mse'] = mean_squared_error(self.y_test, self.y_pred)
        self.metadata['rmse'] = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        export_dir = os.path.abspath(os.environ.get('PS_MODEL_PATH', os.getcwd() + '../../../models'))
        with open(export_dir + '/gradient-model-metadata.json', 'w') as f:
            json.dump(self.metadata, f)

    def do_predict(self, context):
        return self.xgbr.predict(self.X_test).tolist()  # list to convert to json
