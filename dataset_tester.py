import numpy as np
import  pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib

class DatasetTester:
    def test_dataset(self, test_path, model_path, output_path, algorithm_name):
        test =  pd.read_csv(test_path, dtype=str)

        if algorithm_name.lower() == 'adaboost':
            self.test_adaboost_dataset(test, model_path, output_path)

    def test_adaboost_dataset(self, test, model_path, output_path):
        test = test.apply(pd.to_numeric, errors="coerce")

        label = test.status
        drop_columns = ['status']
        test1 = test.drop(drop_columns, axis=1, errors="ignore")
        # замена na на -9999
        test1 = test1.fillna(-9999)
        test1 = test1.as_matrix()
        # test1 = test1.values

        loaded_model = joblib.load(model_path)
        test_pred = loaded_model.predict_proba(test1)
        test["probability"] = test_pred[:, 1]
        test.to_csv(output_path, index=False, quoting=1)
