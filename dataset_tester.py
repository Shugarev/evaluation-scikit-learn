import os.path

import pandas as pd
import xgboost as xgb
from sklearn.externals import joblib
from xgboost import Booster

from analyzer_result import get_df_prediction
from dataset_preprocessing import replace_na


class datasetTester:

    @classmethod
    def run(cls, task: str, test_path: str, algorithm_name: str, model_path: str, output_path: str, analyzer_path: str,
            description: str=None):
        test = pd.read_csv(test_path, dtype=str)
        algorithm_name = algorithm_name.lower()
        if algorithm_name in ['adaboost', 'gausnb', 'decisiontree', 'gradientboost', 'logregression', 'linear_sgd',
                              'xgboost']:
            test = cls.test_dataset(test, algorithm_name, model_path, output_path)
        else:
            raise BaseException("Model {}  does not support .\n".format(algorithm_name))
        if task == 'Tester':
            if not description:
                description = model_path.split('/')[-1]
            df_statistic = get_df_prediction(test, description=description)
            df_statistic.to_csv(analyzer_path, index=False)
            compare_path = output_path.rsplit('/', maxsplit=1)[0] + "/compare-reslults.csv"
            if os.path.exists(compare_path):
                df_comapare = pd.read_csv(compare_path)
                df_comapare = pd.concat([df_comapare, df_statistic], ignore_index=False)
            else:
                df_comapare = df_statistic
            df_comapare.to_csv(compare_path, index=False)
        else:
            cls.show_single_order_info(test)

    @classmethod
    def get_model(cls, algorithm_name: str, model_path: str):
        if algorithm_name == 'xgboost':
            model = xgb.XGBClassifier()
            booster = Booster()
            booster.load_model(model_path)
            model._Booster = booster
        else:
            model = joblib.load(model_path)
        return model

    @classmethod
    def test_dataset(cls, test: pd.DataFrame, algorithm_name: str, model_path: str, output_path: str) -> pd.DataFrame:
        test = test.copy()
        test_modified = test.apply(pd.to_numeric, errors="coerce")
        drop_columns = ['status']
        test_modified = test_modified.drop(drop_columns, axis=1, errors="ignore")
        if algorithm_name != 'xgboost':
            test_modified = replace_na(test_modified)
        test_matrix = test_modified.as_matrix()
        loaded_model = cls.get_model(algorithm_name, model_path)
        test_pred = loaded_model.predict_proba(test_matrix)
        test["probability"] = test_pred[:, 1]
        test.to_csv(output_path, index=False, quoting=1)
        return test

    @classmethod
    def set_threshold(cls, threshold: float):
        cls.threshold = threshold

    @classmethod
    def show_single_order_info(cls, test: pd.DataFrame):
        print("Order probability = ", test.probability.values[0])
        if cls.threshold:
            prediction_status = 1 if test.probability.values[0] >= float(cls.threshold) else 0
            print("Prediction status = ", prediction_status, '\n')
