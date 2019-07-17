import  pandas as pd
import os.path
from sklearn.externals import joblib
from analyzer_result import get_df_prediction
from dataset_preprocessing import replace_na
import xgboost as xgb
from xgboost import Booster


class DatasetTester:

    def test_dataset(self, test_path, model_path, output_path, algorithm_name, analyzer_path, type):
        test = pd.read_csv(test_path, dtype=str)
        if algorithm_name.lower() in  ['adaboost', 'gausnb']:
            test = self.scikit_test_dataset(test, model_path, output_path)
        elif algorithm_name.lower() == 'xgboost':
            test = self.test_xgboost_dataset(test, model_path, output_path)

        if type == 'Tester':
            description = model_path.split('/')[-1]
            df_statistic = get_df_prediction(test, description=description)
            df_statistic.to_csv(analyzer_path, index=False)

            compare_path = '/'.join(output_path.split('/')[:-1]) + "/compare-reslults.csv"
            if os.path.exists( compare_path):
                df_comapare = pd.read_csv(compare_path)
                df_comapare= pd.concat( [df_comapare, df_statistic], ignore_index=False)
            else:
                df_comapare = df_statistic
            df_comapare.to_csv(compare_path, index=False)
        else:
            self.show_one_order(test)

    def scikit_test_dataset(self, test, model_path, output_path):
        test = test.apply(pd.to_numeric, errors="coerce")
        drop_columns = ['status']
        test1 = test.drop(drop_columns, axis=1, errors="ignore")
        test1 = replace_na(test1)
        test1 = test1.values
        loaded_model = joblib.load(model_path)
        test_pred = loaded_model.predict_proba(test1)
        test["probability"] = test_pred[:, 1]
        test.to_csv(output_path, index=False, quoting=1)
        return test

    def test_xgboost_dataset(self, test, model_path, output_path):
        test = test.apply(pd.to_numeric, errors="coerce")
        drop_columns = ['status']
        test1 = test.drop(drop_columns, axis=1, errors="ignore")
        test_matrix = test1.as_matrix()
        model = xgb.XGBClassifier()
        booster = Booster()
        booster.load_model(model_path)
        model._Booster = booster
        test_pred = model.predict_proba(test_matrix)
        test["probability"] = test_pred[:, 1]
        test.to_csv(output_path, index=False, quoting=1)
        return test

    def show_one_order(self, test):
        print("Order probability = ", test.probability.values[0])
        if self.threshold:
            prediction_status = 1 if test.probability.values[0] >= float(self.threshold) else 0
            print("Prediction status = ", prediction_status)
