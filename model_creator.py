from typing import Dict

import pandas as pd
import xgboost as xgb
from sklearn import linear_model
from sklearn import tree
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

from dataset_preprocessing import replace_na, get_sample_weight, save_feature_importances, get_xgb_weight


class ModelCreator:
    def run(self, teach_path: str, algorithm_name: str, params: Dict, model_path: str):
        model, teach, drop_columns = self.create_model(teach_path, algorithm_name, params, model_path)
        if algorithm_name in ['adaboost', 'decisiontree', 'gradientboost', 'xgboost']:
            save_feature_importances(teach, drop_columns, model.feature_importances_,
                                     model_path.split('.')[0] + '-feature.csv')

    def create_model(self, teach_path: str, algorithm_name: str, params: Dict, model_path: str):
        teach = pd.read_csv(teach_path, dtype=str)
        teach = teach.apply(pd.to_numeric, errors="coerce")
        label = teach.status
        drop_columns = ['status']
        train = teach.drop(drop_columns, axis=1, errors="ignore")
        if algorithm_name != 'xgboost':
            train = replace_na(train)
        train = train.as_matrix()
        model = self.get_model(params, algorithm_name)
        if algorithm_name == 'xgboost':
            weight = get_xgb_weight(teach)
            model.fit(train, label, sample_weight=weight, eval_metric="auc")  #
            booster = model.get_booster()
            # booster = model.booster() # для более ранних версий xgboost-a ( 0.6.a2)
            booster.save_model(model_path)
        else:
            if algorithm_name == 'adaboost':
                weight = get_sample_weight(teach)
                model.fit(train, label, sample_weight=weight)
            else:
                model.fit(train, label)
            if 0:
                calibrator = CalibratedClassifierCV(model, cv='prefit')
                calibrator.fit(train, label)
                joblib.dump(calibrator, model_path)
            else:
                joblib.dump(model, model_path)
        return model, teach, drop_columns

    def get_model(self, config: Dict, algorithm_name: str):
        if not config:
            config = {}
        if algorithm_name == 'adaboost':
            return AdaBoostClassifier(**config)
        elif algorithm_name == 'xgboost':
            return xgb.XGBClassifier(**config)
        elif algorithm_name == 'gausnb':
            return GaussianNB()
        elif algorithm_name == 'decisiontree':  # do not calculate probabilities
            return tree.DecisionTreeClassifier()
        elif algorithm_name == 'gradientboost':
            return GradientBoostingClassifier(**config)
        elif algorithm_name == 'logregression':
            return LogisticRegression(**config)
        elif algorithm_name == 'linear_sgd':
            return linear_model.SGDClassifier(**config)

    # TODO подбор параметро для алгоритма
    def find_best_params(self, teach_path: str, params: Dict, algorithm_name: Dict, parameters_range: Dict):
        teach = pd.read_csv(teach_path, dtype=str)
        if algorithm_name in ['adaboost', 'gradientboost', 'xgboost', 'linear_sgd']:
            teach = teach.apply(pd.to_numeric, errors="coerce")
            label = teach.status
            drop_columns = ['status']
            train = teach.drop(drop_columns, axis=1, errors="ignore")
            train = replace_na(train)
            train = train.as_matrix()
            model = self.get_model(params, algorithm_name)
            model = GridSearchCV(model, parameters_range)
            model = model.fit(train, label)
            print("Grid search best params for {}:".format(algorithm_name))
            print(model.best_params_)
            print()
