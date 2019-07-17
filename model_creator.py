import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.externals import joblib
import xgboost as xgb
from dataset_preprocessing import replace_na, get_sample_weight, save_feature_importances,get_xgb_weight


class ModelCreator:

    def create_model(self, algorithm_name, teach_path, params, model_path):
        teach = pd.read_csv(teach_path, dtype=str)
        algorithm_name = algorithm_name.lower()
        if algorithm_name == 'xgboost':
            self.create_xgboost_model(teach, params, model_path, algorithm_name)
        else:
            teach = teach.apply(pd.to_numeric, errors="coerce")
            label = teach.status
            drop_columns = ['status']
            train = teach.drop(drop_columns, axis=1, errors="ignore")
            train = replace_na(train)
            train = train.as_matrix()
            model = self.get_model(params, algorithm_name)
            if algorithm_name == 'adaboost  1':
                weight = get_sample_weight(teach)
                model.fit(train, label, sample_weight=weight)
            else:
                model.fit(train, label)
            joblib.dump(model, model_path)
            if algorithm_name in ['adaboost', 'decisiontree', 'gradientboost']:
                save_feature_importances(teach, drop_columns, model.feature_importances_,
                                         model_path.split('.')[0] + '-feature.csv')

    def create_xgboost_model(self, teach, params, model_path, algorithm_name):
        teach = teach.apply(pd.to_numeric, errors="coerce")
        label = teach.status
        drop_columns = ['status']
        weight = get_xgb_weight(teach)
        train = teach.drop(drop_columns, axis=1, errors="ignore")
        train = train.as_matrix()
        model = self.get_model(params, algorithm_name)
        model.fit(train, label, sample_weight=weight, eval_metric="auc")  #
        # booster = model.get_booster()
        booster = model.booster()
        booster.save_model(model_path)
        save_feature_importances(teach, drop_columns, model.feature_importances_, model_path.split('.')[0] + '-feature.csv')

    def get_model(self, config, algorithm_name):
        if not config:
            config = {}
        if algorithm_name == 'adaboost':
            return AdaBoostClassifier(**config)
        elif algorithm_name == 'xgboost':
            return xgb.XGBClassifier(**config)
        elif algorithm_name == 'gausnb':
            return GaussianNB()
        elif algorithm_name == 'decisiontree':                      # do not calculate probabilities
            return tree.DecisionTreeClassifier()
        elif algorithm_name == 'gradientboost':
            return GradientBoostingClassifier(**config)