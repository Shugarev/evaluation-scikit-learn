import  pandas as pd
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB

from dataset_preprocessing import replace_na, get_sample_weight, save_feature_importances,get_xgb_weight

class ModelCreator:

    def create_model(self, algorithm_name, teach_path, params, model_path):
        teach = pd.read_csv(teach_path, dtype=str)
        if algorithm_name.lower() == 'adaboost':
            self.create_adaboost_model(teach, params, model_path)
        elif algorithm_name.lower() == 'xgboost':
            self.create_xgboost_model(teach, params, model_path)
        elif algorithm_name.lower() == 'gausnb':
            self.create_gaussian_naive_bayes_model(teach, params, model_path)

    def create_adaboost_model(self, teach, config, model_path):
        teach = teach.apply(pd.to_numeric, errors="coerce")
        label = teach.status
        drop_columns = ['status']
        train = teach.drop(drop_columns, axis=1, errors="ignore")
        train = replace_na(train)
        train = train.values  # Some algorithms may need to pass matrices. Example train = train.as_matrix()
        if not config:
            config = {}
        model = AdaBoostClassifier(**config)
        sample_weight = get_sample_weight(teach)
        model.fit(train, label)#, sample_weight=sample_weight
        joblib.dump(model, model_path)
        save_feature_importances(teach, drop_columns, model.feature_importances_, model_path.split('.')[0] + '-feature.csv')

    def create_xgboost_model(self, teach, config, model_path):
        teach = teach.apply(pd.to_numeric, errors="coerce")
        label = teach.status
        drop_columns = ['status']
        weight = get_xgb_weight(teach)
        train = teach.drop(drop_columns, axis=1, errors="ignore")
        train = train.as_matrix()
        model = xgb.XGBClassifier(**config)
        model.fit(train, label, sample_weight=weight, eval_metric="auc")  #
        # booster = model.get_booster()
        booster = model.booster()
        booster.save_model(model_path)
        save_feature_importances(teach, drop_columns, model.feature_importances_,
                                 model_path.split('.')[0] + '-feature.csv')

    def create_gaussian_naive_bayes_model(self, teach, config, model_path):
        teach = teach.apply(pd.to_numeric, errors="coerce")
        label = teach.status
        drop_columns = ['status']
        train = teach.drop(drop_columns, axis=1, errors="ignore")
        train = replace_na(train)
        train = train.as_matrix()
        model = GaussianNB()
        model.fit(train, label)
        joblib.dump(model, model_path)
        #save_feature_importances(teach, drop_columns, model.feature_importances_, model_path.split('.')[0] + '-feature.csv')