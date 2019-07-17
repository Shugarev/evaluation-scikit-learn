import  pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib

from dataset_preprocessing import replace_na, get_sample_weight, save_feature_importances

class ModelCreator:

    def create_model(self, algorithm_name, teach_path, params, model_name):
        teach = pd.read_csv(teach_path, dtype=str)
        if algorithm_name.lower() == 'adaboost':
            self.create_adaboost_model(teach, params, model_name)

    def create_adaboost_model(self, teach, config, model_name):
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
        joblib.dump(model, model_name)
        save_feature_importances(teach, drop_columns, model.feature_importances_, model_name.split('.')[0] + '-feature.csv')