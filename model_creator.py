import numpy as np
import  pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib



class ModelCreator:

    def create_model(self, algorithm_name, teach_path, params, model_name):
        teach = pd.read_csv(teach_path, dtype=str)
        if algorithm_name.lower() == 'adaboost':
            self.create_adaboost_model(teach, params, model_name)

    def create_adaboost_model(self, teach, config, model_name):

        teach = teach.apply(pd.to_numeric, errors="coerce")

        label = teach.status

        # TODO Исправить подсчет sample weight
        # n_sample = teach.shape[0]
        # n_bad = teach[teach.status == 1].shape[0]
        # weight_bad = min(int(n_sample / n_bad), 99)
        # n = n_bad * weight_bad + n_sample - n_bad
        # weight = np.where(teach.status == 1, weight_bad/n, 1/n)

        drop_columns = 'status'
        train = teach.drop(drop_columns, axis=1, errors="ignore")
        # замена na на -9999
        train = train.fillna(-9999)
        train = train.values

        if not config:
            config = {}
        model = AdaBoostClassifier(**config)

        # model = AdaBoostClassifier(n_estimators=40, learning_rate=0.6)
        model.fit(train, label)#, sample_weight=weight
        joblib.dump(model, model_name)


        importances = pd.DataFrame({"feature_name": teach.drop(drop_columns, axis=1, errors="ignore").columns,
                                    "importances": model.feature_importances_})
        importances.sort_values(by=["importances"], ascending=False, inplace=True)
        feature_path = model_name.split('.')[0] + '-feature.csv'
        importances.to_csv(feature_path, index=False, quoting=1)