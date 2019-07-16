evaluation-scikit-learn

python executor.py --type Teacher --input datasets/db_teach.csv --algorithm_name adaboost --model_name models/ada_model --algorithm_config configs/config-adaboost.ini
python executor.py --type Tester  --input datasets/db_test.csv  --algorithm_name adaboost --model_name models/ada_model 