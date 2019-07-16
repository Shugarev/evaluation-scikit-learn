evaluation-scikit-learn
python  ./executor.py/executor.pl Teacher --profile ${shieldd} --input Csv:${path_out}/db_teach.csv --opt model_name=Test1 --opt constants=const.json

#python  /.executor.pl --type Teacher --profile model1 --input db_teach.csv --opt model_name=Test1 --opt config=config.ini
python  /.executor.pl --type Teacher --profile model1 --input db_teach.csv --model_name Test1 --config config.ini