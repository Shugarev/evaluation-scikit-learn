#!/bin/bash

python executor.py  --task Teacher --input db_teach.csv --algorithm_name adaboost --model_name ada_model --algorithm_config config-adaboost.ini
printf "ada_model done \n"
python executor.py  --task Tester  --input db_test.csv  --algorithm_name adaboost --model_name ada_model --output Test-result-adaboost.csv
python executor.py  --task TesterOrder  --input db_test1c.csv  --algorithm_name adaboost --model_name ada_model --threshold 0.505484 --output Test-result-adaboost-1.csv

python executor.py --task Teacher --input db_teach.csv --algorithm_name xgboost --model_name xgb_model --algorithm_config config-xgboost.ini
printf "xgb_model done \n"
python executor.py --task Tester --input db_test.csv --algorithm_name xgboost --model_name xgb_model --output Test-result-xgboost.csv
python executor.py --task TesterOrder  --input db_test1c.csv  --algorithm_name xgboost --model_name xgb_model --threshold 0.979798  --output Test-result-xgboost-1.csv


python executor.py --task Teacher --input db_teach.csv --algorithm_name gradientboost --model_name gradientboost_model --algorithm_config config-gradientboost.ini
printf "gradientboost_model done \n"
python executor.py --task Tester --input db_test.csv --algorithm_name gradientboost --model_name gradientboost_model --output Test-result-gradientboost.csv
python executor.py --task TesterOrder  --input db_test1c.csv  --algorithm_name gradientboost --model_name gradientboost_model --threshold 0.666958 --output Test-result-gradientboost-1.csv


python executor.py --task Teacher --input db_teach.csv --algorithm_name decisiontree --model_name decisiontree_model
printf "decisiontree_model done \n \n"
python executor.py --task Tester --input db_test.csv --algorithm_name decisiontree --model_name decisiontree_model --output Test-result-decisiontree.csv
#python executor.py --task TesterOrder  --input db_test1b.csv  --algorithm_name decisiontree --model_name decisiontree_model --threshold 0.77991

python executor.py --task Teacher --input db_teach.csv --algorithm_name logregression --model_name logregression_model
printf "logregression_model done \n"
python executor.py --task Tester --input db_test.csv --algorithm_name logregression --model_name logregression_model --output Test-result-logregression.csv
python executor.py --task TesterOrder  --input db_test1c.csv  --algorithm_name logregression --model_name logregression_model --threshold 0.166869 --output Test-result-logregression.csv


python executor.py --task Encode_teach --input db_teach-bayes.csv --encode_config config-bayes-encode.ini  --encoded_path encoded_bayes
python executor.py --task Encode_test  --input db_test-bayes.csv  --encode_config config-bayes-encode.ini  --encoded_path encoded_bayes

python executor.py --task Teacher --input db_teach-bayes-encoded.csv --algorithm_name gausnb --model_name gausnb_model
printf "gausnb_model done \n"
python executor.py --task Tester --input db_test-bayes-encoded.csv --algorithm_name gausnb --model_name gausnb_model --output Test-result-gausnb.csv
#python executor.py --task TesterOrder  --input db_test1c.csv  --algorithm_name gausnb --model_name gausnb_model --threshold 0.28838  --output Test-result-gausnb-1.csv



#python executor.py  --task Selection_parameters --input db_teach.csv --algorithm_name adaboost --model_name ada_model --algorithm_config config-adaboost.ini --output Test-result-adabust.csv --input2 db_test.csv
#python executor.py  --task Selection_parameters --input db_teach.csv --algorithm_name gradientboost --model_name gradientboost_model --algorithm_config config-gradientboost.ini --output Test-result-gradientboost.csv --input2 db_test.csv
#python executor.py  --task Selection_parameters --input db_teach.csv --algorithm_name xgboost --model_name xgboost_model --algorithm_config config-xgboost.ini --output Test-result-xgboost.csv --input2 db_test.csv
#
#python executor.py  --task Teacher --input db_teach.csv --algorithm_name linear_sgd --model_name linear_sgd_model --algorithm_config config-linear_sgd.ini
#printf "linear_sgd_model done \n"
#python executor.py  --task Tester  --input db_test.csv  --algorithm_name linear_sgd --model_name linear_sgd_model --output Test-result-linear_sgd.csv
#python executor.py  --task TesterOrder  --input db_test1b.csv  --algorithm_name linear_sgd --model_name linear_sgd_model --threshold 0.50813 --output Test-result-linear_sgd-1.csv
