#Generate data for the regression task case
python3 data/regression_data.py --data_dim 16 --num_tasks_list 8 12 16
python3 data/regression_data.py --data_dim 24 --num_tasks_list 12 18 24
python3 data/regression_data.py --data_dim 50 --num_tasks_list 25 37 50

#Generate data for the classification task case
python3 data/multi_binary_classification_data.py --data_dim 16 --num_tasks_list 8 12 16
python3 data/multi_binary_classification_data.py --data_dim 24 --num_tasks_list 12 18 24
python3 data/multi_binary_classification_data.py --data_dim 50 --num_tasks_list 25 37 50

#Regression task experiments
python3 scripts/vanilla_analysis.py --data_dir regression_num_layer_2_latent_uniform_discrete --data_dim 16 --num_tasks_list 8 12 16 --train_model 1
python3 scripts/vanilla_analysis.py --data_dir regression_num_layer_2_latent_uniform_discrete --data_dim 24 --num_tasks_list 12 18 24 --train_model 1
python3 scripts/vanilla_analysis.py --data_dir regression_num_layer_2_latent_uniform_discrete --data_dim 50 --num_tasks_list 25 37 50 --train_model 1

#Classification task experiments
python3 scripts/vanilla_analysis.py --data_dir multi_bin_classification_num_layer_2_latent_uniform_discrete --final_task 1 --data_dim 16 --num_tasks_list 8 12 16 --train_model 1
python3 scripts/vanilla_analysis.py --data_dir multi_bin_classification_num_layer_2_latent_uniform_discrete --final_task 1 --data_dim 24 --num_tasks_list 12 18 24 --train_model 1
python3 scripts/vanilla_analysis.py --data_dir multi_bin_classification_num_layer_2_latent_uniform_discrete --final_task 1 --data_dim 50 --num_tasks_list 25 37 50 --train_model 1
