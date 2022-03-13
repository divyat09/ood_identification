# Towards efficient representation identification in supervised learning
Code accompanying the paper [Towards efficient representation identification in supervised learning](https://openreview.net/forum?id=7UwoSnMDXWE) published in [CleaR 2022](https://www.cclear.cc/2022)

# Instructions

Follow the notebook 'reproducing_results.ipynb'.


python3 data/regression_data.py --data_dim 16 --num_tasks_list 8 12 16
python3 data/regression_data.py --data_dim 24 --num_tasks_list 12 18 24
python3 data/regression_data.py --data_dim 50 --num_tasks_list 25 37 50

python3 data/multi_binary_classification_data.py --data_dim 16 --num_tasks_list 8 12 16
python3 data/multi_binary_classification_data.py --data_dim 24 --num_tasks_list 12 18 24
python3 data/multi_binary_classification_data.py --data_dim 50 --num_tasks_list 25 37 50


python3 scripts/vanilla_analysis.py --data_dir regression_num_layer_2_latent_uniform_discrete --data_dim 16 --num_tasks_list 8 12 16 --train_model 1
python3 scripts/vanilla_analysis.py --data_dir regression_num_layer_2_latent_uniform_discrete --data_dim 24 --num_tasks_list 12 18 24 --train_model 1
python3 scripts/vanilla_analysis.py --data_dir regression_num_layer_2_latent_uniform_discrete --data_dim 50 --num_tasks_list 25 37 50 --train_model 1

python3 scripts/vanilla_analysis.py --data_dir multi_bin_classification_num_layer_2_latent_uniform_discrete --final_task 1 --data_dim 16 --num_tasks_list 8 12 16 --train_model 1
python3 scripts/vanilla_analysis.py --data_dir multi_bin_classification_num_layer_2_latent_uniform_discrete --final_task 1 --data_dim 24 --num_tasks_list 12 18 24 --train_model 1
python3 scripts/vanilla_analysis.py --data_dir multi_bin_classification_num_layer_2_latent_uniform_discrete --final_task 1 --data_dim 50 --num_tasks_list 25 37 50 --train_model 1
