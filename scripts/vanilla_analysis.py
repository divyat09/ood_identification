import os
import argparse

data_dim_list= [2, 8, 32, 64]
num_tasks_list= [1, 2, 4, 8]
num_layers_list= [2, 3, 4, 5]


# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--latent_pred_task', type=int, default=0,
                    help='')
parser.add_argument('--invertible_model', type=int, default=0,
                   help='')

args = parser.parse_args()
latent_pred_task= args.latent_pred_task
invertible_model= args.invertible_model

base_script= 'python3 train.py --num_seeds 3  --lr 0.001 --batch_size 32 --num_epochs 100'

if latent_pred_task:
    base_script = base_script + ' --latent_pred_task 1 '

if invertible_model:
    base_script= base_script + ' --invertible_model 1 '
    
for num_layers in num_layers_list:
    
    if invertible_model:
        res_dir= 'results/invertible/fc_' + str(num_layers) + '/'
    else:
        res_dir= 'results/non_invertible/fc_' + str(num_layers) + '/'
    
    if not os.path.exists(res_dir):
        os.makedirs(res_dir) 
    
    for data_dim in data_dim_list:
        for num_tasks in num_tasks_list:
            script= base_script + ' --num_tasks ' + str(num_tasks) + ' --data_dim ' + str(data_dim) + ' --num_layers ' + str(num_layers)

            if latent_pred_task:
                script= script + ' > ' + res_dir + 'num_tasks_' + str(num_tasks) + '_data_dim_' + str(data_dim) + '_latent_prediction_task.txt'
            else:
                script= script + ' > ' + res_dir + 'num_tasks_' + str(num_tasks) + '_data_dim_' + str(data_dim) + '.txt'

            os.system(script)