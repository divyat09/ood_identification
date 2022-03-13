import os
import argparse


# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--latent_pred_task', type=int, default=0,
                    help='')
parser.add_argument('--invertible_model', type=int, default=0,
                   help='')
parser.add_argument('--final_task', type=int, default=0,
                   help='0: regression; 1: classification')
parser.add_argument('--train_model', type=int, default=0,
                   help='')
parser.add_argument('--data_dir', type=str, default='regression_num_layer_2_latent_uniform_discrete',
                   help='')
parser.add_argument('--data_dim_list', nargs='+', type=int, default=[16], 
                    help='')
parser.add_argument('--num_tasks_list', nargs='+', type=int, default=[8, 12, 16], 
                    help='')

args = parser.parse_args()
latent_pred_task= args.latent_pred_task
invertible_model= args.invertible_model
final_task= args.final_task
train_model= args.train_model

data_dim_list= args.data_dim_list
num_tasks_list= args.num_tasks_list
num_layers_list= [2]

if final_task:
    base_script= 'python3 train.py --num_seeds 10  --lr 0.05 --batch_size 512 --num_epochs 200'   
else:
    base_script= 'python3 train.py --num_seeds 3  --lr 0.01 --batch_size 512 --num_epochs 1000'

base_script= base_script + ' --data_dir ' + str(args.data_dir) + ' --train_model ' + str(train_model) + ' --final_task ' + str(final_task) + ' --invertible_model ' + str(invertible_model) + ' --latent_pred_task ' + str(latent_pred_task)

for num_layers in num_layers_list:
    
    if invertible_model:
        res_dir= 'results/invertible/fc_' + str(num_layers) + '/'
    else:
        res_dir= 'results/non_invertible/fc_' + str(num_layers) + '/'
    
    if not os.path.exists(res_dir):
        os.makedirs(res_dir) 
    
    for data_dim in data_dim_list:
        for num_tasks in num_tasks_list:
            
            print('Data Dim: ', data_dim, ' Num Tasks: ', num_tasks)
            
            script= base_script + ' --num_tasks ' + str(num_tasks) + ' --data_dim ' + str(data_dim) + ' --num_layers ' + str(num_layers)

            if latent_pred_task:
                script= script + ' > ' + res_dir + 'final_task_' + str(final_task) + '_num_tasks_' + str(num_tasks) + '_data_dim_' + str(data_dim) + '_latent_prediction_task.txt'
            else:
                script= script + ' > ' + res_dir + 'final_task_' + str(final_task) + '_num_tasks_' + str(num_tasks) + '_data_dim_' + str(data_dim) + '.txt'

            os.system(script)