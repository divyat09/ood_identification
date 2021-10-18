import os
import argparse

# data_dim_list= [2, 8, 32, 64, 128]
# num_tasks_list= [1, 8, 32, 64]
# num_layers_list= [2, 3, 4]

data_dim_list= [16]
# num_tasks_list= [1, 4, 8, 12, 16]
num_tasks_list= [13, 14, 15]
num_layers_list= [2]

# data_dim_list= [50]
# num_tasks_list= [25, 37, 50]
# num_layers_list= [2]

# data_dim_list= [4]
# num_tasks_list= [1, 2, 4]
# num_layers_list= [2]

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--latent_pred_task', type=int, default=0,
                    help='')
parser.add_argument('--invertible_model', type=int, default=0,
                   help='')
parser.add_argument('--final_task', type=int, default=0,
                   help='')
parser.add_argument('--train_model', type=int, default=0,
                   help='')
parser.add_argument('--data_dir', type=str, default='clf_non_linear_dgp_uniform',
                   help='')

args = parser.parse_args()
latent_pred_task= args.latent_pred_task
invertible_model= args.invertible_model
final_task= args.final_task
train_model= args.train_model

base_script= 'python3 train.py --num_seeds 3  --lr 0.05 --batch_size 512 --num_epochs 1000'

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
                script= script + ' > ' + res_dir + 'num_tasks_' + str(num_tasks) + '_data_dim_' + str(data_dim) + '_latent_prediction_task.txt'
            else:
                script= script + ' > ' + res_dir + 'num_tasks_' + str(num_tasks) + '_data_dim_' + str(data_dim) + '.txt'

            os.system(script)