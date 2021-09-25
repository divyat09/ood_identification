import os

num_task_list= [1, 2, 4, 8]
data_dim_list= [2, 8, 32, 64]

base_script= 'python3 data/regression_data.py '
for data_dim in data_dim_list:
    for num_task in num_task_list:
        script = base_script + ' --num_tasks ' + str(num_task) + ' --data_dim ' + str(data_dim)
        os.system(script)
