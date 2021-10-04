import os

data_dim_list= [2, 8, 32, 64, 128]

base_script= 'python3 data/regression_data.py '
for data_dim in data_dim_list:
    script = base_script + ' --data_dim ' + str(data_dim)
    os.system(script)
