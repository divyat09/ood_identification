import matplotlib
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dim_list', nargs='+', type=int, default=[16], 
                    help='')
parser.add_argument('--num_tasks_list', nargs='+', type=int, default=[8, 12, 16], 
                    help='')
args = parser.parse_args()

data_dim_list= args.data_dim_list
num_tasks_list= args.num_tasks_list
num_fc= 2

x_ticks=[]
for item in num_tasks_list:
    x_ticks.append(str(item))        

        
res={}
for data_dim in data_dim_list:
    if data_dim not in res.keys():
        res[data_dim]= {}
    for num_tasks in num_tasks_list:
        if num_tasks not in res[data_dim].keys():            
            res[data_dim][num_tasks]= { 'target_pred_rmse': {}, 
                                       'target_pred_r2': {}, 
                                       'ica_target_pred_rmse':{},
                                       'ica_target_pred_r2':{},
                                       'pca_target_pred_rmse':{},
                                       'pca_target_pred_r2':{},
                                       'latent_pred_score':{}, 
                                       'ica_latent_pred_score':{}, 
                                       'pca_latent_pred_score':{}
                                      }
            
            f= open('results/non_invertible/fc_' + str(num_fc) + '/final_task_0_num_tasks_' + str(num_tasks) + '_data_dim_' + str(data_dim) + '.txt', 'r')
            data= f.readlines()
        
            res[data_dim][num_tasks]['target_pred_rmse']['mean'] = float(data[-9].split(' ')[-2].replace('\n',''))
            res[data_dim][num_tasks]['target_pred_rmse']['std'] = float(data[-9].split(' ')[-1].replace('\n',''))
            
            res[data_dim][num_tasks]['target_pred_r2']['mean'] = float(data[-8].split(' ')[-2].replace('\n',''))
            res[data_dim][num_tasks]['target_pred_r2']['std'] = float(data[-8].split(' ')[-1].replace('\n',''))

            res[data_dim][num_tasks]['ica_target_pred_rmse']['mean'] = float(data[-7].split(' ')[-2].replace('\n',''))
            res[data_dim][num_tasks]['ica_target_pred_rmse']['std'] = float(data[-7].split(' ')[-1].replace('\n',''))
            
            res[data_dim][num_tasks]['ica_target_pred_r2']['mean'] = float(data[-6].split(' ')[-2].replace('\n',''))
            res[data_dim][num_tasks]['ica_target_pred_r2']['std'] = float(data[-6].split(' ')[-1].replace('\n',''))

            res[data_dim][num_tasks]['pca_target_pred_rmse']['mean'] = float(data[-5].split(' ')[-2].replace('\n',''))
            res[data_dim][num_tasks]['pca_target_pred_rmse']['std'] = float(data[-5].split(' ')[-1].replace('\n',''))
            
            res[data_dim][num_tasks]['pca_target_pred_r2']['mean'] = float(data[-4].split(' ')[-2].replace('\n',''))
            res[data_dim][num_tasks]['pca_target_pred_r2']['std'] = float(data[-4].split(' ')[-1].replace('\n',''))
            

            res[data_dim][num_tasks]['latent_pred_score']['mean'] = float(data[-3].split(' ')[-2].replace('\n',''))
            res[data_dim][num_tasks]['latent_pred_score']['std'] = float(data[-3].split(' ')[-1].replace('\n',''))

            res[data_dim][num_tasks]['ica_latent_pred_score']['mean'] = float(data[-2].split(' ')[-2].replace('\n',''))
            res[data_dim][num_tasks]['ica_latent_pred_score']['std'] = float(data[-2].split(' ')[-1].replace('\n',''))

            res[data_dim][num_tasks]['pca_latent_pred_score']['mean'] = float(data[-1].split(' ')[-2].replace('\n',''))
            res[data_dim][num_tasks]['pca_latent_pred_score']['std'] = float(data[-1].split(' ')[-1].replace('\n',''))
            

fontsize=80
fontsize_lgd= fontsize/1.2
marker_list = ['o', '^', '*']
            
            
#Label Prediction RMSE
matplotlib.rcParams.update({'errorbar.capsize': 2})
fig, ax = plt.subplots(1, len(data_dim_list), figsize=(20, 15))
# ax.set_ylim(1.0, 5.0)

idx=0
for data_dim in data_dim_list:
    
    ax.tick_params(labelsize=fontsize)
    ax.set_xticklabels(num_tasks_list, rotation=25)
    ax.set_ylabel('RMSE', fontsize=fontsize)
    ax.set_xlabel('Number of Tasks', fontsize=fontsize)
#     ax.set_title('Label Prediction ' + ':' + ' Data Dim: ' + str(data_dim), fontsize=fontsize)    
    ax.set_title('Label Prediction', fontsize=fontsize)    
    
    key='target_pred_rmse'
    out=[]
    out_err=[]
    for num_tasks in num_tasks_list:
        data= res[data_dim][num_tasks][key]
        out.append(data['mean'])
        out_err.append(data['std'])

    ax.errorbar(x_ticks, out, yerr=out_err, marker= marker_list[0], color='C0', ls='-', lw=20, mfc='w', mew=2.5, ms=60, alpha=0.9, fmt='o--', label='ERM')

    key='ica_target_pred_rmse'
    out=[]
    out_err=[]
    for num_tasks in num_tasks_list:
        data= res[data_dim][num_tasks][key]
        out.append(data['mean'])
        out_err.append(data['std'])

    ax.errorbar(x_ticks, out, yerr=out_err, marker= marker_list[1], color='C1', ls='--', lw=20, mfc='w', mew=2.5, ms=60, alpha=0.9, fmt='o--', label='ERM-ICA')

    key='pca_target_pred_rmse'
    out=[]
    out_err=[]
    for num_tasks in num_tasks_list:
        data= res[data_dim][num_tasks][key]
        out.append(data['mean'])
        out_err.append(data['std'])

    ax.errorbar(x_ticks, out, yerr=out_err, marker= marker_list[2], color='C2', ls='-.', lw=20, mfc='w', mew=2.5, ms=60, alpha=0.9, fmt='o--', label='ERM-PCA')
    
    
    idx=idx+1
    
    
lines, labels = fig.axes[-1].get_legend_handles_labels()    
lgd= fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.20), fontsize=fontsize, ncol=3)
    
plt.tight_layout()
plt.savefig('plots_final/rmse_label.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight',  dpi=600)



#Label Prediction R2            
matplotlib.rcParams.update({'errorbar.capsize': 2})
fig, ax = plt.subplots(1, len(data_dim_list), figsize=(20, 15))
ax.set_ylim(0.5, 1.1)

idx=0
for data_dim in data_dim_list:
    
    ax.tick_params(labelsize=fontsize)
    ax.set_xticklabels(num_tasks_list, rotation=25)
    ax.set_ylabel('R2', fontsize=fontsize)
    ax.set_xlabel('Number of Tasks', fontsize=fontsize)
#     ax.set_title('Label Prediction ' + ':' + ' Data Dim: ' + str(data_dim), fontsize=fontsize)    
    ax.set_title('Label Prediction', fontsize=fontsize)    
    
    key='target_pred_r2'
    out=[]
    out_err=[]
    for num_tasks in num_tasks_list:
        data= res[data_dim][num_tasks][key]
        out.append(data['mean'])
        out_err.append(data['std'])

    ax.errorbar(x_ticks, out, yerr=out_err, marker= marker_list[0], color='C0', ls='-', lw=20, mfc='w', mew=2.5, ms=60, alpha=0.9, fmt='o--', label='ERM')

    key='ica_target_pred_r2'
    out=[]
    out_err=[]
    for num_tasks in num_tasks_list:
        data= res[data_dim][num_tasks][key]
        out.append(data['mean'])
        out_err.append(data['std'])

    ax.errorbar(x_ticks, out, yerr=out_err, marker= marker_list[1], color='C1', ls='--', lw=20, mfc='w', mew=2.5, ms=60, alpha=0.9, fmt='o--', label='ERM-ICA')

    key='pca_target_pred_r2'
    out=[]
    out_err=[]
    for num_tasks in num_tasks_list:
        data= res[data_dim][num_tasks][key]
        out.append(data['mean'])
        out_err.append(data['std'])

    ax.errorbar(x_ticks, out, yerr=out_err, marker= marker_list[2], color='C2', ls='-.', lw=20, mfc='w', mew=2.5, ms=60, alpha=0.9, fmt='o--', label='ERM-PCA')
    
    
    idx=idx+1
    
    
lines, labels = fig.axes[-1].get_legend_handles_labels()    
lgd= fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.20), fontsize=fontsize, ncol=3)
    
plt.tight_layout()
plt.savefig('plots_final/r2_label.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight',  dpi=600)


#Latent Prediction MCC Score
matplotlib.rcParams.update({'errorbar.capsize': 2})
fig, ax = plt.subplots(1, len(data_dim_list), figsize=(20, 15))
ax.set_ylim(0, 100)

idx=0
for data_dim in data_dim_list:
    
    ax.tick_params(labelsize=fontsize)
    ax.set_xticklabels(num_tasks_list, rotation=25)
    ax.set_ylabel('MCC', fontsize=fontsize)
    ax.set_xlabel('Number of Tasks', fontsize=fontsize)
#     ax.set_title('Latent Prediction ' + ':' + ' Data Dim: ' + str(data_dim), fontsize=fontsize)    
    ax.set_title('Latent Prediction', fontsize=fontsize)    
    
    key='latent_pred_score'
    out=[]
    out_err=[]
    for num_tasks in num_tasks_list:
        data= res[data_dim][num_tasks][key]
        out.append(data['mean'])
        out_err.append(data['std'])

    ax.errorbar(x_ticks, out, yerr=out_err, marker= marker_list[0], color='C0', ls='-', lw=20, mfc='w', mew=2.5, ms=60, alpha=0.9, fmt='o--', label='ERM')

    key='ica_latent_pred_score'
    out=[]
    out_err=[]
    for num_tasks in num_tasks_list:
        data= res[data_dim][num_tasks][key]
        out.append(data['mean'])
        out_err.append(data['std'])

    ax.errorbar(x_ticks, out, yerr=out_err, marker= marker_list[1], color='C1', ls='--', lw=20, mfc='w', mew=2.5, ms=60, alpha=0.9, fmt='o--', label='ERM-ICA')

    key='pca_latent_pred_score'
    out=[]
    out_err=[]
    for num_tasks in num_tasks_list:
        data= res[data_dim][num_tasks][key]
        out.append(data['mean'])
        out_err.append(data['std'])

    ax.errorbar(x_ticks, out, yerr=out_err, marker= marker_list[2], color='C2', ls='-.', lw=20, mfc='w', mew=2.5, ms=60, alpha=0.9, fmt='o--', label='ERM-PCA')
    
    
    idx=idx+1
    
    
lines, labels = fig.axes[-1].get_legend_handles_labels()    
lgd= fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.20), fontsize=fontsize, ncol=3)
    
plt.tight_layout()
plt.savefig('plots_final/mcc_latent.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight',  dpi=600)