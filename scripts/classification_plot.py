import matplotlib
import matplotlib.pyplot as plt

num_fc= 2

# data_dim_list= [16]
# num_tasks_list= [8, 12, 16]

data_dim_list= [50]
num_tasks_list= [25, 37, 50]

# data_dim_list= [32]
# num_tasks_list= [16, 24, 32]

# data_dim_list= [24]
# num_tasks_list= [12, 18, 24]

x_ticks=[]
for item in num_tasks_list:
    x_ticks.append(str(item))        

        
res={}
for data_dim in data_dim_list:
    if data_dim not in res.keys():
        res[data_dim]= {}
    for num_tasks in num_tasks_list:
        if num_tasks not in res[data_dim].keys():            
            res[data_dim][num_tasks]= { 'target_pred_acc': {}, 
                                       'ica_target_pred_acc':{},
                                       'pca_target_pred_acc':{},
                                       'latent_pred_score':{}, 
                                       'ica_latent_pred_score':{}, 
                                       'pca_latent_pred_score':{}
                                      }
            
            f= open('results/non_invertible/fc_' + str(num_fc) + '/final_task_1_num_tasks_' + str(num_tasks) + '_data_dim_' + str(data_dim) + '.txt', 'r')
            data= f.readlines()
            
        
            res[data_dim][num_tasks]['target_pred_acc']['mean'] = float(data[-6].split(' ')[-2].replace('\n',''))
            res[data_dim][num_tasks]['target_pred_acc']['std'] = float(data[-6].split(' ')[-1].replace('\n',''))
            
            res[data_dim][num_tasks]['ica_target_pred_acc']['mean'] = float(data[-5].split(' ')[-2].replace('\n',''))
            res[data_dim][num_tasks]['ica_target_pred_acc']['std'] = float(data[-5].split(' ')[-1].replace('\n',''))

            res[data_dim][num_tasks]['pca_target_pred_acc']['mean'] = float(data[-4].split(' ')[-2].replace('\n',''))
            res[data_dim][num_tasks]['pca_target_pred_acc']['std'] = float(data[-4].split(' ')[-1].replace('\n',''))
            

            res[data_dim][num_tasks]['latent_pred_score']['mean'] = float(data[-3].split(' ')[-2].replace('\n',''))
            res[data_dim][num_tasks]['latent_pred_score']['std'] = float(data[-3].split(' ')[-1].replace('\n',''))

            res[data_dim][num_tasks]['ica_latent_pred_score']['mean'] = float(data[-2].split(' ')[-2].replace('\n',''))
            res[data_dim][num_tasks]['ica_latent_pred_score']['std'] = float(data[-2].split(' ')[-1].replace('\n',''))

            res[data_dim][num_tasks]['pca_latent_pred_score']['mean'] = float(data[-1].split(' ')[-2].replace('\n',''))
            res[data_dim][num_tasks]['pca_latent_pred_score']['std'] = float(data[-1].split(' ')[-1].replace('\n',''))
            

fontsize=70
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
    ax.set_ylabel('Accuracy', fontsize=fontsize)
    ax.set_xlabel('Number of Tasks', fontsize=fontsize)
#     ax.set_title('Label Prediction ' + ':' + ' Data Dim: ' + str(data_dim), fontsize=fontsize)
    ax.set_title('Label Prediction', fontsize=fontsize)
    ax.set_ylim(0, 100)
    
    key='target_pred_acc'
    out=[]
    out_err=[]
    for num_tasks in num_tasks_list:
        data= res[data_dim][num_tasks][key]
        out.append(data['mean'])
        out_err.append(data['std'])

    ax.errorbar(x_ticks, out, yerr=out_err, marker= marker_list[0], color='C0', ls='-', lw=20, mfc='w', mew=2.5, ms=60, alpha=0.9, fmt='o--', label='ERM')

    key='ica_target_pred_acc'
    out=[]
    out_err=[]
    for num_tasks in num_tasks_list:
        data= res[data_dim][num_tasks][key]
        out.append(data['mean'])
        out_err.append(data['std'])

    ax.errorbar(x_ticks, out, yerr=out_err, marker= marker_list[1], color='C1', ls='--', lw=20, mfc='w', mew=2.5, ms=60, alpha=0.9, fmt='o--', label='ERM-ICA')

    key='pca_target_pred_acc'
    out=[]
    out_err=[]
    for num_tasks in num_tasks_list:
        data= res[data_dim][num_tasks][key]
        out.append(data['mean'])
        out_err.append(data['std'])

    ax.errorbar(x_ticks, out, yerr=out_err, marker= marker_list[2], color='C2', ls='-.', lw=20, mfc='w', mew=2.5, ms=60, alpha=0.9, label='ERM-PCA')
    
    
    idx=idx+1
    
    
lines, labels = fig.axes[-1].get_legend_handles_labels()    
lgd= fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.20), fontsize=fontsize, ncol=3)
    
plt.tight_layout()
plt.savefig('plots_final/acc_label.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight',  dpi=600)



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