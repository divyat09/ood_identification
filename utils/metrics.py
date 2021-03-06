import torch
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
from sklearn.feature_selection import mutual_info_regression

def get_pca_sources(pred_z, pca_transform):
    
    return { 'tr': pca_transform.transform(pred_z['tr']), 'te': pca_transform.transform(pred_z['te']) }    

def get_ica_sources(pred_z, ica_transform):
    
    return { 'tr': ica_transform.transform(pred_z['tr']), 'te': ica_transform.transform(pred_z['te']) }

def linear_regression_approx(x, y, fit_intercept=False):
    reg= LinearRegression(fit_intercept= fit_intercept).fit(x, y)
    return reg

def get_predictions_check(train_dataset, test_dataset):    
    
    true_x={'tr':[], 'te':[]}
    true_z= {'tr':[], 'te':[]}
    
    data_case_list= ['train', 'test']
    for data_case in data_case_list:
        
        if data_case == 'train':
            dataset= train_dataset
            key='tr'
        elif data_case == 'test':
            dataset= test_dataset
            key='te'
    
        for batch_idx, (x, y, z) in enumerate(dataset):

            with torch.no_grad():
                                
                true_x[key].append(x)
                true_z[key].append(z)

        true_x[key]= torch.cat(true_x[key]).detach().numpy()
        true_z[key]= torch.cat(true_z[key]).detach().numpy()
    
    return true_x, true_z


def get_predictions(model, train_dataset, test_dataset, invertible_model):
    
    #Divyat: Question about batch-normalization; should I turn it off or not while collecting these samples    
    model.eval()
    
    true_y={'tr':[], 'te':[]}
    pred_y={'tr':[], 'te':[]}
    true_z= {'tr':[], 'te':[]}
    pred_z= {'tr':[], 'te':[]}
    
    data_case_list= ['train', 'test']
    for data_case in data_case_list:
        
        if data_case == 'train':
            dataset= train_dataset
            key='tr'
        elif data_case == 'test':
            dataset= test_dataset
            key='te'
    
        for batch_idx, (x, y, z) in enumerate(dataset):

            with torch.no_grad():
                true_z[key].append(z)
                pred_z[key].append(model.rep_net(x))

                true_y[key].append(y)
                
                if invertible_model:
                    out, _, _= model(x)
                    pred_y[key].append(out)                    
                else:
                    out, _= model(x)
                    pred_y[key].append(out)

        true_z[key]= torch.cat(true_z[key]).detach().numpy()
        pred_z[key]= torch.cat(pred_z[key]).detach().numpy()

        true_y[key]= torch.cat(true_y[key]).detach().numpy()
        pred_y[key]= torch.cat(pred_y[key]).detach().numpy()
    
#     print('Sanity Check: ')
#     print( true_y['tr'].shape, pred_y['tr'].shape, true_z['tr'].shape, pred_z['tr'].shape )
#     print( true_y['te'].shape, pred_y['te'].shape, true_z['te'].shape, pred_z['te'].shape )
    return true_y, pred_y, true_z, pred_z

def get_direct_prediction_error(pred_score, true_score, case='test', final_task=0):
    
    if case == 'train':
        key= 'tr'
    elif case == 'test':
        key= 'te'
    
    if final_task:        
        target= true_score[key]
        pred= pred_score[key]
        pred= 1/(1+np.exp(-1*pred))
        pred[pred>=0.5]= 1 
        pred[pred<0.5]= 0
        res= 100*np.mean(pred==target)
        
#         res= 100*np.sum( np.argmax(pred_score[key], axis=1) == true_score[key] )/true_score[key].shape[0]
        
    else:
        res= np.sqrt(np.mean((true_score[key] - pred_score[key])**2)), r2_score(true_score[key], pred_score[key])
    
    return res

def get_indirect_prediction_error(pred_latent, true_score, case='test', final_task=0):
    
    if case == 'train':
        key= 'tr'
    elif case == 'test':
        key= 'te'
        
    if final_task:
        
#         clf= LogisticRegression(multi_class='multinomial', penalty='none').fit(pred_latent['tr'], true_score['tr'])
#         pred_score= clf.predict(pred_latent[key])
#         res= 100*np.sum( pred_score == true_score[key] )/true_score[key].shape[0]
        N= true_score[key].shape[0]
        D= true_score[key].shape[1]
        res=0.0
        for idx in range(D):
            clf= LogisticRegression(penalty='none').fit(pred_latent['tr'], true_score['tr'][:, idx])
            pred_score= clf.predict(pred_latent[key])
            res+= 100*np.sum( pred_score == true_score[key][:, idx] )/(N*D)
    else:
        reg= linear_regression_approx(pred_latent['tr'], true_score['tr'], fit_intercept=True)
        pred_score= reg.predict(pred_latent[key])
        res= np.sqrt(np.mean((true_score[key] - pred_score)**2)), r2_score(true_score[key], pred_score)    
    
    return res

    
def get_mi_score(pred_latent, true_latent, case='test'):
    
    if case == 'train':
        key= 'tr'
    elif case == 'test':
        key= 'te'
    
    n= pred_latent[key].shape[0]
    dim= pred_latent[key].shape[1]
    mutual_info= 0.0
    for i in range(dim):
        for j in range(dim):
            if i != j:
                mutual_info+= mutual_info_regression( np.reshape( pred_latent[key][:, i], (n, 1) ), true_latent[key][:, j] )
    
    print('Mutual Information')
    print(mutual_info/(dim**2 - dim))
    return 

    
def get_independence_score(pred_latent, true_latent, case='test'):
    
    if case == 'train':
        key= 'tr'
    elif case == 'test':
        key= 'te'
    
    dim= pred_latent[key].shape[1]
    cross_corr= np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            cross_corr[i,j]= (np.cov( pred_latent[key][:,i], true_latent[key][:,j] )[0,1]) / ( np.std(pred_latent[key][:,i])*np.std(true_latent[key][:,j]) )
    
    print('Independence Score')
    print(cross_corr)
    print(np.linalg.norm( cross_corr - np.eye(dim),  ord='fro'))
    return 

def get_cross_correlation(pred_latent, true_latent, case='test'):
    
    if case == 'train':
        key= 'tr'
    elif case == 'test':
        key= 'te'
    
    dim= pred_latent[key].shape[1]
    cross_corr= np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            cross_corr[i,j]= (np.cov( pred_latent[key][:,i], true_latent[key][:,j] )[0,1]) / ( np.std(pred_latent[key][:,i])*np.std(true_latent[key][:,j]) )
    
    cost= -1*np.abs(cross_corr)
    row_ind, col_ind= linear_sum_assignment(cost)
    
    score= 100*np.sum( -1*cost[row_ind, col_ind].sum() )/(dim)
#     score= 100*np.sum( -1*cost[row_ind, col_ind] > 0.80 )/(dim)
    
    return score
