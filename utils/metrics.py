import torch
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

def get_ica_sources(pred_z, ica_transform):
    return ica_transform.transform(pred_z)

def linear_regression_approx(x, y, fit_intercept=False):
    reg= LinearRegression(fit_intercept= fit_intercept).fit(x, y)
    return reg.predict(x)

def get_test_predictions(model, dataset):
    
    model.eval()
    true_y=[]
    pred_y=[]
    true_z= []
    pred_z= []
    for batch_idx, (x, y, z) in enumerate(dataset):
        
        with torch.no_grad():
            true_z.append(z)
            pred_z.append(model.rep_net(x))

            true_y.append(y)
            pred_y.append(model(x))

    true_z= torch.cat(true_z).detach().numpy()
    pred_z= torch.cat(pred_z).detach().numpy()

    true_y= torch.cat(true_y).detach().numpy()
    pred_y= torch.cat(pred_y).detach().numpy()
    
    return true_y, pred_y, true_z, pred_z

def get_label_prediction_error(pred_y, true_y):
    
    return np.sqrt(np.mean((true_y - pred_y)**2)), r2_score(true_y, pred_y) 

def get_label_prediction_error_ica(ica_z, true_y):
    
    pred_y= linear_regression_approx(ica_z, true_y, fit_intercept=True)
    return np.sqrt(np.mean((true_y - pred_y)**2)), r2_score(true_y, pred_y)

def get_latent_prediction_error(pred_z, true_z):
    
    pred_z= linear_regression_approx(pred_z, true_z)    
    return np.sqrt(np.mean((true_z - pred_z)**2)), r2_score(true_z, pred_z) 