# Evaluation metrics that used to assest the regressors and other metrics in main code

import numpy as np
from copy import deepcopy
import copy
import os
import subprocess
from math import sqrt
from scipy import stats
from sklearn import preprocessing, metrics
print( subprocess.call(["ls", "-l"]))


def mse(y,f):
    
    mse = ((y - f)**2).mean(axis=0)
    return mse
#----------------------------------------------------

def get_cindex(Y, P):
    
    summ = 0
    pair = 0
    
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i != j:
                if(Y[i] > Y[j]):
                    pair +=1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])
            
    if pair != 0:
        return summ/pair
    
    else:
        return 0

#------------------------------------------------------------------------
def get_aupr(Y, P,threshold):
    
    if hasattr(Y, 'A'): Y = Y.A
    if hasattr(P, 'A'): P = P.A
    Y = np.where(Y>threshold, 1, 0)
    Y = Y.ravel()
    P = P.ravel()
    
    f = open("aupr/P_Y.txt", 'w')
    for i in range(Y.shape[0]):
        f.write("%f %d\n" %(P[i], Y[i]))
    f.close()
    
    f = open("aupr/aupr_metric.txt", 'w')
    subprocess.call(["java", "-jar", "aupr/auc.jar", "aupr/P_Y.txt", "list"], stdout=f)
    f.close()
    
    f = open("aupr/aupr_metric.txt")
    lines = f.readlines()
    aucpr = float(lines[-2].split()[-1])
    f.close()
    
    return aucpr

#----------------------------------------------------
def average_AUPRC(y,f):

    """
    Task:    To compute average area under the ROC curves (AUC) given ten
             interaction threshold values from the pKd interval [6 M, 8 M]
             to binarize pKd's into true class labels.
    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])
    Output:  avAUC   average AUC
    """

    thr = np.linspace(6,7,10)
    auc = np.empty(np.shape(thr)); auc[:] = np.nan

    for i in range(len(thr)):
        y_binary = copy.deepcopy(y)
        y_binary = preprocessing.binarize(y_binary.reshape(1,-1), threshold=thr[i], copy=False)[0]
        precision, recall, thresholds = metrics.precision_recall_curve(y_binary, f, pos_label=1)
        auc[i] = metrics.auc(recall, precision)

    avAUPR = np.mean(auc)

    return avAUPR

#------------------------------------------------------------------------------------------

def r_squared_error(y_actual,y_pred):
    
    y_actual = np.array(y_actual)
    y_pred = np.array(y_pred)
    y_actual_mean = [np.mean(y_actual) for y in y_actual]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_actual - y_actual_mean))
    mult = mult * mult

    y_obs_sq = sum((y_actual - y_actual_mean)*(y_actual - y_actual_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)
#----------------------------------------------------

def get_k(y_actual,y_pred):
    
    y_actual = np.array(y_actual)
    y_pred = np.array(y_pred)

    return sum(y_actual*y_pred) / float(sum(y_pred*y_pred))
#----------------------------------------------------

def squared_error_zero(y_actual,y_pred):
    
    k = get_k(y_actual,y_pred)

    y_actual = np.array(y_actual)
    y_pred = np.array(y_pred)
    y_actual_mean = [np.mean(y_actual) for y in y_actual]
    upp = sum((y_actual - (k*y_pred)) * (y_actual - (k* y_pred)))
    down= sum((y_actual - y_actual_mean)*(y_actual - y_actual_mean))

    return 1 - (upp / float(down))
#----------------------------------------------------

def get_rm2(ys_orig,ys_line):
    
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))
#----------------------------------------------------   

def pearson(y,f):
    
    rp = np.corrcoef(y, f)[0,1]
    return rp

###########################EOF#######################################
