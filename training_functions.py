'''
*******************************************************
Some functions that are needed for training process..
*******************************************************
'''
from copy import deepcopy
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import scipy.spatial
from sklearn.preprocessing import MinMaxScaler
import collections

#---------------------------------------------------------------------------------------------
#### ALL needed functions for training process
#---------------------------------------------------------------------------------------------

######### TRANING FUNCTION ###############
def Mask_test_index(test_idx, x, DrTr, drugID, targetID):
    
    DrTr_train = deepcopy(DrTr)
    # get the drug index and target index 
    # mask drug,target = 1 of test data to be 0 (i.e. remove the edge)
    for i in test_idx:

        dr = x[i,0]
        dr_index = drugID[dr]

        tr = x[i,1]
        tr_index = targetID[tr]

        DrTr_train[dr_index, tr_index] = 0
    
    return DrTr_train

##----------------------------------------------------

## normalize simiarities to be in positive range [0,1]
def normalizedMatrix(matrix):
    
#     scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
#     scaler.fit(matrix)
#     normMat = scaler.transform(matrix)
    normMat = (matrix - matrix.min()) / (matrix.max() - matrix.min())

    return normMat

#--------------------------------------------------------
def Strongest_k_sim(Mat,K):
    
    m,n = Mat.shape

    Ssim = np.zeros((m,n))
    for i in range(m):
        index =  np.argsort(Mat[i,:])[-K:] # sort based on strongest k edges
        Ssim[i,index] = Mat[i,index] # keep only the nearest neighbors (strongest k edges)
    
    np.fill_diagonal(Ssim , 1)  
     
    return Ssim

#---------------------------------------------------------------------------------------------

## To compute drug-drug FV cosine similarity and target-target FV cosine similarity

def Cosine_Similarity(Matrix):
    cos_sim_m = np.zeros((Matrix.shape[0],Matrix.shape[0]))
    for item_i in range(Matrix.shape[0]):
        for item_j in range(Matrix.shape[0]):
            cos_sim_m[item_i][item_j] = 1-(scipy.spatial.distance.cosine(Matrix[item_i,:],Matrix[item_j,:]))
    
    return cos_sim_m

#---------------------------------------------------------------------
## Get Similarity of targets or drugs within specific threshold
def keep_sim_threshold(simMat, threshold):

    newMat = np.zeros(simMat.shape)
    #print(newMat)
    for i,x in enumerate(simMat):
    #print(i)
        for j,y in enumerate(x):
            #print(i,j)
#             if y <= threshold:
#                  simMat[i,j] = 0.0
            if y >= threshold:
                newMat[i,j] = simMat[i,j]
            
            # print(newMat)
            
            if (np.count_nonzero(newMat[i]) == 0):
                col = np.argmax(simMat[i])
                # print('col', col)
                newMat[i,col] = simMat[i,col] 

    return newMat
#----------------------------------------------
# Convert weighted edgelist into adjacency matrix

def edgelist_to_adjMat(Wedgelist,ligDic,prDic):
    row_inds = []
    col_inds = []
    adj = np.zeros((len(ligDic.keys()),len(prDic.keys())))
    for element in Wedgelist:
        i = ligDic[element[0]]
        j = prDic[element[1]]
        adj[i][j] = element[2]
        row_inds.append(i)
        col_inds.append(j)
    return adj, row_inds,col_inds 
#------------------------------------------------
#### Encoding Functions ###################
###########################################
def integer_encoding(s,dic,maxLen):
    
    s_labelE = np.zeros(maxLen)
    
    for i,char in enumerate(s[:maxLen]):
        s_labelE[i] = dic[char]
        

    return s_labelE
############################################3
def oneHOT_encoding(s, dic, maxLen):
    
    oneHot_e = np.zeros((maxLen, len(dic)))
    
    for i,char in enumerate(s[:maxLen]):
        oneHot_e[i, (dic[char])-1] = 1
    
    return oneHot_e

#--------------------------------
def get_unique_tokens(allSMILES):
    all_smiles_char = []
    for i in range(len(allSMILES)):
        all_smiles_char += str(allSMILES[i])

    tokens = sorted(list(set(all_smiles_char)))
    num_Tokens = len(tokens)
    
    return tokens, num_Tokens
#--------------------------------------------------------------------
