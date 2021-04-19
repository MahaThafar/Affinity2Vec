#!/usr/bin/env python
# coding: utf-8
######################### Affinity2Vec Model Using Python ##########################
#  First author:  maha.thafar@kaust.edu.sa
   
# Done: March, 2021
 
# Description
# This script predict drug-target binding affinity (DTBA) and evaluate the model using 4 evaluation metrics

###############################################################################
# all needed packages
import pandas as pd
from copy import deepcopy
import math as math
import numpy as np
from math import exp
# import argparse
import json,csv, pickle
import itertools,collections
import matplotlib.pyplot as plt

# ML packages
from sklearn.metrics import *
from sklearn.metrics import mean_squared_error, r2_score
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
# from joblib import Parallel, delayed
# import multiprocessing
import xgboost as xgb

# Similarity and normalization packages
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.preprocessing import MinMaxScaler, minmax_scale
import scipy
from scipy import *

# DL Keras
from keras.wrappers.scikit_learn import KerasRegressor
from keras.preprocessing.sequence import skipgrams
from keras.layers import Embedding, Input, Reshape, Dense, merge
from keras.models import Sequential, Model

# for protein embedding
import biovec

# for drugs embedding
import tensorflow as tf
from tensorflow import keras
import deepchem as dc
from deepchem.feat import Featurizer
from deepchem.models.optimizers import ExponentialDecay, Adam
from deepchem.models.seqtoseq import AspuruGuzikAutoEncoder
from deepchem.metrics import to_one_hot
from deepchem.models.graph_models import GraphConvModel,L2Loss,Dense,Reshape,Dropout 

# Import my files
from training_functions import *
from pathScores_functions import *
from evaluation import *

######################################## START MAIN #########################################
#############################################################################################

## Affinity2Vec_Pscorex source code for the best results using KIBA dataset

def main():

# get the parameters from the user
    # args = parse_args()

    # ##  Kinase Inhibitor BioActivity (KIBA) Dataset
    # ### Drug No. 2116 --- Protein No. 229,  Known Interaction  118,254
    # 
    Drug_Dict = eval(open("Input/KIBA/ligands_can.txt").read())
    DrugID = list(Drug_Dict.keys())

    Protein_Dict = eval(open("Input/KIBA/proteins.txt").read())
    ProteinID = list(Protein_Dict.keys())

    DrPr_file = "Input/KIBA/kiba_binding_affinity_v2.txt"
    DrPr_Matrix = np.loadtxt(DrPr_file,dtype=np.float32 ,skiprows=0)

    Dsim_file = "Input/KIBA/kiba_drug_sim.txt"
    Dr_SimM = np.loadtxt(Dsim_file, dtype=np.float32 ,skiprows=0)

    Psim_file = "Input/KIBA/kiba_target_sim.txt"
    Pr_SimM = np.loadtxt(Psim_file, delimiter='\t',dtype=np.float32 ,skiprows=0)
    #-------------------------------------------------------------------------

    # create 2 dictionaries for drugs. First one the keys are their order numbers
    #the second  one the keys are their names -- same for targets

    drugID = dict([(d, i) for i, d in enumerate(DrugID)])
    targetID = dict([(t, i) for i, t in enumerate(ProteinID)])

    AllDrPr = list(itertools.product(DrugID,ProteinID))
    allD = list(set(DrugID))
    allT = list(set(ProteinID))

    label = []
    pairX = []
    X_ind = []
    counter = 0

    for i in range(DrPr_Matrix.shape[0]):
        for j in range(DrPr_Matrix.shape[1]):
            
            d = DrugID[i]
            p = ProteinID[j]
            lab = DrPr_Matrix[i][j] 
            pair = d, p
            edgeList = d, p,  lab
            label.append(lab)
            pairX.append(pair)
            X_ind.append(counter)
            counter = counter+1

    # prepare X = pairs, Y = labels and build the random forest model
    X_ind = np.asarray(X_ind) #keys --> in jupyter same order but not all compilers!!
    Y_nan = np.asarray(label)
    Y = np.nan_to_num(Y_nan)  #### for labels

    Pr_SimM = normalizedMatrix(Pr_SimM)
    Pr_SimM = keep_sim_threshold(Pr_SimM ,0.08)
    Dr_SimM = keep_sim_threshold(Dr_SimM ,0.3)

    Y = np.array(Y, dtype = np.float32)
    Y_nan = np.array(Y_nan, dtype = np.float32)

    # Read Y from the input
    fpath = 'Input/KIBA/'
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
    label_row_inds, label_col_inds = np.where(np.isnan(affinity)==False)
    affinity = np.array(affinity, dtype=np.float32)

    aff_df = pd.DataFrame(affinity)

    column_maxes = aff_df.max()
    df_max = column_maxes.max()

    column_mins = aff_df.min()
    df_min = column_mins.min()

    # Preprocess binding affinity values to be consistent with graph edges' weights
    norm_aff_df = (aff_df - df_min) / (df_max - df_min)

    aff_exp = np.exp((-1)*norm_aff_df)
    #aff_exp = aff_exp/ sum(aff_exp)

    aff_exp = pd.DataFrame(aff_exp)
    aff_p = aff_exp.fillna(0)
    aff_p = np.array(aff_p)

    #____________________ Validation and Feature Extraction ________________

    XX = np.asarray(X_ind)
    XX = np.reshape(XX, (X_ind.shape[0], 1))
    X_pair = np.array(pairX)

    Y_nan = np.reshape(Y_nan, (X_ind.shape[0], 1))
    XY_nan = np.concatenate((XX, Y_nan),axis=1)
    XY_nan_df = pd.DataFrame(XY_nan)

    df = XY_nan_df[XY_nan_df[1].notna()]
    Y_df_s = df[1]
    Y_s = np.array(Y_df_s)

    df_X_s = df.drop(columns=1)
    X_s = np.array(df_X_s)

    print('number of rows & cols with no nan:', len(label_row_inds))

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #### Code to generate Seq2Seq Drugs' Embeddings


   
    #------------------------------------------------------------------------------
    
    ### Code to generate ProtVec Protein' Embeddings




   #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

   # read generated drugs' embeddings and targets' embeddings
    drEMBED = np.loadtxt('EMBED/KIBA/Dr_seq2seq_EMBED.txt')
    prEMBED = np.loadtxt('EMBED/KIBA/Pr_ProtVec_EMBED.txt')

    DD_Sim_sm = Cosine_Similarity(drEMBED)
    TT_Sim_sq = Cosine_Similarity(prEMBED)

    # #normalize simiarities to be in positive range [0,1]
    DD_Sim_sm = normalizedMatrix(DD_Sim_sm)
    TT_Sim_sq = normalizedMatrix(TT_Sim_sq)
    #_________________________________________________________________________________

    # Define the regressors
    xg_reg = xgb.XGBRegressor(booster = 'gbtree', objective ='reg:squarederror', eval_metric = 'rmse',
                        colsample_bytree = 0.9, learning_rate = 0.04, max_depth = 18, scale_pos_weight = 0, gamma=0,
                         alpha = 5,n_estimators = 1005, tree_method='auto',min_child_weight =5 ,seed=10)

    ########################################## Start the Evaluation process ############################################
    # ## Using SAME FOLDS for trainnig and testing as state-of-the-art methods

    test_fold = json.load(open("Input/KIBA/folds/test_fold_setting1.txt"))
    train_folds = json.load(open("Input/KIBA/folds/train_fold_setting1.txt")) 

    foldCounter = 1     # fold counter
    
    # Training evalutation metrics
    MSE = []
    CI = []
    # Testing evaluation metrics
    MSE_test = []
    rm2_test = []
    CI_test = []
    aupr_test = []
    thresh = 12.1

    for tr_counter in range(len(train_folds)):
        
        print('===================================================')
        print("** Working with Fold %i : **" %foldCounter)
        
        train_5folds = deepcopy(train_folds)
        valid_index = train_folds[tr_counter]
        trainingSET = train_5folds.pop(tr_counter)
        train_index = list(itertools.chain.from_iterable(train_5folds))
        test_index = test_fold
        
        #first thing with affinity train to remove all edges in test (use it when finding path)
        train_aff_M = Mask_test_index(test_index, X_pair, aff_p, drugID, targetID)
        
        # Generate all featres from the matrix multiplication of each path strucutre
        # list for each feature (Graph G2)
        sumDDD, maxDDD = DDD_TTT_sim(DD_Sim_sm)
        sumTTT, maxTTT= DDD_TTT_sim(TT_Sim_sq)

        sumDDT,maxDDT = metaPath_Dsim_DT(DD_Sim_sm,train_aff_M,2)
        sumDTT,maxDTT = metaPath_DT_Tsim(TT_Sim_sq,train_aff_M,2)

        sumDDDT,_ = metaPath_Dsim_DT(sumDDD,train_aff_M,3)
        _,maxDDDT = metaPath_Dsim_DT(maxDDD,train_aff_M,3)

        sumDTTT,_ = metaPath_DT_Tsim(sumTTT,train_aff_M,3)
        _,maxDTTT = metaPath_DT_Tsim(maxTTT,train_aff_M,3)

        sumDTDT,maxDTDT = metaPath_DTDT(train_aff_M)
        sumDDTT,maxDDTT = metaPath_DDTT(train_aff_M,DD_Sim_sm,TT_Sim_sq)
       
        ### Build feature vector and class labels
        DT_score = []
        lab = []
        for i,j in zip(label_row_inds,label_col_inds): #di
            #for j in label_col_inds: #tj        
            pair_scores = (sumDDT[i][j],sumDDDT[i][j],sumDTT[i][j],sumDTTT[i][j], sumDDTT[i][j],sumDTDT[i][j],
            				maxDDT[i][j],maxDDDT[i][j],maxDTT[i][j],maxDTTT[i][j],maxDDTT[i][j],maxDTDT[i][j])

            DT_score.append(pair_scores)
            label = affinity[i][j]
            lab.append(label)

        XX = np.array(DT_score, dtype=np.float32)
        YY = np.array(lab,dtype=np.float32)

        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit(XX[train_index])
        X_train_transform = min_max_scaler.transform(XX[train_index])

        X_valid_transform = min_max_scaler.transform(XX[valid_index])

        ################################################ Validation SET Evaluation ############################################
        # Train the model using the training sets
        xg_reg.fit(X_train_transform, YY[train_index])
       
        # Make predictions using the testing set
        predictedY= xg_reg.predict(X_valid_transform)

        # Evaluation Metric MSE for validation data
        # The mean squared error
        print('(Validation set) MSE: %.4f' % mean_squared_error(YY[valid_index], predictedY))
        MSE.append(mean_squared_error(YY[valid_index], predictedY))

        # print('ML Concordance index: %3f' % concordance_index(YY[test_index], predictedY))
        CI.append(concordance_index(YY[valid_index], predictedY))

        ##################################### Test Set Evaluation ##################################
        print("--------------- Test Set Evaluation #----------------")
        X_test_transform = min_max_scaler.transform(XX[test_index])
        predictedY= xg_reg.predict(X_test_transform)


        ## Plot the predicted values of test set vs. actual affinity values 
        plt.figure()
        plt.scatter(YY[test_index], predictedY,color = "blue", s=3)
        plt.scatter(YY[test_index], YY[test_index], color = "red", s=1)
        plt.title("Predicted vs Actual for KIBA (test set)")
        plt.xlabel("Actual Affinities")
        plt.ylabel("Predicted Affinities")
        plt.show()
        
        # Evaluation Metrics (MSE, RMSE, CI, and R2)
        # The mean squared error
        print('Mean Squared Error: %.4f' % mean_squared_error(YY[test_index], predictedY))
        MSE_test.append(mean_squared_error(YY[test_index], predictedY))

        print('Concordance index: %4f' % concordance_index(YY[test_index], predictedY))
        CI_test.append(concordance_index(YY[test_index], predictedY))

        print('aupr: %.4f' % get_aupr(YY[test_index], predictedY,thresh))
        aupr_test.append(get_aupr(YY[test_index], predictedY,thresh))

        print('rm2: %.4f' % get_rm2(YY[test_index], predictedY))
        rm2_test.append(get_rm2(YY[test_index], predictedY))

        foldCounter = foldCounter +1

    print('\n*************************************************')   
    print('\nAverage results for validation data:')
    print("MSE = " + str( np.array(MSE).mean().round(decimals=3) ))
    print("CI = " + str( np.array(CI).mean().round(decimals=3) ))
    #<<<<<<<<<<<<<<<<<<<<<Test
    print('\n*************************************************')
    print('Average results for testing data:')
    print("MSE = " + str( np.array(MSE_test).mean().round(decimals=3) ))
    print("CI = " + str( np.array(CI_test).mean().round(decimals=3) ))
    print("rm2 = " + str( np.array(rm2_test).mean().round(decimals=3) ))
    print("AUPR = " + str( np.array(aupr_test).mean().round(decimals=3) ))
#########################################################################################

if __name__ == "__main__":
    main()

################################## END ##################################################