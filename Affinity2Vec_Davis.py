#!/usr/bin/env python
# coding: utf-8
######################### Affinity2Vec Model Using Python ##########################
#  First author:  maha.thafar@kaust.edu.sa
   
# Done: April, 2021
 
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

## Affinity2Vec_Hybrid source code for the best results using Davis Dataset

def main():

	# get the parameters from the user
	    # args = parse_args()

	# ## Davis Dataset
	# ### Drug No. 68 --- Protein No. 442,  Known Interaction  30,056
	# ## Read the data...
	DrugID = np.genfromtxt("Input/Davis/drug_PubChem_CIDs.txt",delimiter='\n', dtype=str)
	ProteinID = np.genfromtxt("Input/Davis/target_gene_names.txt", delimiter='\n',dtype=str)

	DrPr_file = "Input/Davis/drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt"
	DrPr_Matrix = np.loadtxt(DrPr_file,dtype=str ,skiprows=0)

	Dsim_file = "Input/Davis/drug-drug_similarities_2D.txt"
	Dr_SimM = np.loadtxt(Dsim_file, dtype=float ,skiprows=0)

	Psim_file = "Input/Davis/target-target_similarities_WS.txt"
	Pr_SimM = np.loadtxt(Psim_file, delimiter=" ",dtype=float ,skiprows=0)

	ProteinID = pd.DataFrame(ProteinID)
	ProteinID['Protein NO.'] = ProteinID.index+1

	#ProteinID = ProteinID.reset_index(level = 0, column='Protein NO.', inplace=False)
	ProteinID.rename(columns={0:'Protein Name'},  inplace=True)
	ProteinID['Protein NO.'] = ProteinID['Protein NO.'].astype(str)

	ProteinNO = ProteinID['Protein NO.']

	Pr_SimM = normalizedMatrix(Pr_SimM)

    # create 2 dictionaries for drugs. First one the keys are their order numbers
    #the second  one the keys are their names -- same for targets
	drugID = dict([(d, i) for i, d in enumerate(DrugID)])
	targetID = dict([(t, i) for i, t in enumerate(ProteinNO)])

	AllDrPr = list(itertools.product(DrugID,ProteinNO))

	allD = list(set(DrugID))
	allT = list(set(ProteinNO))
	    
	label = []
	pairX = []
	for i in range(DrPr_Matrix.shape[0]):
	    for j in range(DrPr_Matrix.shape[1]):
	        d = DrugID[i]
	        p = ProteinNO[j]
	        lab = DrPr_Matrix[i][j]
	        pair = d,p
	        label.append(lab)
	        pairX.append(pair)


	# prepare X = pairs, Y = labels and build the random forest model
	X = np.asarray(pairX) #keys --> in jupyter same order but not all compilers!!
	Y = np.asarray(label)

	print('dimensions of all pairs', X.shape)

	Pr_SimM = keep_sim_threshold(Pr_SimM ,0.04)
	Dr_SimM = keep_sim_threshold(Dr_SimM ,0.35)

	Y_transform = np.array(Y, dtype = np.float64)
	YY = -(np.log10(Y_transform/(math.pow(10,9)))) # YY is the affinity value after transform in 1d form

	# Read Y from the input
	fpath = 'Input/Davis/'
	affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
	affinity = np.array(affinity, dtype=np.float64)
	aff = -(np.log10(affinity/(math.pow(10,9))))

	aff_mat = (aff - aff.min()) / (aff.max() - aff.min())
	DP = (YY - YY.min()) / (YY.max() - YY.min())

	# New affinity score
	aff_exp = np.exp((-1)*aff)
	
	# aff_exp_p = aff_exp/ sum(aff_exp)
	# aff_p = np.array(aff_exp_p)

 #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #### Code to generate Seq2Seq Drugs' Embeddings


   
    #------------------------------------------------------------------------------
    
    ### Code to generate ProtVec Protein' Embeddings




   #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

   # read generated drugs SMILES' embeddings and targets sequences' embeddings
	drEMBED = np.loadtxt('EMBED/Davis/Dr_seq2seq_EMBED.txt')
	prEMBED = np.loadtxt('EMBED/Davis/Pr_ProtVec_EMBED.txt')

	DD_Sim_sm = Cosine_Similarity(drEMBED)
	TT_Sim_sq = Cosine_Similarity(prEMBED)

	# #normalize simiarities to be in positive range [0,1]
	DD_Sim_sm = normalizedMatrix(DD_Sim_sm)
	TT_Sim_sq = normalizedMatrix(TT_Sim_sq)
	#------------------------------------------------------------------------------
	# Create EMbeddings (SMILES + Sequences) FVs 
	FV_targets_sq = np.array(prEMBED, dtype = float)
	FV_drugs_sq = np.array(drEMBED, dtype = float)

	concatenateFV = []
	class_labels = []
	lab = []
	for i,j in itertools.product((range(aff.shape[0])),range(aff.shape[1])):
	    #print(i,j)
	    features = list(FV_drugs_sq[i]) + list(FV_targets_sq[j])
	    concatenateFV.append(features)

	ConcatFV_seq = np.array(concatenateFV)
	print('ConcatFV shape',ConcatFV_seq.shape)
    #_________________________________________________________________________________

    # Define the regressors

	xg_reg = xgb.XGBRegressor(booster = 'gbtree', objective ='reg:squarederror', eval_metric = 'rmse',
                    colsample_bytree = 0.8, learning_rate = 0.03, max_depth = 19, scale_pos_weight = 1, gamma=0,
                     alpha = 5,n_estimators = 855, tree_method='auto',min_child_weight =5, 
                    seed=10, n_jobs=-1) #better & faster

	########################################## Start the Evaluation process ############################################
	# ## Using SAME FOLDS for trainnig and testing as state-of-the-art methods

	test_fold = json.load(open("Input/Davis/folds/test_fold_setting1.txt"))
	train_folds = json.load(open("Input/Davis/folds/train_fold_setting1.txt")) 

	foldCounter = 1     # fold counter

	# Training evalutation metrics
	MSE = []
	CI = []
	# Testing evaluation metrics
	MSE_test = []
	rm2_test = []
	CI_test = []
	aupr_test = []
	thresh = 7

	for tr_counter in range(len(train_folds)):

		print('===================================================')
		print("** Working with Fold %i : **" %foldCounter)

		train_5folds = deepcopy(train_folds)
		valid_index = train_folds[tr_counter]
		trainingSET = train_5folds.pop(tr_counter)
		train_index = list(itertools.chain.from_iterable(train_5folds))
		test_index = test_fold

		#first thing with affinity train to remove all edges in test (use it when finding path)
		train_aff_M = Mask_test_index(test_index, X, aff_exp, drugID, targetID)

		# Generate all featres from the matrix multiplication of each path strucutre
		# list for each feature (Graph G1)
		sumDDD, maxDDD = DDD_TTT_sim(Dr_SimM)
		sumTTT, maxTTT= DDD_TTT_sim(Pr_SimM)

		sumDDT,maxDDT = metaPath_Dsim_DT(Dr_SimM,train_aff_M,2)
		sumDTT,maxDTT = metaPath_DT_Tsim(Pr_SimM,train_aff_M,2)

		sumDDDT,_ = metaPath_Dsim_DT(sumDDD,train_aff_M,3)
		_,maxDDDT = metaPath_Dsim_DT(maxDDD,train_aff_M,3)

		sumDTTT,_ = metaPath_DT_Tsim(sumTTT,train_aff_M,3)
		_,maxDTTT = metaPath_DT_Tsim(maxTTT,train_aff_M,3)

		sumDTDT,maxDTDT = metaPath_DTDT(train_aff_M)
		sumDDTT,maxDDTT = metaPath_DDTT(train_aff_M,Dr_SimM,Pr_SimM)
	    #--------------------------------------------------------------------------
	        #     # list for each feature (Graph G2), using seq/smiles cosine similarities

		# print('generate path scores from seq embedding cos sim:')
		# sumDDD2, maxDDD2 = DDD_TTT_sim(DD_Sim_sm)
		# sumTTT2, maxTTT2= DDD_TTT_sim(TT_Sim_sq)

		# sumDDT2,maxDDT2 = metaPath_Dsim_DT(DD_Sim_sm,train_aff_M,2) 
		# sumDTT2,maxDTT2 = metaPath_DT_Tsim(TT_Sim_sq,train_aff_M,2)

		# sumDDDT2,_ = metaPath_Dsim_DT(sumDDD2,train_aff_M,3)
		# _,maxDDDT2 = metaPath_Dsim_DT(maxDDD2,train_aff_M,3)

		# sumDTTT2,_ = metaPath_DT_Tsim(sumTTT2,train_aff_M,3)
		# _,maxDTTT2 = metaPath_DT_Tsim(maxTTT2,train_aff_M,3)

		# sumDTDT2,maxDTDT2 = metaPath_DTDT(train_aff_M)
		# sumDDTT2,maxDDTT2 = metaPath_DDTT(train_aff_M,DD_Sim_sm,TT_Sim_sq)

	    ### Build feature vector and class labels - (path scores from G1)
		DT_score = []
		lab = []
		for i in range(aff_mat.shape[0]):
			for j in range(aff_mat.shape[1]):
				pair_scores = sumDDT[i][j],sumDDDT[i][j],sumDTT[i][j],sumDTTT[i][j], sumDDTT[i][j], sumDTDT[i][j],\
								maxDDT[i][j],maxDDDT[i][j], maxDTT[i][j],maxDTTT[i][j],maxDDTT[i][j],maxDTDT[i][j]

				DT_score.append(pair_scores) 
				# same label as the begining
				label = aff[i][j]
				lab.append(label)     
		################################################
		FV= np.asarray(DT_score)
			#XX = np.array(DT_score, dtype=np.float32)
		XX = np.concatenate((FV,ConcatFV_seq), axis=1)
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
		print('(Validation set) MSE: %.3f' % mean_squared_error(YY[valid_index], predictedY))
		MSE.append(mean_squared_error(YY[valid_index], predictedY))

		print('ML Concordance index: %3f' % concordance_index(YY[valid_index], predictedY))
		CI.append(concordance_index(YY[valid_index], predictedY))

		##################################### Test Set Evaluation ##################################
		print("--------------- Test Set Evaluation #----------------")
		X_test_transform = min_max_scaler.transform(XX[test_index])
		predictedY_test= xg_reg.predict(X_test_transform)


		## Plot the predicted values of test set vs. actual affinity values 
		# print('data samples', len(predictedY_test))
		# plt.figure()
		# plt.scatter(YY[test_index], predictedY_test,color = "blue", s=3)
		# plt.scatter(YY[test_index], YY[test_index], color = "red", s=1)
		# plt.title("Predicted vs Actual for Davis (test set)")
		# plt.xlabel("Actual Affinities")
		# plt.ylabel("Predicted Affinities")
		# plt.show()

		# Evaluation Metrics (MSE, RMSE, CI, and R2)
		# The mean squared error
		print('Mean Squared Error: %.4f' % mean_squared_error(YY[test_index], predictedY_test))
		MSE_test.append(mean_squared_error(YY[test_index], predictedY_test))

		print('Concordance index: %4f' % concordance_index(YY[test_index], predictedY_test))
		CI_test.append(concordance_index(YY[test_index], predictedY_test))

		print('aupr: %.4f' % get_aupr(YY[test_index], predictedY_test,thresh))
		aupr_test.append(get_aupr(YY[test_index], predictedY_test,thresh))

		print('rm2: %.4f' % get_rm2(YY[test_index], predictedY_test))
		rm2_test.append(get_rm2(YY[test_index], predictedY_test))

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
#----------------MAIN Function --------------------------------------------------

if __name__ == "__main__":
    main()

################################## END ##################################################
