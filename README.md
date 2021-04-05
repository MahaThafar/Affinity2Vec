# Affinity2Vec: Drug-Target Binding Affinity Prediction Method Developed using Representation Learning, Graph Mining, and Machine Learning Techniques

 
#### This repositery provides an implementation of Affinity2Vec tool which is described in a research paper:
(Published in ... 2021):

> Affinity2Vec+ is a regression-based and network-bbased.

Received: 15 April 2021                                      
Accepted:               
Published: 


----
This code is implemented using Python 3.8.

For any qutions please contact the first author:


  Maha Thafar

Email: maha.thafar@kaust.edu.sa

Computer, Electrical and Mathematical Sciences and Engineering Division (CEMSE), Computational Bioscience Research Center, Computer (CBRC), King Abdullah University of Science and Technology (KAUST) - Faculty of Computers and Information Technology, Taif University (TU)

----

## Getting Started

### Prerequisites:

There are several required Python packages to run the code:
- gensim
- numpy
- Scikit-learn
- keras
- deepchem
- protVec
- xgboost

These packages can be installed using pip or conda as the follwoing example
```
pip install -r requirements.txt
```
----

### Files Description:
#### *There are Three folders:*

  **1.Input folder:** 
  that includes two folder for 2 datasets include: 
   - Davis dataset,
   - KIBA dataset
     which each one of them has all required data of drug-target binding affinity (in Adjacency matrix format), drug-drug and target-target similarities in (square matrix format), the drugs' SMILES in dictionary format with drugs' IDs, and the proteins' amino-acid sequences in dictionary format with proteins' IDs
  
  **2.Embedding folder:**
  that has also two folders coressponding for two datasets,
     each folder contains the generated seq2seq embeddings for drugs, and generated ProtVec embeddings for proteins. 
  
---
#### *There are 8 files:*
(two main functions, one main for each dataset, and the other functions are same for all datasets which are imported in each main function)

- **load_datasets.py** -->  read the input data including binding affinityies, SMILES, Sequences, and similarities
- **training_functions.py** --> for several training and processing functions such as edgeList, Cosine_similarity, ..
- **pathScores.py** --> to calculate and return all path scores for 6 path structures
- **evaluation.py** --> define all evalution metrics used in our experments.


- **Four main functions**
one for each dataset:
> - DTIs_Main_nr.py
> - DTIs_Main_gpcr.py

---
## Installing:

To get the development environment runining, the code get one parameter from the user which is the dataset name (the defual dataset is nr)
run:

```
python DTIs_Main_nr.py --data nr
```
```
python DTIs_Main_gpcr.py --data gpcr
```


------------------
### For citation:
---
Thafar, M.A., 





----
This code is implemented using Python 2.7.9, but since Python 2.7 will no longer supported after January 2020, It is upgraded to be run using Python 3.6

For any qutions please contact the first author:


  Maha Thafar

Email: maha.thafar@kaust.edu.sa

Computer, Electrical and Mathematical Sciences and Engineering Division (CEMSE), Computational Bioscience Research Center, Computer (CBRC), King Abdullah University of Science and Technology (KAUST) - Faculty of Computers and Information Technology, Taif University (TU)

----

## Getting Started

### Prerequisites:

There are several required Python packages to run the code:
- gensim (for node2vec code)
- numpy
- Scikit-learn
- imblearn
- pandas

These packages can be installed using pip or conda as the follwoing example
```
pip install -r requirements.txt
```
----

### Files Description:
#### *There are Three folders:*

  **1.Input folder:** 
  that includes four folder for 4 datasets include: 
   - Nuclear Receptor dataset (nr),
   - G-protein-coupled receptor (gpcr),
   - Ion Channel (IC), 
   - Enzyme (e)
     which each one of them has all required data of drug-target interactions (in Adjacency matrix and edgelist format) and drug-drug and target-target similarities in (square matrix format) - all matrices include items names.
  
  **2.Embedding folder:**
  that has also four folders coressponding for four datasets,
     each folder contains the generated node2vec embeddings files for each fold of training data
     
  **3.Novel_DTIs folder:**
  that has also four folders coressponding for four datasets, 
     to write the novel DTIs (you should create directory for each dataset)
  
---
#### *There are 10 files:*
(Four main functions, one main for each dataset, and the other functions are same for all datasets which are imported in each main function)

- **load_datasets.py** --> to read the input data including interactions and similarities
- **get_Embedding_FV.py** --> to read the node2vec generated embedding and get the FV for each drug and target (CV random seed=22)
- **training_functions.py** --> for several training and processing functions such as edgeList, Cosine_similarity, ..
- **pathScores.py** --> to calculate and return all path scores for 6 path structures
- **snf_code.py** --> Similarity Network Fusion functions
- **GIP.py** --> to calculate and return gussian interaction profile similarity

- **Four main functions**
one for each dataset:
> - DTIs_Main_nr.py
> - DTIs_Main_gpcr.py
> - DTIs_Main_ic.py
> - DTIs_Main_enzyme.py

---
## Installing:

To get the development environment runining, the code get one parameter from the user which is the dataset name (the defual dataset is nr)
run:

```
python DTIs_Main_nr.py --data nr
```
```
python DTIs_Main_gpcr.py --data gpcr
```
```
python DTIs_Main_ic.py --data ic
```
```
python DTIs_Main_e.py --data e
```

------------------

#### The repositery also provides an example of seq2seq and ProtVec code implemented inside the DTBA prediction code in the folder (DTBA_Embeddings)

 ***About the folder (DTBA_Embeddings):***
 - This example is applied on Davis dataset

 - To run this code:
```
python DTIs_Main.py
```
 
 #### *Note:*
 >  When you run the code the AUPR result could be a little bit different than the other code (DTIs_Main_ic.py) because of randomness in seq2seq when generates the embedding
 

### For citation:
---
> Thafar, M.A., 
---
