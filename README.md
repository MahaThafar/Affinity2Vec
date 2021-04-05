# Affinity2Vec Method
Drug-target binding affinity prediction using representation learning, graph mining, and machine learning

 
#### This repositery provides an implementation of Affinity2Vec tool which is described in a research paper:
(Published in ... 2021):

> Affinity2Vec+: a regression-based and network-bbased method to predict Drug-Target binding affinity using representation learning, graph mining, and machine learning


---

#### The repositery also provides an example of seq2seq and ProtVec code implemented inside the DTBA prediction code in the folder (DTBA_Embeddings)

 ***About the folder (DTBA_Embeddings):***
 - This example is applied on Davis dataset

 - To run this code:
```
python DTIs_Main.py
```

 
 #### *Note:*
 >  When you run the code the AUPR result could be a little bit different than the other code (DTIs_Main_ic.py) because of randomness in seq2seq when generates the embedding
 

---

#### For original node2vec code to generate new embeddings instead of reading generated embedding you can visit:

(all details to run the code as well as required parameters are provided with node2vec source code)

https://github.com/aditya-grover/node2vec

---

### IF you use any part of this code please cite:
Thafar, M.A., 
