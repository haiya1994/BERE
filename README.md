# BERE
Implementation of the paper [BERE: A novel machine learning framework for accurate biomedical entity relation extraction].

## Environment
Tested on Python 3.5.2 and PyTorch 1.0.0.

## Data
The download link of full datasets: https://cloud.tsinghua.edu.cn/d/9a117f17200e4b55b79f/.

The download link of word embedding: http://evexdb.org/pmresources/vec-space-models/.

The download link of DDI'13 dataset can also be found in: https://github.com/arwhirang/DDI-recursive-NN/. We transformed them into '.json'.

## How to Run
Download 'PubMed-and-PMC-w2v.bin' from http://evexdb.org/pmresources/vec-space-models/ and put it at data/


[DDI Expirement]

Run data/ddi/data_prepare.py for data preparation

Run train_ddi.py for training

Run test_ddi.py for testing


[DTI Expirement]

Run train_dti.py for training

Run test_dti.py for testing

Run predict.py for prediction
