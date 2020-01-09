# BERE
Implementation of the paper [BERE: A novel machine learning framework for accurate biomedical entity relation extraction].

## Environment
Tested on Python 3.5.2 and PyTorch 1.0.0.

## Data
The download link of full datasets: https://cloud.tsinghua.edu.cn/d/9a117f17200e4b55b79f/.

The download link of word embedding: http://evexdb.org/pmresources/vec-space-models/. Please put it in /data/.

The download link of DDI'13 dataset can also be found in: https://github.com/arwhirang/DDI-recursive-NN/.

## How to Run
[DDI Expirement]

Run data/ddi/data_prepare.py

Run train_ddi.py for training

Run test_ddi.py for testing

&nbsp;

[DTI Expirement]

Run data/ati/data_prepare.py

Run train_dti.py for training

Run test_dti.py for testing

Run predict.py for prediction
