# BERE
Implementation of the paper [A novel machine learning framework for automated biomedical relation extraction from large-scale literature repositories](https://www.nature.com/articles/s42256-020-0189-y).

## Environments
Tested on a linux server with GeForce GTX 1080 and the running environment is as follows:

- Python    3.5.2

- PyTorch   1.0.0

- sklearn   0.20.2

- numpy     1.15.4

- cuda      9.0

## Installation Guide
1. Download the pretrained word embedding `PubMed-and-PMC-w2v.bin` from http://evexdb.org/pmresources/vec-space-models/ and put it in `./data/`.

2. Download the complete DTI dataset from https://www.aliyundrive.com/s/QqSG9H3guEP and put it in `./data/dti/`.

## How to Run
[DDI Expirement](less than 1h for one training)
1. Run  `./data/ddi/data_prepare.py` to preprocess the DDI dataset.

2. Run `./train_ddi.py` to train BERE with different learning rates.

3. Run `./test_ddi.py` to test BERE with the best model.

&nbsp;

[DTI Expirement](taking 10~20h before convergence)
1. Run  `./data/dti/data_prepare.py` to preprocess the DTI dataset.

2. Run `./train_dti.py` to train BERE with different learning rates.

3. Run `./test_dti.py` to test BERE with the best model.

&nbsp;

[Demo of DTI Prediction]

1. Train the model on the DTI dataset.

2. Run `./data/dti/transform.py` to preprocess the `pmc_nintedanib` dataset.

3. Run `./predict.py` to predict the targets of the drug nintedanib by the well-trained model.

## Data Description
- `PubMed-and-PMC-w2v.bin`: The pretrained word embedding.
- `train.json`, `valid.json`, `test.json`: The original dataset.
- `label2id.json`: The label file.
- `pmc_nintedanib.json`: The data for DTI Prediction demo.
- `tree_examples.json`: The data for visualization demo.
- `config.py`: The hyper-parameter settings.

## File Description
- `./data/`: This directory contains DDI dataset, DTI dataset and pretrained word embedding.

- `./network/`: This directory contains the codes of our model.

- `./checkpoint/`(generated): This directory contains the checkpoints of model in the training process.

- `./result/`(generated): This directory contains the test results and prediction results

- `./output/`(generated): This directory contains the prediction results, which is more convenient for reading. 

- `./train_ddi.py`: This is a demo for training the BERE on DDI dataset.

- `./train_dti.py`: This is a demo for testing the BERE on DDI dataset.

- `./test_ddi.py`: This is a demo for training the BERE on DTI dataset.

- `./test_dti.py`: This is a demo for testing the BERE on DTI dataset.

- `./predict.py`: This is a demo for predicting the targets of the drug nintedanib.

- `./plot_pr.py`: This file is used to plot the precision-recall curve of the results in `./result/`.

- `./visualize.py`(optional): This file is used for the visualization of word attention, sentence attention and sentence tree structures.

## Notes
- The full datasets for discovering novel DTIs is available from the corresponding authors upon request. 
- If you have any other questions or comments, please feel free to email Lixiang Hong (honglx17[at]mails[dot]tsinghua[dot]edu[dot]cn) and/or Jianyang Zeng (zengjy321[at]tsinghua[dot]edu[dot]cn).
