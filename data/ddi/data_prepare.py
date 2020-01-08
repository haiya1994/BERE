"""
Data preprocessing (For training and test):
1. Load the pretrained word vectors.
2. Replace each word with an ID.
3. Count the basic statistics of the input data.
4. Transform the file with '.json' format into '.pt' format.
"""


import sys

import config

from dataset import *

sys.path.append("../..")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

vocab = Vocab(label_path='label2id.json', emb_path='../PubMed-and-PMC-w2v.bin')

logging.info('Number of classes: {}'.format(vocab.class_num))

if config.BAG_MODE:
    DatasetClass = REDataset_BAG
else:
    DatasetClass = REDataset_INS


def dump_dataset(data_name):
    dataset = DatasetClass(vocab, data_dir='.', data_name=data_name + '.json', max_length=config.MAX_LENGTH)
    torch.save(dataset, data_name + '.pt')


dump_dataset('train')
dump_dataset('valid')
dump_dataset('test')

vocab.post_process()
logging.info('Used pretrained vectors: {}*{}'.format(vocab.word_num, vocab.word_dim))
torch.save(vocab, 'vocab.pt')
