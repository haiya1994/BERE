"""
Data Transforming (for prediction):
1. Load the statistic file (vocab.pt).
2. Replace each word with an ID.
3. Transform the file with '.json' format into '.pt' format.
"""

import sys

import config

from dataset import *

sys.path.append("../..")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

vocab = torch.load('vocab.pt')
logging.info('Number of classes: {}'.format(vocab.class_num))

if config.BAG_MODE:
    DatasetClass = REDataset_BAG
else:
    DatasetClass = REDataset_INS


def dump_dataset(data_name):
    dataset = DatasetClass(vocab, data_dir='.', data_name=data_name + '.json', max_length=config.MAX_LENGTH)
    torch.save(dataset, data_name + '.pt')


dump_dataset('pmc_nintedanib')
