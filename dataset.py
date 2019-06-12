import json
import logging
import os
import random
import re
from itertools import chain

import gensim
import numpy as np
import torch
from nltk import pos_tag
from torch.utils.data import Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


class Vocab(object):
    def __init__(self, label_path, emb_path):
        self.pad_id = 0
        self.unk_id = 1
        self.ent1_id = 2
        self.ent2_id = 3

        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(emb_path,
                                                                        binary=True)
        self.word_dim = self.word2vec.vector_size

        self.word2id = {'<pad>': self.pad_id, '<unk>': self.unk_id, '<ent1>': self.ent1_id, '<ent2>': self.ent2_id}
        self.tag2id = {'<pad>': self.pad_id, '<unk>': self.unk_id, '<ent1>': self.ent1_id, '<ent2>': self.ent2_id}

        self.vectors = [np.zeros(self.word_dim, dtype=np.float32), np.random.normal(size=self.word_dim),
                        np.random.normal(size=self.word_dim), np.random.normal(size=self.word_dim)]

        self.label2id = json.load(open(label_path, 'r'))
        self.NA_id = self.label2id['NA']

        self.class_num = len(self.label2id)
        self.freeze = False

    def get_id2label(self):
        id2label = {}
        for label, id in self.label2id.items():
            id2label[id] = label
        return id2label

    def add_tags(self, tags):
        for tag in tags:
            if tag not in self.tag2id:
                self.tag2id[tag] = len(self.tag2id)

    def add_words(self, words):
        for word in words:
            if word in self.word2vec and word not in self.word2id:
                self.word2id[word] = len(self.word2id)
                self.vectors.append(self.word2vec[word])

    def post_process(self):
        self.word2vec = None
        self.freeze = True

        self.word_num = len(self.word2id)
        self.tag_num = len(self.tag2id)

        self.vectors = torch.FloatTensor(self.vectors)


class REDataset(Dataset):
    def __init__(self, vocab, data_dir, data_name, max_length, sort=True):
        self.max_length = max_length

        self.pad_id = vocab.pad_id

        data_path = os.path.join(data_dir, data_name)
        data = json.load(open(data_path, 'r'))

        self._data = []

        logging.info('Process: {}'.format(data_path))
        self.process(vocab, data)

        if sort:
            self._data.sort(key=lambda a: np.max(a[4]), reverse=True)

    def process(self, vocab, data):
        raise NotImplementedError

    def make_new_sent(self, sent, name1, name2, pos1, pos2):
        assert pos1 <= pos2
        new_sent = '{0} {1} {2} {3} {4}'.format(sent[:pos1[0]], name1, sent[pos1[1]:pos2[0]], name2,
                                                sent[pos2[1]:])
        return new_sent

    def convert(self, vocab, ins):
        head, tail, sent, rel = ins['head'], ins['tail'], ins['sentence'].strip(), ins['relation']

        head_word = head['word']
        tail_word = tail['word']

        head_pos = sent.index(head_word)
        head_pos = [head_pos, head_pos + len(head_word)]
        tail_pos = sent.index(tail_word)
        tail_pos = [tail_pos, tail_pos + len(tail_word)]

        if head_pos <= tail_pos:
            sent = self.make_new_sent(sent, '<ent1>', '<ent2>', head_pos, tail_pos)
        else:
            sent = self.make_new_sent(sent, '<ent2>', '<ent1>', tail_pos, head_pos)

        sent = re.sub(r"\s+", r" ", sent).strip().split()
        tags = [item[1] for item in pos_tag(sent)]

        head_pos = sent.index('<ent1>')
        tail_pos = sent.index('<ent2>')

        tags[head_pos] = '<ent1>'
        tags[tail_pos] = '<ent2>'

        head_pos = min(self.max_length - 1, head_pos)
        tail_pos = min(self.max_length - 1, tail_pos)

        sent = sent[:self.max_length]
        tags = tags[:self.max_length]
        length = len(sent)

        if not vocab.freeze:
            vocab.add_words(sent)
            vocab.add_tags(tags)

        sent = [vocab.word2id.get(w, vocab.unk_id) for w in sent]
        tags = [vocab.tag2id.get(w, vocab.unk_id) for w in tags]

        label = vocab.label2id.get(rel, vocab.NA_id)

        pos1 = [i - head_pos + self.max_length for i in range(length)]
        pos2 = [i - tail_pos + self.max_length for i in range(length)]

        return sent, tags, pos1, pos2, length, label

    def pad_seq(self, batch_seq, pad_value):
        max_length = max(len(seq) for seq in batch_seq)

        padded = [seq + [pad_value] * (max_length - len(seq))
                  for seq in batch_seq]

        return padded

    def get_labels(self):
        labels = [item[5] for item in self._data]
        return labels

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def collate(self, batch):
        sent_batch, tag_batch, pos1_batch, pos2_batch, length_batch, label_batch, id_batch, size_batch = list(
            zip(*batch))

        sent_batch = list(chain(*sent_batch))
        tag_batch = list(chain(*tag_batch))
        pos1_batch = list(chain(*pos1_batch))
        pos2_batch = list(chain(*pos2_batch))
        length_batch = list(chain(*length_batch))

        sent_batch = self.pad_seq(sent_batch, self.pad_id)
        tag_batch = self.pad_seq(tag_batch, self.pad_id)
        pos1_batch = self.pad_seq(pos1_batch, self.pad_id)
        pos2_batch = self.pad_seq(pos2_batch, self.pad_id)

        sent_batch = torch.LongTensor(sent_batch)
        tag_batch = torch.LongTensor(tag_batch)
        pos1_batch = torch.LongTensor(pos1_batch)
        pos2_batch = torch.LongTensor(pos2_batch)
        length_batch = torch.LongTensor(length_batch)

        label_batch = torch.LongTensor(label_batch)

        scope_batch = np.cumsum(size_batch)

        return {'sent': sent_batch, 'tag': tag_batch,
                'pos1': pos1_batch, 'pos2': pos2_batch, 'length': length_batch,
                'label': label_batch, 'id': id_batch, 'scope': scope_batch}


class REDataset_INS(REDataset):
    def __init__(self, vocab, data_dir, data_name, max_length, sort=True):
        super(REDataset_INS, self).__init__(vocab, data_dir, data_name, max_length, sort)

    def process(self, vocab, data):
        for ins in tqdm(data):
            sent, tag, pos1, pos2, length, label = self.convert(vocab, ins)
            ins_id = ins['head']['id'] + '#' + ins['tail']['id']
            self._data.append([[sent], [tag], [pos1], [pos2], [length], label, ins_id, 1])


class REDataset_BAG(REDataset):
    def __init__(self, vocab, data_dir, data_name, max_length, sort=True):
        super(REDataset_BAG, self).__init__(vocab, data_dir, data_name, max_length, sort)

    def process(self, vocab, data):
        data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'])
        last_ins_id = 'None#None'
        for ins in tqdm(data):
            sent, tag, pos1, pos2, length, label = self.convert(vocab, ins)
            ins_id = ins['head']['id'] + '#' + ins['tail']['id']
            if ins_id != last_ins_id:
                self._data.append([[sent], [tag], [pos1], [pos2], [length], label, ins_id, 1])
            else:
                self._data[-1][0].append(sent)
                self._data[-1][1].append(tag)
                self._data[-1][2].append(pos1)
                self._data[-1][3].append(pos2)
                self._data[-1][4].append(length)
                self._data[-1][7] += 1

            last_ins_id = ins_id


class DataLoader_BAG(object):
    def __init__(self, dataset, batch_size, collate_fn, shuffle=False):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.batch_size = batch_size

        self.order = list(range(0, len(dataset)))

        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.order)
        i = 0
        while i < len(self.dataset):
            j = 0
            batch = []
            while j < self.batch_size and i < len(self.dataset):
                batch.append(self.dataset[self.order[i]])

                j += self.dataset[self.order[i]][7]
                i += 1

            yield self.collate_fn(batch)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class DataLoader_INS(object):
    def __init__(self, dataset, batch_size, collate_fn, shuffle=False):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.batch_size = batch_size

        self.order = list(range(0, len(dataset)))
        self.order = [self.order[i:i + batch_size] for i in range(0, len(dataset), batch_size)]

        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.order)

        for indices in self.order:
            batch = self.collate_fn([self.dataset[i] for i in indices])

            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
