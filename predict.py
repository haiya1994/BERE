from sklearn import metrics

from dataset import *
from network.model import *
from collections import OrderedDict
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

DEVICE = 'cuda:0'


def predict(config, model_name, data_name):
    if config.BAG_MODE:
        REModel = REModel_BAG
        DataLoader = DataLoader_BAG

    else:
        REModel = REModel_INS
        DataLoader = DataLoader_INS

    vocab = torch.load(os.path.join(config.ROOT_DIR, 'vocab.pt'))

    logging.info('Load pretrained vectors: {}*{}'.format(vocab.word_num, vocab.word_dim))
    logging.info('Number of classes: {}'.format(vocab.class_num))

    predict_dataset = torch.load(os.path.join(config.ROOT_DIR, data_name + '.pt'))
    predict_loader = DataLoader(predict_dataset, batch_size=config.BATCH_SIZE, collate_fn=predict_dataset.collate,
                                shuffle=False)

    logging.info('Number of predict pair: {}'.format(len(predict_dataset)))

    model = REModel(vocab=vocab, tag_dim=config.TAG_DIM,
                    max_length=config.MAX_LENGTH,
                    hidden_dim=config.HIDDEN_DIM, dropout_prob=config.DROP_PROB,
                    bidirectional=config.BIDIRECTIONAL)

    num_params = sum(np.prod(p.size()) for p in model.parameters())
    num_embedding_params = np.prod(model.word_emb.weight.size()) + np.prod(model.tag_emb.weight.size())
    print('# of parameters: {}'.format(num_params))
    print('# of word embedding parameters: {}'.format(num_embedding_params))
    print('# of parameters (excluding embeddings): {}'.format(num_params - num_embedding_params))

    model.load_state_dict(
        torch.load(os.path.join(config.SAVE_DIR, config.DATA_SET, model_name), map_location='cpu'))
    model.eval()
    model.to(DEVICE)
    model.display()

    torch.set_grad_enabled(False)

    logging.info('Using device {}'.format(DEVICE))

    predict_ids = []
    predict_labels = []
    predict_logits = []
    predict_preds = []

    predict_result = []

    def run_iter(batch):
        sent = batch['sent'].to(DEVICE)
        tag = batch['tag'].to(DEVICE)

        length = batch['length'].to(DEVICE)

        label = batch['label']
        id = batch['id']
        scope = batch['scope']

        logits = model(sent, tag, length, scope)
        logits = F.softmax(logits, dim=1)
        label_pred = logits.max(1)[1]

        return id, label, logits.detach().cpu(), label_pred.detach().cpu()

    for batch in tqdm(predict_loader):
        id, label, logits, label_pred = run_iter(batch)

        predict_ids.extend(id)
        predict_labels.extend(label)
        predict_logits.extend(logits)
        predict_preds.extend(label_pred)

    result = metrics.precision_recall_fscore_support(predict_labels, predict_preds, labels=[1], average='micro')

    for i in range(len(predict_dataset)):
        j = np.argmax(predict_logits[i])
        if j > 0:
            predict_result.append({'pair_id': predict_ids[i], 'score': float(predict_logits[i][j]),
                                   'relation': int(j)})

    logging.info(
        'precision =  {:.4f}: recall = {:.4f}, fscore = {:.4f}'.format(result[0], result[1], result[2]))

    predict_result.sort(key=lambda x: x['score'], reverse=True)
    if not os.path.isdir(config.RESULT_DIR):
        os.makedirs(config.RESULT_DIR)
    logging.info('Save result to {}'.format(config.RESULT_DIR))
    json.dump(predict_result, open(os.path.join(config.RESULT_DIR, config.DATA_SET + '_' + data_name + '.json'), 'w'))


def output(data_name):
    output_data = OrderedDict()

    predict_data = json.load(open(os.path.join(config.RESULT_DIR, config.DATA_SET + '_' + data_name + '.json'), 'r'))
    origin_data = json.load(open(os.path.join(config.ROOT_DIR, data_name + '.json'), 'r'))
    label2id = json.load(open(os.path.join(config.ROOT_DIR, 'label2id'+ '.json'), 'r'))
    id2label = {v: k for k, v in label2id.items()}

    for item in predict_data:
        pair_id = item['pair_id'].split('#')
        drug_id = pair_id[0]
        target_id = pair_id[1]
        rel = item['relation']
        score = item['score']
        output_data[(drug_id, target_id)] = {'drug_id': drug_id, 'target_id': target_id, 'relation': id2label[rel],
                                             'score': score, 'supporting_entry': []}

    for item in origin_data:
        drug_id = item['head']['id']
        target_id = item['tail']['id']

        if (drug_id, target_id) in output_data:
            try:
                pmid = item['pmid']
            except:
                pmid = None
            drug_name = item['head']['word']
            target_name = item['tail']['word']
            sentence = item['sentence']
            output_data[(drug_id, target_id)]['drugbank_relation'] = item['relation']
            output_data[(drug_id, target_id)]['supporting_entry'].append(
                {'pmid': pmid, 'sentence': sentence, 'drug': drug_name, 'target': target_name})

    if not os.path.isdir(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    logging.info('Save result to {}'.format(config.OUTPUT_DIR))
    result = list(output_data.values())
    json.dump(result,
              open(os.path.join(config.OUTPUT_DIR, config.DATA_SET + '_' + data_name + '.json'), 'w'))


if __name__ == '__main__':
    from data.dti import config

    predict(config, 'dti-0.5419.pkl', 'pmc_nintedanib')

    output(data_name='pmc_nintedanib')
