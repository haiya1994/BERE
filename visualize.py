from collections import OrderedDict

import matplotlib.pyplot as plt

from dataset import *
from network.model import *

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def top_k(config, model_name, top_k=10):
    if config.BAG_MODE:
        REModel = REModel_BAG
        DataLoader = DataLoader_BAG

    else:
        REModel = REModel_INS
        DataLoader = DataLoader_INS

    vocab = torch.load(os.path.join(config.ROOT_DIR, 'vocab.pt'))

    logging.info('Load pretrained vectors: {}*{}'.format(vocab.word_num, vocab.word_dim))
    logging.info('Number of classes: {}'.format(vocab.class_num))

    test_dataset = torch.load(os.path.join(config.ROOT_DIR, 'test.pt'))
    valid_dataset = torch.load(os.path.join(config.ROOT_DIR, 'valid.pt'))
    train_dataset = torch.load(os.path.join(config.ROOT_DIR, 'train.pt'))

    all_dataset = test_dataset + valid_dataset + train_dataset

    all_loader = DataLoader(all_dataset, config.BATCH_SIZE, collate_fn=test_dataset.collate, shuffle=False)

    logging.info('Number of total pair: {}'.format(len(all_dataset)))

    model = REModel(vocab=vocab, tag_dim=config.TAG_DIM,
                    max_length=config.MAX_LENGTH,
                    hidden_dim=config.HIDDEN_DIM, dropout_prob=config.DROP_PROB,
                    bidirectional=config.BIDIRECTIONAL)

    model.load_state_dict(
        torch.load(os.path.join(config.SAVE_DIR, config.DATA_SET, model_name), map_location='cpu'))
    model.eval()
    model.to(DEVICE)
    model.display()

    torch.set_grad_enabled(False)

    def run_iter(batch):
        sent = batch['sent'].to(DEVICE)
        tag = batch['tag'].to(DEVICE)

        length = batch['length'].to(DEVICE)
        scope = batch['scope']

        label = batch['label']
        id = batch['id']

        logit = model(sent, tag, length, scope)

        return id, label, logit.cpu()

    def collect_data(config, data_map):
        all_data = json.load(open(os.path.join(config.ROOT_DIR, 'train.json'), 'r')) + json.load(
            open(os.path.join(config.ROOT_DIR, 'valid.json'), 'r')) + json.load(
            open(os.path.join(config.ROOT_DIR, 'test.json'), 'r'))

        data = []
        for item in all_data:
            drug_id = item['head']['id']
            target_id = item['tail']['id']

            if (drug_id, target_id) in data_map:
                item['score'] = data_map[(drug_id, target_id)]
                data.append(item)

        data = sorted(data, key=lambda x: (x['score'], x['head']['id'], x['tail']['id']), reverse=True)
        return data

    result = []

    for test_batch in all_loader:
        all_id, all_label, all_logit = run_iter(batch=test_batch)

        for idx in range(len(all_logit)):
            for rel in range(1, 2):
                if all_label[idx] == rel:
                    result.append(
                        {'id': all_id[idx], 'label': rel, 'score': float(all_logit[idx][rel])})

    top_result = sorted(result, key=lambda x: x['score'], reverse=True)[:top_k]

    top_map = OrderedDict()
    for item in top_result:
        pair_id = item['id'].split('#')
        drug_id = pair_id[0]
        target_id = pair_id[1]
        score = item['score']
        top_map[(drug_id, target_id)] = score

    data = collect_data(config, top_map)

    if not os.path.isdir(config.RESULT_DIR):
        os.makedirs(config.RESULT_DIR)

    json.dump(data,
              open(os.path.join(config.RESULT_DIR, '{}_top_{}_inhibitor.json'.format(config.DATA_SET, top_k)), 'w'))


def visualize(config, model_name, case_name):
    if config.BAG_MODE:
        REModel = REModel_BAG


    else:
        REModel = REModel_INS

    vocab = torch.load(os.path.join(config.ROOT_DIR, 'vocab.pt'))

    logging.info('Load pretrained vectors: {}*{}'.format(vocab.word_num, vocab.word_dim))
    logging.info('Number of classes: {}'.format(vocab.class_num))

    case_data = json.load(open(os.path.join(config.ROOT_DIR, case_name), 'r'))
    case_dataset = REDataset_INS(vocab, data_dir=config.ROOT_DIR, data_name=case_name, max_length=config.MAX_LENGTH,
                                 sort=False)
    case_loader = DataLoader_INS(case_dataset, batch_size=1, collate_fn=case_dataset.collate, shuffle=False)

    logging.info('Number of total pair: {}'.format(len(case_dataset)))

    model = REModel(vocab=vocab, tag_dim=config.TAG_DIM,
                    max_length=config.MAX_LENGTH,
                    hidden_dim=config.HIDDEN_DIM, dropout_prob=config.DROP_PROB,
                    bidirectional=config.BIDIRECTIONAL)

    model.load_state_dict(
        torch.load(os.path.join(config.SAVE_DIR, config.DATA_SET, model_name), map_location='cpu'))
    print(model.attn.gamma)

    model.eval()
    model.to(DEVICE)
    model.display()

    torch.set_grad_enabled(False)

    def run_iter(batch):
        sent = batch['sent'].to(DEVICE)
        tag = batch['tag'].to(DEVICE)

        length = batch['length'].to(DEVICE)
        scope = batch['scope']

        logit, word_attn, tree_order, sent_attn = model(sent, tag, length, scope, verbose_output=True)

        return logit.cpu(), word_attn.cpu(), tree_order, sent_attn

    def plot_attn(word_attn, sent):
        plt.matshow(word_attn)
        plt.colorbar()
        x, y = word_attn.shape
        x = np.array(range(x))
        y = np.array(range(y))

        plt.xticks(x, sent, rotation=90, fontsize=12)
        plt.yticks(y, sent, fontsize=12)

        plt.tight_layout()
        plt.show()

        # plot_path = os.path.join(config.RESULT_DIR, "word_attn.pdf")
        # plt.savefig(plot_path)
        # print('Attention map plot saved at: {}'.format(plot_path))

    def make_new_sent(sent, name1, name2, pos1, pos2):
        assert pos1 <= pos2
        new_sent = '{0} {1} {2} {3} {4}'.format(sent[:pos1[0]], name1, sent[pos1[1]:pos2[0]], name2,
                                                sent[pos2[1]:])
        return new_sent

    def get_sentence(item):
        sent = item['sentence']
        head_word = item['head']['word']
        tail_word = item['tail']['word']

        head_pos = sent.index(head_word)
        head_pos = [head_pos, head_pos + len(head_word)]
        tail_pos = sent.index(tail_word)
        tail_pos = [tail_pos, tail_pos + len(tail_word)]

        if head_pos <= tail_pos:
            sent = make_new_sent(sent, '<ent1>', '<ent2>', head_pos, tail_pos)
        else:
            sent = make_new_sent(sent, '<ent2>', '<ent1>', tail_pos, head_pos)

        sent = re.sub(r"\s+", r" ", sent).strip().split()

        head_idx = sent.index('<ent1>')
        tail_idx = sent.index('<ent2>')
        sent[head_idx] = head_word
        sent[tail_idx] = tail_word
        return sent

    def get_parse_tree(sent, tree_order):
        comp_order = []
        for order in tree_order:
            order = order.cpu()

            index = torch.nonzero(order)[0][1]
            comp_order.append(index)
        comp_word = []
        for order in comp_order:
            order = int(order)
            comp_word.append((sent[order], sent[order + 1]))
            sent[order] = sent[order] + ' ' + sent[order + 1]
            sent.pop(order + 1)
        comp_word.append((sent[0], sent[1]))
        sent[0] = sent[0] + ' ' + sent[1]
        sent.pop(1)
        return comp_word

    index = 0
    for case in case_loader:
        item = case_data[index]
        print(item)
        sent = get_sentence(item)

        case_logit, case_word_attn, case_tree_order, case_sent_attn = run_iter(batch=case)
        case_word_attn = np.mean(np.array(case_word_attn), axis=0)

        print(case_logit)
        plot_attn(case_word_attn, sent)

        parse_tree = get_parse_tree(sent, case_tree_order)
        print(sent)
        print(parse_tree)
        print(case_sent_attn)

        index += 1


if __name__ == '__main__':
    from data.dti import config

    DEVICE = 'cuda:0'

    visualize(config, 'dti-0.5419.pkl', case_name='tree_example.json')
