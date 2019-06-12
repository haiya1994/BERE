import numpy
from sklearn import metrics

from dataset import *
from network.model import *

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

DEVICE = 'cuda:2'


def test(config, model_name):
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
    test_loader = DataLoader(test_dataset, config.BATCH_SIZE, collate_fn=test_dataset.collate, shuffle=False)

    test_labels = numpy.array(test_dataset.get_labels())
    test_rel_num = sum(test_labels != vocab.NA_id)

    logging.info('Number of test pair: {}'.format(len(test_dataset)))

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

    def run_iter(batch):
        sent = batch['sent'].to(DEVICE)
        tag = batch['tag'].to(DEVICE)

        length = batch['length'].to(DEVICE)
        scope = batch['scope']

        logit = model(sent, tag, length, scope)

        return logit.cpu()

    test_result = []

    for test_batch in test_loader:
        test_logit = run_iter(batch=test_batch)

        for idx in range(len(test_logit)):
            for rel in range(1, vocab.class_num):
                test_result.append(
                    {'score': test_logit[idx][rel], 'flag': test_batch['label'][idx] == rel})

    sorted_test_result = sorted(test_result, key=lambda x: x['score'])

    prec = []
    recall = []
    correct = 0
    for i, item in enumerate(sorted_test_result[::-1]):
        correct += int(item['flag'])
        prec.append(float(correct) / (i + 1))
        recall.append(float(correct) / test_rel_num)

    x, y = np.array(recall), np.array(prec)

    auc = metrics.auc(x=x, y=y)

    logging.info('auc =  {:.4f}'.format(auc))

    result_dir = os.path.join(config.RESULT_DIR, config.DATA_SET)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    np.save(os.path.join(result_dir, "BERE_x.npy"), x)
    np.save(os.path.join(result_dir, "BERE_y.npy"), y)


if __name__ == '__main__':
    from data.dti import config

    test(config, 'dti-0.5419.pkl')
