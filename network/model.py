from network.encoder import *
from network.selector import *


class REModel_INS(nn.Module):
    """
    The relation extraction model with INS mode, which will classify each
    sentence instance into an individual class.

    Args:
        vocab (object): The vocab object which contains the
            basic statics of the corpus. See dataset.py for
            more details.
        tag_dim (int): The dimension of POS (part-of-speech)
            embedding.
        max_length (int): All the sentences will be cropped
            or padded to the max_length.
        hidden_dim (int): The dimension of hidden unit in
            GRU.
        dropout_prob (float): Probability of an element
            to be zeroed.
        bidirectional (bool): If true, bi-directional GRU
            will be used.

    Inputs: sent, tag, length, verbose_output
        - **sent** of shape `(batch, seq_len, word_dim)`.
        - **tag** of shape `(batch, seq_len, tag_dim)`.
        - **length** of shape `(batch)`.

    Outputs: logit
        - **logit** of shape `(batch, class_num)`.
    """

    def __init__(self, vocab, tag_dim, max_length, hidden_dim,
                 dropout_prob,
                 bidirectional=True):
        super(REModel_INS, self).__init__()

        self.vocab = vocab

        class_num = vocab.class_num
        word_num = vocab.word_num

        word_dim = vocab.word_dim
        tag_num = vocab.tag_num

        self.ent1_id = vocab.ent1_id
        self.ent2_id = vocab.ent2_id

        self.max_length = max_length
        self.word_emb = nn.Embedding(word_num, word_dim, padding_idx=0)
        self.tag_emb = nn.Embedding(tag_num, tag_dim, padding_idx=0)

        self.word_emb.weight.data.set_(vocab.vectors)

        in_dim = word_dim + tag_dim

        self.attn = MultiAttn(in_dim)
        self.leaf_rnn = PackedGRU(in_dim, hidden_dim, bidirectional=bidirectional)

        if bidirectional:
            hidden_dim = 2 * hidden_dim

        self.encoder = GumbelTreeGRU(hidden_dim)
        self.selector = BagAttention(3 * hidden_dim)

        feat_dim = 3 * hidden_dim

        self.classifier = nn.Sequential(nn.Linear(feat_dim, feat_dim // 10), nn.ReLU(),
                                        nn.Linear(feat_dim // 10, class_num))

        self.dropout = nn.Dropout(dropout_prob)

    def display(self):
        print(self)

    def forward(self, sent, tag, length, verbose_output=False):
        ent1_mask = torch.eq(sent, self.ent1_id).unsqueeze(-1).float()
        ent2_mask = torch.eq(sent, self.ent2_id).unsqueeze(-1).float()

        word_embedding = self.dropout(self.word_emb(sent))
        tag_embedding = self.dropout(self.tag_emb(tag))

        embedding = torch.cat([word_embedding, tag_embedding], dim=-1)

        # -- Prepare masks
        attn_mask = utils.padding_mask(sent)
        non_pad_mask = utils.non_padding_mask(sent)

        embedding, word_attn = self.attn(embedding, attn_mask, non_pad_mask)
        embedding = self.leaf_rnn(embedding, length)

        tree_feat, tree_order = self.encoder(embedding, length)

        ent1_feat = (embedding * ent1_mask).sum(1)  # (B,D)
        ent2_feat = (embedding * ent2_mask).sum(1)  # (B,D)

        feat = torch.cat([tree_feat, ent1_feat, ent2_feat], -1)

        feat = self.dropout(feat)
        logit = self.classifier(feat)

        if verbose_output:
            return logit, word_attn, tree_order

        else:
            return logit


class REModel_BAG(nn.Module):
    """
    The relation extraction model with BAG mode, which will classify each
    sentence bag into an individual class.

    Args:
        vocab (object): The vocab object which contains the
            basic statics of the corpus. See dataset.py for
            more details.
        tag_dim (int): The dimension of POS (part-of-speech)
            embedding.
        max_length (int): All the sentences will be cropped
            or padded to the max_length.
        hidden_dim (int): The dimension of hidden unit in
            GRU.
        dropout_prob (float): Probability of an element
            to be zeroed.
        bidirectional (bool): If true, bi-directional GRU
            will be used.

    Inputs: sent, tag, length, verbose_output
        - **sent** of shape `(batch, seq_len, word_dim)`.
        - **tag** of shape `(batch, seq_len, tag_dim)`.
        - **length** of shape `(batch)`.

    Outputs: logit
        - **logit** of shape `(batch, class_num)`.
    """

    def __init__(self, vocab, tag_dim, max_length, hidden_dim,
                 dropout_prob,
                 bidirectional=True):
        super(REModel_BAG, self).__init__()

        self.vocab = vocab

        class_num = vocab.class_num
        word_num = vocab.word_num

        word_dim = vocab.word_dim
        tag_num = vocab.tag_num

        self.ent1_id = vocab.ent1_id
        self.ent2_id = vocab.ent2_id

        self.max_length = max_length
        self.word_emb = nn.Embedding(word_num, word_dim, padding_idx=0)
        self.tag_emb = nn.Embedding(tag_num, tag_dim, padding_idx=0)

        self.word_emb.weight.data.set_(vocab.vectors)

        in_dim = word_dim + tag_dim

        self.attn = MultiAttn(in_dim)
        self.leaf_rnn = LeafRNN(in_dim, hidden_dim, bidirectional=bidirectional)

        if bidirectional:
            hidden_dim = 2 * hidden_dim

        self.encoder = GumbelTreeGRU(hidden_dim)

        self.selector = BagAttention(3 * hidden_dim)

        feat_dim = 3 * hidden_dim

        self.classifier = nn.Sequential(nn.Linear(feat_dim, feat_dim // 10), nn.ReLU(),
                                        nn.Linear(feat_dim // 10, class_num))

        self.dropout = nn.Dropout(dropout_prob)

    def display(self):
        print(self)

    def forward(self, sent, tag, length, scope, verbose_output=False):
        ent1_mask = torch.eq(sent, self.ent1_id).unsqueeze(-1).float()
        ent2_mask = torch.eq(sent, self.ent2_id).unsqueeze(-1).float()

        word_embedding = self.dropout(self.word_emb(sent))

        tag_embedding = self.dropout(self.tag_emb(tag))

        embedding = torch.cat([word_embedding, tag_embedding], dim=-1)

        # -- Prepare masks
        attn_mask = utils.padding_mask(sent)
        non_pad_mask = utils.non_padding_mask(sent)

        embedding, word_attn = self.attn(embedding, attn_mask, non_pad_mask)
        embedding = self.leaf_rnn(embedding, non_pad_mask, length)

        tree_feat, tree_order = self.encoder(embedding, length)

        ent1_feat = (embedding * ent1_mask).sum(1)  # (B,D)
        ent2_feat = (embedding * ent2_mask).sum(1)  # (B,D)

        feat = torch.cat([tree_feat, ent1_feat, ent2_feat], -1)

        feat, sent_attn = self.selector(feat, scope)

        feat = self.dropout(feat)
        logit = self.classifier(feat)

        if verbose_output:
            return logit, word_attn, tree_order, sent_attn

        else:
            return logit
