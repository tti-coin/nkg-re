import numpy as np
import json
import math
import torch
from torch import nn, Tensor

from transformers import AutoModel

from dgl.nn import GraphConv

__all__ = ["Model"]


def orthonormal_initializer(input_size, output_size):
    """from https://github.com/patverga/bran/blob/32378da8ac339393d9faa2ff2d50ccb3b379e9a2/src/tf_utils.py#L154"""
    I = np.eye(output_size)
    lr = 0.1
    eps = 0.05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI**2 / 2)
            Q2 = Q**2
            Q -= (
                lr
                * Q.dot(QTQmI)
                / (np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            )
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print("Orthogonal pretrainer loss: %.2e" % loss)
    else:
        print("Orthogonal pretrainer failed, using non-orthogonal random matrix")
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return Q.astype(np.float32)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerConv(nn.Module):
    def __init__(self, dim: int, vocab_size: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dim = dim
        self.token_embeddings = nn.Embedding(vocab_size, dim)
        self.token_embeddings.weight.data.uniform_(-0.1, 0.1)

        self.pos_encoder = PositionalEncoding(dim)
        self.transformer1 = torch.nn.TransformerEncoderLayer(
            d_model=self.dim, nhead=4, batch_first=True, dropout=dropout
        )
        self.conv11 = torch.nn.Conv1d(self.dim, self.dim, 1, padding="same")
        self.conv12 = torch.nn.Conv1d(self.dim, self.dim, 5, padding="same")
        self.conv13 = torch.nn.Conv1d(self.dim, self.dim, 1, padding="same")
        # torch.nn.MultiheadAttention(self.dim, 4, batch_first=True),
        self.transformer2 = torch.nn.TransformerEncoderLayer(
            d_model=self.dim, nhead=4, batch_first=True, dropout=dropout
        )
        self.conv21 = torch.nn.Conv1d(self.dim, self.dim, 1, padding="same")
        self.conv22 = torch.nn.Conv1d(self.dim, self.dim, 5, padding="same")
        self.conv23 = torch.nn.Conv1d(self.dim, self.dim, 1, padding="same")

        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        src = self.token_embeddings(input_ids) * math.sqrt(self.dim)
        # print("token shape", src.shape)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        h = self.transformer1(src, src_key_padding_mask=attention_mask)
        h = h.transpose(1, 2)
        h = self.conv11(h)
        h = self.conv12(h)
        h = self.conv13(h)
        h = h.transpose(1, 2)
        h = self.transformer2(h, src_key_padding_mask=attention_mask)
        h = h.transpose(1, 2)
        h = self.conv21(h)
        h = self.conv22(h)
        h = self.conv23(h)
        h = h.transpose(1, 2)
        return h


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.dim = config["dim"]
        self.graph_dim = config["graph_dim"]
        self.tokenizer_org_size = config["tokenizer_org_size"]
        with open(config["data_path"] + "/entid2kgid.json") as f:
            self.entid2kgid = json.load(f)
        self.embedding = nn.Embedding(len(self.entid2kgid), config["graph_dim"])
        self.feat = self.embedding(torch.tensor(list(range(len(self.entid2kgid)))))

        if self.config["gnn_model"] == "gcn":
            if self.config["num_gcn"] == 1:
                self.gcn1 = GraphConv(self.graph_dim, self.graph_dim, norm="both", weight=True, bias=True)
            elif self.config["num_gcn"] == 2:
                self.gcn1 = GraphConv(self.graph_dim, self.graph_dim, norm="both", weight=True, bias=True)
                self.gcn2 = GraphConv(self.graph_dim, self.graph_dim, norm="both", weight=True, bias=True)
            elif self.config["num_gcn"] == 3:
                self.gcn1 = GraphConv(self.graph_dim, self.graph_dim, norm="both", weight=True, bias=True)
                self.gcn2 = GraphConv(self.graph_dim, self.graph_dim, norm="both", weight=True, bias=True)
                self.gcn3 = GraphConv(self.graph_dim, self.graph_dim, norm="both", weight=True, bias=True)
        self.num_rel = len(json.loads(open(config["data_path"] + "/relation_map.json").read()))
        with open(config["data_path"] + "/rel2kgid.json") as f:
            rel2kgid = json.load(f)
        self.num_rel_kg = len(rel2kgid)
        self.init_node_norm_list = []
        if self.config["norm_gnn"] == True:
            self.layer_norm = nn.LayerNorm(self.graph_dim)
        if self.config["encoder_type"] == "transformer_conv":
            self.encoder = TransformerConv(self.dim, self.config["vocabsize"])
            self.D = self.dim
        elif self.config["encoder_type"] == "transformer":
            self.encoder = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True)
            self.encoder.init_weights()
            self.D = self.encoder.config.hidden_size
        else:
            self.encoder = AutoModel.from_pretrained(config["encoder_type"])
            self.D = self.encoder.config.hidden_size

        if self.config["model"] == "biaffine":
            self.head_layer0 = torch.nn.Linear(self.D, self.D)
            self.tail_layer0 = torch.nn.Linear(self.D, self.D)
            self.head_layer1 = torch.nn.Linear(self.D, self.dim)
            self.tail_layer1 = torch.nn.Linear(self.D, self.dim)
            if self.config["kind_of_token"] == "text_graph":
                self.head_kg_layer0 = torch.nn.Linear(self.D, self.D)
                self.tail_kg_layer0 = torch.nn.Linear(self.D, self.D)
                self.head_kg_layer1 = torch.nn.Linear(self.D, self.dim)
                self.tail_kg_layer1 = torch.nn.Linear(self.D, self.dim)
                mat = orthonormal_initializer(self.dim*2, self.dim*2)[:, None, :]

            else:
                mat = orthonormal_initializer(self.dim, self.dim)[:, None, :]
            self.relu = torch.nn.ReLU()
            biaffine_mat = np.concatenate([mat] * (self.num_rel + 1), axis=1)
            self.biaffine_mat = torch.nn.Parameter(torch.tensor(biaffine_mat), requires_grad=True)  # (dim, R, dim)
            self.multi_label = config["multi_label"]
            self.softmax = torch.nn.Softmax(dim=1)
            self.sigmoid = torch.nn.Sigmoid()

        elif self.config["model"] == "dot":
            if self.config["kind_of_token"] == "text_graph":
                self.layer1 = torch.nn.Linear(self.D * 4, self.dim)
            elif self.config["kind_of_token"] == "cls_text_graph":
                self.layer1 = torch.nn.Linear(self.D * 5, self.dim)
            elif self.config["kind_of_token"] == "text" or self.config["kind_of_token"] == "only_graph":
                self.layer1 = torch.nn.Linear(self.D * 2, self.dim)
            self.layer2 = torch.nn.Linear(self.dim, self.num_rel + 1)
            self.relu = torch.nn.ReLU()
            self.multi_label = config["multi_label"]
            self.softmax = torch.nn.Softmax(dim=1)
            self.sigmoid = torch.nn.Sigmoid()
            self.dropout = torch.nn.Dropout(p=config["dropout_rate"])

    def bi_affine(self, e1_vec, e2_vec):
        # e1_vec: batchsize, text_length, dim
        # e2_vec: batchsize, text_length, dim
        # output: batchsize, text_length, text_length, R
        batchsize, text_length, dim = e1_vec.shape

        # (batchsize * text_length, dim) (dim, R*dim) -> (batchsize * text_length, R*dim)
        lin = torch.matmul(
            torch.reshape(e1_vec, [-1, dim]),
            torch.reshape(self.biaffine_mat, [dim, (self.num_rel + 1) * dim]),
        )
        if self.config["kind_of_token"] == "text_graph":
            # (batchsize, text_length * R, D*2) (batchsize, D*2, text_length) -> (batchsize, text_length * R, text_length)
            bilin = torch.matmul(
                torch.reshape(lin, [batchsize, text_length * (self.num_rel + 1), self.dim*2]),
                torch.transpose(e2_vec, 1, 2),
            )
        else:
            # (batchsize, text_length * R, D) (batchsize, D, text_length) -> (batchsize, text_length * R, text_length)
            bilin = torch.matmul(
                torch.reshape(lin, [batchsize, text_length * (self.num_rel + 1), self.dim]),
                torch.transpose(e2_vec, 1, 2),
            )

        output = torch.reshape(bilin, [batchsize, text_length, self.num_rel + 1, text_length])
        output = torch.transpose(output, 2, 3)
        return output

    def forward(
        self,
        input_ids,
        attention_mask,
        attention_mask_graph,
        position_ids,
        ep_mask,
        e1_indicator,
        e2_indicator,
        e1_indicator_kg,
        e2_indicator_kg,
        graphs,
        e1s,
        e2s,
        input_kgid,
        num_kgid,
    ):
        # input_ids: (batchsize, text_length)
        # attention_mask: (batchsize, text_length)
        # ep_mask: (batchsize, num_ep, text_length, text_length)
        # e1_indicator: not used
        # e2_indicator: not used
        # e1s, e2s: entity_id_list
        assert e1_indicator.shape == e1_indicator_kg.shape == e2_indicator.shape == e2_indicator_kg.shape
        for graph in graphs:
            if self.config["gnn_model"] == "gat":
                hidden = torch.max(self.gat1(graph, self.feat), dim=1).values
                output = torch.max(self.gat2(graph, self.feat), dim=1).values
            elif self.config["gnn_model"] == "gcn":
                if self.config["num_gcn"] == 1:
                    output = self.gcn1(graph, self.feat)
                elif self.config["num_gcn"] == 2:
                    hidden = self.gcn1(graph, self.feat)
                    output = self.gcn2(graph, hidden)
                elif self.config["num_gcn"] == 3:
                    hidden1 = self.gcn1(graph, self.feat)
                    hidden2 = self.gcn2(graph, hidden1)
                    output = self.gcn3(graph, hidden2)
            if self.config["norm_gnn"] == True:
                output = self.layer_norm(output)
            self.encoder.embeddings.word_embeddings.weight.data[self.tokenizer_org_size :] = output
        if self.config["encoder_type"] == "transformer_conv":
            h = self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long())
        elif self.config["encoder_type"] == "transformer":
            h = self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long())[2][2]  # Two hidden layer
        elif self.config["encode_type"] == "only_graph":
            h = self.encoder(
                input_ids=input_ids.long(),
                attention_mask=attention_mask_graph.long(),
                position_ids=position_ids.long(),
            )[0].unsqueeze(
                1
            )  # (batchsize, text_length, D)
        else:
            h = self.encoder(
                input_ids=input_ids.long(),
                attention_mask=attention_mask.long(),
                position_ids=position_ids.long(),
            )[0].unsqueeze(
                1
            )  # (batchsize, text_length, D)
        if self.config["model"] == "biaffine":
            h = h.squeeze(1)
            e1_vec_ = self.relu(self.head_layer0(h))
            e2_vec_ = self.relu(self.tail_layer0(h))
            e1_vec = self.head_layer1(e1_vec_)  # (batchsize, text_length, dim)
            e2_vec = self.tail_layer1(e2_vec_)  # (batchsize, text_length, dim)
            if self.config["kind_of_token"] == "text_graph":
                e1_indicator_kg = e1_indicator_kg.max(1)[0]  # (batchsize, D) 
                e2_indicator_kg = e2_indicator_kg.max(1)[0]  # (batchsize, D) 
                e1_indicator_kg_mask = 1000 * torch.ones_like(e1_indicator_kg) * (1 - e1_indicator_kg)  # (batchsize, D)
                e2_indicator_kg_mask = 1000 * torch.ones_like(e2_indicator_kg) * (1 - e2_indicator_kg)  # (batchsize, D)
                e1_vec_kg = h * e1_indicator_kg.unsqueeze(2) - e1_indicator_kg_mask.unsqueeze(2)  # (batchsize, text_length, D)
                e2_vec_kg = h * e2_indicator_kg.unsqueeze(2) - e2_indicator_kg_mask.unsqueeze(2)  # (batchsize, text_length, D)
                if self.config["norm_gnn"] == True:
                    e1_vec_kg = self.layer_norm(e1_vec_kg)
                    e2_vec_kg = self.layer_norm(e2_vec_kg)
                    e1_vec_ = self.layer_norm(e1_vec_)
                    e2_vec_ = self.layer_norm(e2_vec_)
                e1_vec_kg_ = self.relu(self.head_kg_layer0(e1_vec_kg)) # (batchsize, text_length, D)
                e2_vec_kg_ = self.relu(self.head_kg_layer0(e2_vec_kg)) # (batchsize, text_length, D)
                e1_vec_kg = self.head_kg_layer1(e1_vec_kg_)  # (batchsize, text_length, dim)
                e2_vec_kg = self.tail_kg_layer1(e2_vec_kg_)  # (batchsize, text_length, dim)
                if self.config["aggregation"] == "concat":
                    e1_vec = torch.cat([e1_vec, e1_vec_kg], 2)  # (batchsize, text_length, dim*2)
                    e2_vec = torch.cat([e2_vec, e2_vec_kg], 2)  # (batchsize, text_length, dim*2)

            pairwise_scores = self.bi_affine(e1_vec, e2_vec).unsqueeze(1)

            ep_mask = ep_mask.unsqueeze(4)
            # batchsize, num_ep, text_length, text_length, R + 1
            pairwise_scores = pairwise_scores + ep_mask
            pairwise_scores = torch.logsumexp(pairwise_scores, dim=[2, 3])  # batchsize, num_ep, R + 1

        elif self.config["model"] == "dot":
            # (batchsize, num_ep, text_length, 1)
            e1_indicator = e1_indicator.unsqueeze(3)
            e2_indicator = e2_indicator.unsqueeze(3)
            e1_indicator_mask = 1000 * torch.ones_like(e1_indicator) * (1 - e1_indicator)
            e2_indicator_mask = 1000 * torch.ones_like(e2_indicator) * (1 - e2_indicator)

            # (batchsize, num_ep, text_length, D)
            e1_vec = h * e1_indicator - e1_indicator_mask
            # (batchsize, num_ep, text_length, D)
            e2_vec = h * e2_indicator - e2_indicator_mask
            e1_vec = e1_vec.max(2)[0]  # (batchsize, num_ep, D)
            e2_vec = e2_vec.max(2)[0]  # (batchsize, num_ep, D)

            if self.config["kind_of_token"] == "text_graph":  # (head, tail, head of graph, tail of graph)
                e1_indicator_kg = e1_indicator_kg.unsqueeze(3)
                e2_indicator_kg = e2_indicator_kg.unsqueeze(3)
                e1_indicator_kg_mask = 1000 * torch.ones_like(e1_indicator_kg) * (1 - e1_indicator_kg)
                e2_indicator_kg_mask = 1000 * torch.ones_like(e2_indicator_kg) * (1 - e2_indicator_kg)
                e1_vec_kg = h * e1_indicator_kg - e1_indicator_kg_mask  # (batchsize, num_ep, text_length, D)
                e2_vec_kg = h * e2_indicator_kg - e2_indicator_kg_mask  # (batchsize, num_ep, text_length, D)
                e1_vec_kg = e1_vec_kg.max(2)[0]  # (batchsize, num_ep, D)
                e2_vec_kg = e2_vec_kg.max(2)[0]  # (batchsize, num_ep, D)
                e1e2_vec = self.dropout(
                    self.relu(self.layer1(torch.cat([e1_vec, e1_vec_kg, e2_vec, e2_vec_kg], 2)))
                )  # (batchsize, num_ep, 4D)

            elif self.config["kind_of_token"] == "cls_text_graph":
                e1_indicator_kg = e1_indicator_kg.unsqueeze(3)
                e2_indicator_kg = e2_indicator_kg.unsqueeze(3)
                e1_indicator_kg_mask = 1000 * torch.ones_like(e1_indicator_kg) * (1 - e1_indicator_kg)
                e2_indicator_kg_mask = 1000 * torch.ones_like(e2_indicator_kg) * (1 - e2_indicator_kg)
                e1_vec_kg = h * e1_indicator_kg - e1_indicator_kg_mask  # (batchsize, num_ep, text_length, D)
                e2_vec_kg = h * e2_indicator_kg - e2_indicator_kg_mask  # (batchsize, num_ep, text_length, D)
                e1_vec_kg = e1_vec_kg.max(2)[0]  # (batchsize, num_ep, D)
                e2_vec_kg = e2_vec_kg.max(2)[0]  # (batchsize, num_ep, D)
                cls = h[:, :, 0, :].repeat(1, e1_vec.size(1), 1)  # (batchsize, num_ep, D)
                e1e2_vec = self.dropout(
                    self.relu(self.layer1(torch.cat([cls, e1_vec, e1_vec_kg, e2_vec, e2_vec_kg], 2)))
                )  # (batchsize, num_ep, 4D)
            elif self.config["kind_of_token"] == "only_graph":
                e1_indicator_kg = e1_indicator_kg.unsqueeze(3)
                e2_indicator_kg = e2_indicator_kg.unsqueeze(3)
                e1_indicator_kg_mask = 1000 * torch.ones_like(e1_indicator_kg) * (1 - e1_indicator_kg)
                e2_indicator_kg_mask = 1000 * torch.ones_like(e2_indicator_kg) * (1 - e2_indicator_kg)
                e1_vec_kg = h * e1_indicator_kg - e1_indicator_kg_mask  # (batchsize, num_ep, text_length, D)
                e2_vec_kg = h * e2_indicator_kg - e2_indicator_kg_mask  # (batchsize, num_ep, text_length, D)
                e1_vec_kg = e1_vec_kg.max(2)[0]  # (batchsize, num_ep, D)
                e2_vec_kg = e2_vec_kg.max(2)[0]  # (batchsize, num_ep, D)
                e1e2_vec = self.dropout(
                    self.relu(self.layer1(torch.cat([e1_vec_kg, e2_vec_kg], 2)))
                )  # (batchsize, num_ep, 5D)

            elif self.config["kind_of_token"] == "text":
                e1e2_vec = self.dropout(
                    self.relu(self.layer1(torch.cat([e1_vec, e2_vec], 2)))
                )  # (batchsize, num_ep, 2D)
            pairwise_scores = self.layer2(e1e2_vec)  # (batchsize, num_ep, R+1)

        if self.multi_label == True:
            # (batchsize, num_ep, R)
            pairwise_scores = pairwise_scores[:, :, :-1]

        return pairwise_scores
