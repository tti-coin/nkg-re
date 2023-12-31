import json
import random
import sys
import numpy as np
import torch
import unidecode
from tqdm import tqdm
import dgl

__all__ = [
    "Dataloader",
]

ENTITY_PAIR_TYPE_SET = set([("Chemical", "Disease"), ("Chemical", "Gene"), ("Gene", "Disease")])


def map_index(chars, tokens):
    # position index mapping from character level offset to token level offset
    ind_map = {}
    i, k = 0, 0  # (character i to token k)
    len_char = len(chars)
    num_token = len(tokens)
    while k < num_token:
        if i < len_char and chars[i].strip() == "":
            ind_map[i] = k
            i += 1
            continue
        token = tokens[k]
        if token[:2] == "##":
            token = token[2:]
        if token[:1] == "Ġ":
            token = token[1:]

        # assume that unk is always one character in the input text.
        if token != chars[i : (i + len(token))]:
            ind_map[i] = k
            i += 1
            k += 1
        else:
            for _ in range(len(token)):
                ind_map[i] = k
                i += 1
            k += 1

    return ind_map


def get_pairs(ent_id, csr, upper_limit_index, num_hop):
    pairs = np.empty([0, 2], dtype=np.int32)
    hop1_nodes = csr.indices[csr.indptr[ent_id] : csr.indptr[ent_id + 1]].astype(np.int32)
    if len(hop1_nodes) > upper_limit_index:
        hop1_nodes = np.random.choice(hop1_nodes, upper_limit_index, replace=False)
    if num_hop == 1:
        pairs_hop01 = np.column_stack((hop1_nodes, np.full_like(hop1_nodes, ent_id)))
        pairs = np.concatenate([pairs, pairs_hop01])
    elif num_hop == 2:
        pairs_hop01 = np.column_stack((hop1_nodes, np.full_like(hop1_nodes, ent_id)))
        pairs = np.concatenate([pairs, pairs_hop01])
        for hop1 in hop1_nodes:
            hop2_nodes = csr.indices[csr.indptr[hop1] : csr.indptr[hop1 + 1]].astype(np.int32)
            if len(hop2_nodes) > upper_limit_index:
                hop2_nodes = np.random.choice(hop2_nodes, upper_limit_index, replace=False)
            pairs_hop12 = np.column_stack((hop2_nodes, np.full_like(hop2_nodes, hop1)))
            pairs = np.concatenate([pairs, pairs_hop12])
    elif num_hop == 3:
        pairs_hop01 = np.column_stack((hop1_nodes, np.full_like(hop1_nodes, ent_id)))
        pairs = np.concatenate([pairs, pairs_hop01])
        for hop1 in hop1_nodes:
            hop2_nodes = csr.indices[csr.indptr[hop1] : csr.indptr[hop1 + 1]].astype(np.int32)
            if len(hop2_nodes) > upper_limit_index:
                hop2_nodes = np.random.choice(hop2_nodes, upper_limit_index, replace=False)
            pairs_hop12 = np.column_stack((hop2_nodes, np.full_like(hop2_nodes, hop1)))
            pairs = np.concatenate([pairs, pairs_hop12])
            for hop2 in hop2_nodes:
                hop3_nodes = csr.indices[csr.indptr[hop2] : csr.indptr[hop2 + 1]].astype(np.int32)
                if len(hop3_nodes) > upper_limit_index:
                    hop3_nodes = np.random.choice(hop3_nodes, upper_limit_index, replace=False)
                pairs_hop23 = np.column_stack((hop3_nodes, np.full_like(hop3_nodes, hop2)))
                pairs = np.concatenate([pairs, pairs_hop23])

    pairs = np.concatenate([pairs, pairs[:, ::-1]])
    pairs = set(map(tuple, pairs.tolist()))
    return pairs


def get_subgraph(e1s, e2s, entid2kgid, csr, upper_limit_index, num_hop):
    pairs = []
    for e1 in list(set(e1s)):
        if (e1 == "") or (e1 not in entid2kgid):
            continue
        for e2 in list(set(e2s)):
            if (e2 == "") or (e2 not in entid2kgid):
                continue
            e1_id = entid2kgid[e1]
            e2_id = entid2kgid[e2]
            pairs1 = get_pairs(ent_id=e1_id, csr=csr, upper_limit_index=upper_limit_index, num_hop=num_hop)
            pairs2 = get_pairs(ent_id=e2_id, csr=csr, upper_limit_index=upper_limit_index, num_hop=num_hop)
            pairs_ = list(pairs1 | pairs2)
            if ((e1_id, e2_id) in pairs_) or ((e2_id, e1_id) in pairs_):
                pairs_.remove((e1_id, e2_id))
                pairs_.remove((e2_id, e1_id))
            pairs.append(pairs_)
    pairs_list = list(set(sum(pairs, [])))
    if pairs_list == []:
        src = []
        dst = []
    else:
        src, dst = tuple(zip(*pairs_list))
    subgraph = (src, dst)
    return subgraph


def preprocess(
    data_entry,
    tokenizer,
    max_text_length,
    relation_map,
    entid2kgid,
    sym_csr,
    # db_cur,
    upper_limit_index,
    num_hop,
    baseline,
    lower=True,
):
    """convert to index array, cut long sentences, cut long document, pad short sentences, pad short document"""
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    padid = tokenizer.pad_token_id
    unk_token = tokenizer.unk_token
    cls_token_length = len(cls_token)

    docid = data_entry["docid"]

    if "title" in data_entry and "abstract" in data_entry:
        text = data_entry["title"] + data_entry["abstract"]
        if lower == True:
            text = text.lower()
    else:
        text = data_entry["text"]
        if lower == True:
            text = text.lower()
    entities_info = data_entry["entity"]
    relations_info = data_entry["relation"]
    rel_vocab_size = len(relation_map)

    # tokenizer will automatically add cls and sep at the beginning and end of each sentence
    # [CLS] --> 101, [PAD] --> 0, [SEP] --> 102, [UNK] --> 100

    text = unidecode.unidecode(text)
    tokens = tokenizer.tokenize(text)[: (max_text_length - 2)]
    tokens = [cls_token] + tokens + [sep_token]
    token_wids = tokenizer.convert_tokens_to_ids(tokens)  # tokenized ids
    text = cls_token + " " + text + " " + sep_token

    input_array = np.ones(max_text_length, dtype=np.int) * int(padid)
    input_length = len(token_wids)
    input_array[0 : len(token_wids)] = token_wids  # padded_input
    # pad_array = np.array(input_array != padid, dtype=np.long)  # pad

    ind_map = map_index(text, tokens) 
    num_added_kgid = len(entities_info) + 1  

    sys.stdout.flush()
    entity_indicator = {}
    entity_indicator_kg = {}
    entity_type = {}
    entity_id_set = set([])
    entity_pos = []
    entid2pos = {}
    input_kgid = [tokenizer.convert_tokens_to_ids(sep_token)]
    for i, entity in enumerate(entities_info):
        # if entity mention is outside max_text_length, ignore. +6 indicates additional offset due to "[CLS] "
        entity_id_set.add(entity["id"])
        entity_type[entity["id"]] = entity["type"]
        if entity["id"] not in entity_indicator:
            entity_indicator[entity["id"]] = np.zeros(max_text_length)
            entity_indicator_kg[entity["id"]] = np.zeros(max_text_length)

        if entity["start"] + cls_token_length in ind_map:
            startid = ind_map[entity["start"] + cls_token_length]
        else:
            startid = 0
        if (max_text_length - num_added_kgid) <= startid:
            num_added_kgid -= 1
        if entity["end"] + cls_token_length in ind_map:
            endid = ind_map[entity["end"] + cls_token_length]
            endid += 1
        else:
            endid = 0

        if startid >= endid:
            endid = startid + 1

        entity_indicator[entity["id"]][startid:endid] = 1
        if tokenizer.convert_tokens_to_ids(entity["id"]) not in entid2pos:
            entid2pos[tokenizer.convert_tokens_to_ids(entity["id"])] = [i]
        else:
            entid2pos[tokenizer.convert_tokens_to_ids(entity["id"])].append(i)
        # start_pos
        entity_pos.append(startid)
        input_kgid.append(tokenizer.convert_tokens_to_ids(entity["id"]))
    if baseline == True:
        num_added_kgid = 0
    input_kgid = input_kgid[:num_added_kgid]
    for kgid in input_kgid:
        if kgid == tokenizer.convert_tokens_to_ids(sep_token) or kgid == tokenizer.convert_tokens_to_ids(unk_token):
            continue
        for pos in entid2pos[kgid]:
            if pos < num_added_kgid - 1:
                entity_indicator_kg[tokenizer.convert_ids_to_tokens(kgid)][
                    max_text_length - num_added_kgid + 1 + pos
                ] = 1
    input_kgid = np.array(input_kgid)
    entity_pos = np.array(entity_pos[: num_added_kgid - 1])
    input_array = np.concatenate([input_array[: (max_text_length - num_added_kgid)], input_kgid])
    mask = np.array([1 if id != padid else 0 for id in input_array])
    mask_graph = np.concatenate([np.zeros(max_text_length - num_added_kgid, dtype=np.int64), input_kgid])
    if baseline == True:
        position_ids = np.arange(max_text_length)
    else:
        position_ids = np.concatenate([np.arange(max_text_length)[: (max_text_length - num_added_kgid + 1)], entity_pos])
    pad_array = np.array(input_array != padid, dtype=np.long)  # pad
    assert len(input_array) == len(position_ids)

    relations_vector = {}
    relations = {}
    for rel in relations_info:
        rel_type, e1, e2 = rel["type"], rel["subj"], rel["obj"]
        if e1 in entity_indicator and e2 in entity_indicator:
            if (e1, e2) not in relations_vector:
                relations_vector[(e1, e2)] = [0] * rel_vocab_size
            if rel_type in relation_map:
                # NA should not be in the relation_map to generate all-zero vector.
                relations_vector[(e1, e2)][relation_map[rel_type]] = 1
            if (e1, e2) not in relations:
                relations[(e1, e2)] = []
            relations[(e1, e2)].append(rel_type)

    label_vectors = []
    label_names = []
    e1_indicators = []
    e2_indicators = []
    e1_indicators_kg = []
    e2_indicators_kg = []
    e1s, e2s = [], []
    e1_types = []
    e2_types = []

    # in this mode, NA relation label occurs either when it is shown in the data, or there is no label between the pair
    for e1 in list(entity_id_set):
        for e2 in list(entity_id_set):
            if (entity_type[e1], entity_type[e2]) in ENTITY_PAIR_TYPE_SET and e1 != "-" and e2 != "-":
                e1s.append(e1)
                e2s.append(e2)

                e1_indicator, e2_indicator = (
                    entity_indicator[e1],
                    entity_indicator[e2],
                )  
                e1_indicator_kg, e2_indicator_kg = (
                    entity_indicator_kg[e1],
                    entity_indicator_kg[e2],
                )  
                if (e1, e2) in relations_vector:
                    label_vector = relations_vector[(e1, e2)]
                else:
                    label_vector = [0] * rel_vocab_size
                sys.stdout.flush()
                label_vectors.append(label_vector)
                if (e1, e2) in relations:
                    label_name = relations[(e1, e2)]
                else:
                    label_name = []
                label_names.append(label_name)
                e1_indicators.append(e1_indicator)
                e2_indicators.append(e2_indicator)
                e1_indicators_kg.append(e1_indicator_kg)
                e2_indicators_kg.append(e2_indicator_kg)
                e1_types.append(entity_type[e1])
                e2_types.append(entity_type[e2])
    assert len(e1_indicators) == len(e1_indicators_kg) == len(e2_indicators) == len(e2_indicators_kg)
    graph = get_subgraph(
        e1s=e1s, e2s=e2s, entid2kgid=entid2kgid, csr=sym_csr, upper_limit_index=upper_limit_index, num_hop=num_hop
    )
    sep_pos = input_length - 1
    return {
        "input": input_array,
        "pad": pad_array,
        "mask": mask,
        "mask_graph": mask_graph,
        "position": position_ids,
        "docid": docid,
        "input_length": input_length,
        "label_vectors": label_vectors,
        "label_names": label_names,
        "e1_indicators": e1_indicators,
        "e2_indicators": e2_indicators,
        "e1_indicators_kg": e1_indicators_kg,
        "e2_indicators_kg": e2_indicators_kg,
        "e1s": e1s,
        "e2s": e2s,
        "e1_types": e1_types,
        "e2_types": e2_types,
        "input_kgid": input_kgid,
        "graph": graph,
        "sep_pos": sep_pos,
        "num_kgid": num_added_kgid,
    }


class Dataloader(object):
    """Dataloader"""

    def __init__(
        self,
        data_path,
        tokenizer,
        entid2kgid,
        sym_csr,
        upper_limit_index,
        num_hop,
        device,
        baseline,
        debug_num,
        seed=0,
        max_text_length=512,
        training=False,
        logger=None,
        lowercase=True,
    ):
        # shape of input for each batch: (batchsize, max_text_length, max_sent_length)
        self.device = device
        self.entid2kgid = entid2kgid
        self.train = []
        self.val = []
        self.test_ctd = []
        self.test_ds_ctd = []
        self.test_anno_ctd = []
        self.test_anno_all = []
        self.tokenizer = tokenizer
        self.logger = logger

        self.relation_map = json.loads(open(data_path + "/relation_map.json").read())  # name_to_id
        self.relation_name = dict([(i, r) for r, i in self.relation_map.items()])  # id_to_name

        def calculate_stat(data):
            num_pos_rels = 0
            num_neg_pairs = 0
            num_pos_pairs = 0
            per_rel_stat = {}
            entity_type_pair_stat = {}
            for d in data:
                for i, (rel_names, e1t, e2t) in enumerate(list(zip(d["label_names"], d["e1_types"], d["e2_types"]))):
                    for rel_name in rel_names:
                        if rel_name not in per_rel_stat:
                            per_rel_stat[rel_name] = 0
                    if (e1t, e2t) not in entity_type_pair_stat:
                        entity_type_pair_stat[(e1t, e2t)] = {
                            "num_pos_pairs": 0,
                            "num_neg_pairs": 0,
                            "num_pos_rels": 0,
                        }

                    num_pos_ = sum(d["label_vectors"][i])
                    if num_pos_ == 0:
                        num_neg_pairs += 1
                        entity_type_pair_stat[(e1t, e2t)]["num_neg_pairs"] += 1
                    else:
                        num_pos_rels += num_pos_
                        num_pos_pairs += 1
                        entity_type_pair_stat[(e1t, e2t)]["num_pos_rels"] += num_pos_
                        entity_type_pair_stat[(e1t, e2t)]["num_pos_pairs"] += 1
                        for rel_name in rel_names:
                            per_rel_stat[rel_name] += 1

            return (
                num_pos_rels,
                num_pos_pairs,
                num_neg_pairs,
                entity_type_pair_stat,
                per_rel_stat,
            )

        with open(data_path + "/test.anno_all.json") as f:
            test_json = json.loads(f.read())
            docid2size_of_subgraph = {}
            docid2num_common_nodes = {}
            for i, data in enumerate(tqdm(test_json[:])):
                processed_data = preprocess(
                    data,
                    tokenizer,
                    max_text_length,
                    self.relation_map,
                    entid2kgid,
                    sym_csr,
                    # db_cur,
                    upper_limit_index,
                    num_hop,
                    baseline,
                    lowercase,
                )
                if processed_data["label_vectors"] == []:
                    continue
                self.test_anno_all.append(processed_data)
                if debug_num and i >= debug_num:
                    break
            

            (
                num_pos_rels,
                num_pos_pairs,
                num_neg_pairs,
                entity_type_pair_stat,
                per_rel_stat,
            ) = calculate_stat(self.test_anno_all)
            self.logger.info(f"=======================================")
            self.logger.info(f"Test annotated all:     # of docs = {len(test_json)}")
            self.logger.info(f"          # of positive pairs = {num_pos_pairs}")
            self.logger.info(f"          # of positive labels = {num_pos_rels}")
            self.logger.info(f"          # of negative pairs = {num_neg_pairs}")
            self.logger.info(f"---------------------------------------")
            for e1t, e2t in entity_type_pair_stat:
                self.logger.info(
                    f"          ({e1t}, {e2t}): # of positive pairs = {entity_type_pair_stat[(e1t, e2t)]['num_pos_pairs']}"
                )
                self.logger.info(
                    f"          ({e1t}, {e2t}): # of positive labels = {entity_type_pair_stat[(e1t, e2t)]['num_pos_rels']}"
                )
                self.logger.info(
                    f"          ({e1t}, {e2t}): # of negative pairs = {entity_type_pair_stat[(e1t, e2t)]['num_neg_pairs']}"
                )
            self.logger.info(f"---------------------------------------")
            for rel_name in per_rel_stat:
                self.logger.info(f"          {rel_name}: # of labels = {per_rel_stat[rel_name]}")
            self.logger.info(f"=======================================")
        if training == True:
            with open(data_path + "/train.json") as f: 
                train_json = json.loads(f.read())

                for i, data in enumerate(tqdm(train_json[:])):
                    processed_data = preprocess(
                        data,
                        tokenizer,
                        max_text_length,
                        self.relation_map,
                        entid2kgid,
                        sym_csr,
                        upper_limit_index,
                        num_hop,
                        baseline,
                        lowercase,
                    )
                    sys.stdout.flush()
                    if processed_data["label_vectors"] == []:
                        continue
                    self.train.append(processed_data)
                    if debug_num and i >= debug_num:
                        break
                (
                    num_pos_rels,
                    num_pos_pairs,
                    num_neg_pairs,
                    entity_type_pair_stat,
                    per_rel_stat,
                ) = calculate_stat(self.train)
                self.logger.info(f"=======================================")
                self.logger.info(f"Training: # of docs = {len(train_json)}")
                self.logger.info(f"          # of positive pairs = {num_pos_pairs}")
                self.logger.info(f"          # of positive labels = {num_pos_rels}")
                self.logger.info(f"          # of negative pairs = {num_neg_pairs}")
                self.logger.info(f"---------------------------------------")
                for e1t, e2t in entity_type_pair_stat:
                    self.logger.info(
                        f"          ({e1t}, {e2t}): # of positive pairs = {entity_type_pair_stat[(e1t, e2t)]['num_pos_pairs']}"
                    )
                    self.logger.info(
                        f"          ({e1t}, {e2t}): # of positive labels = {entity_type_pair_stat[(e1t, e2t)]['num_pos_rels']}"
                    )
                    self.logger.info(
                        f"          ({e1t}, {e2t}): # of negative pairs = {entity_type_pair_stat[(e1t, e2t)]['num_neg_pairs']}"
                    )
                self.logger.info(f"---------------------------------------")
                for rel_name in per_rel_stat:
                    self.logger.info(f"          {rel_name}: # of labels = {per_rel_stat[rel_name]}")
                self.logger.info(f"=======================================")

        with open(data_path + "/valid.json") as f:
            valid_json = json.loads(f.read())
            for i, data in enumerate(tqdm(valid_json[:])):
                processed_data = preprocess(
                    data,
                    tokenizer,
                    max_text_length,
                    self.relation_map,
                    entid2kgid,
                    sym_csr,
                    upper_limit_index,
                    num_hop,
                    baseline,
                    lowercase,
                )
                if processed_data["label_vectors"] == []:
                    continue
                self.val.append(processed_data)
                if debug_num and i >= debug_num:
                    break

            (
                num_pos_rels,
                num_pos_pairs,
                num_neg_pairs,
                entity_type_pair_stat,
                per_rel_stat,
            ) = calculate_stat(self.val)
            self.logger.info(f"=======================================")
            self.logger.info(f"Valid:    # of docs = {len(valid_json)}")
            self.logger.info(f"          # of positive pairs = {num_pos_pairs}")
            self.logger.info(f"          # of positive labels = {num_pos_rels}")
            self.logger.info(f"          # of negative pairs = {num_neg_pairs}")
            self.logger.info(f"---------------------------------------")
            for e1t, e2t in entity_type_pair_stat:
                self.logger.info(
                    f"          ({e1t}, {e2t}): # of positive pairs = {entity_type_pair_stat[(e1t, e2t)]['num_pos_pairs']}"
                )
                self.logger.info(
                    f"          ({e1t}, {e2t}): # of positive labels = {entity_type_pair_stat[(e1t, e2t)]['num_pos_rels']}"
                )
                self.logger.info(
                    f"          ({e1t}, {e2t}): # of negative pairs = {entity_type_pair_stat[(e1t, e2t)]['num_neg_pairs']}"
                )
            self.logger.info(f"---------------------------------------")
            for rel_name in per_rel_stat:
                self.logger.info(f"          {rel_name}: # of labels = {per_rel_stat[rel_name]}")
            self.logger.info(f"=======================================")

        with open(data_path + "/test.json") as f:
            test_json = json.loads(f.read())
            for i, data in enumerate(tqdm(test_json[:])):
                processed_data = preprocess(
                    data,
                    tokenizer,
                    max_text_length,
                    self.relation_map,
                    entid2kgid,
                    sym_csr,
                    upper_limit_index,
                    num_hop,
                    baseline,
                    lowercase,
                )
                if processed_data["label_vectors"] == []:
                    continue
                self.test_ctd.append(processed_data)
                if debug_num and i >= debug_num:
                    break

            (
                num_pos_rels,
                num_pos_pairs,
                num_neg_pairs,
                entity_type_pair_stat,
                per_rel_stat,
            ) = calculate_stat(self.test_ctd)
            self.logger.info(f"=======================================")
            self.logger.info(f"Test ctd:     # of docs = {len(test_json)}")
            self.logger.info(f"          # of positive pairs = {num_pos_pairs}")
            self.logger.info(f"          # of positive labels = {num_pos_rels}")
            self.logger.info(f"          # of negative pairs = {num_neg_pairs}")
            self.logger.info(f"---------------------------------------")
            for e1t, e2t in entity_type_pair_stat:
                self.logger.info(
                    f"          ({e1t}, {e2t}): # of positive pairs = {entity_type_pair_stat[(e1t, e2t)]['num_pos_pairs']}"
                )
                self.logger.info(
                    f"          ({e1t}, {e2t}): # of positive labels = {entity_type_pair_stat[(e1t, e2t)]['num_pos_rels']}"
                )
                self.logger.info(
                    f"          ({e1t}, {e2t}): # of negative pairs = {entity_type_pair_stat[(e1t, e2t)]['num_neg_pairs']}"
                )
            self.logger.info(f"---------------------------------------")
            for rel_name in per_rel_stat:
                self.logger.info(f"          {rel_name}: # of labels = {per_rel_stat[rel_name]}")
            self.logger.info(f"=======================================")

        with open(data_path + "/test.anno_ctd.json") as f:
            test_json = json.loads(f.read())
            for i, data in enumerate(tqdm(test_json[:])):
                processed_data = preprocess(
                    data,
                    tokenizer,
                    max_text_length,
                    self.relation_map,
                    entid2kgid,
                    sym_csr,
                    upper_limit_index,
                    num_hop,
                    baseline,
                    lowercase,
                )
                if processed_data["label_vectors"] == []:
                    continue
                self.test_anno_ctd.append(processed_data)
                if debug_num and i >= debug_num:
                    break

            (
                num_pos_rels,
                num_pos_pairs,
                num_neg_pairs,
                entity_type_pair_stat,
                per_rel_stat,
            ) = calculate_stat(self.test_anno_ctd)
            self.logger.info(f"=======================================")
            self.logger.info(f"Test annotated ctd:     # of docs = {len(test_json)}")
            self.logger.info(f"          # of positive pairs = {num_pos_pairs}")
            self.logger.info(f"          # of positive labels = {num_pos_rels}")
            self.logger.info(f"          # of negative pairs = {num_neg_pairs}")
            self.logger.info(f"---------------------------------------")
            for e1t, e2t in entity_type_pair_stat:
                self.logger.info(
                    f"          ({e1t}, {e2t}): # of positive pairs = {entity_type_pair_stat[(e1t, e2t)]['num_pos_pairs']}"
                )
                self.logger.info(
                    f"          ({e1t}, {e2t}): # of positive labels = {entity_type_pair_stat[(e1t, e2t)]['num_pos_rels']}"
                )
                self.logger.info(
                    f"          ({e1t}, {e2t}): # of negative pairs = {entity_type_pair_stat[(e1t, e2t)]['num_neg_pairs']}"
                )
            self.logger.info(f"---------------------------------------")
            for rel_name in per_rel_stat:
                self.logger.info(f"          {rel_name}: # of labels = {per_rel_stat[rel_name]}")
            self.logger.info(f"=======================================")

        self.max_text_length = max_text_length
        self._bz = 1
        self._datasize = len(self.train)
        self._idx = 0
        self.num_trained_data = 0
        random.seed(seed)
        random.shuffle(self.train)

    def __len__(self):
        return self._datasize

    def __iter__(self):
        while True:
            if self._idx + self._bz > self._datasize:
                random.shuffle(self.train)
                self._idx = 0
            # data = self.train[self._idx : (self._idx + self._bz)][0]
            data_list = self.train[self._idx : (self._idx + self._bz)]
            self._idx += self._bz
            (
                input_array,
                pad_array,
                position_ids,
                e1s,
                e2s,
                input_kgid,
                graph,
                num_kgid,
                label_array,
                ep_masks,
                e1_indicators,
                e2_indicators,
                e1_indicators_kg,
                e2_indicators_kg,
            ) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [])
            for data in data_list:
                input_array.append(data["input"])
                pad_array.append(data["pad"])
                position_ids.append(data["position"])
                e1s.append(data["e1s"])
                e2s.append(data["e2s"])
                input_kgid.append(data["input_kgid"])
                g = dgl.graph(data["graph"], num_nodes=len(self.entid2kgid))
                g = dgl.add_self_loop(g)
                g = g.to(self.device)
                graph.append(g)
                num_kgid.append(data["num_kgid"])

                if len(data["label_vectors"]) > 100:
                    shuffle_indexes = list(range(len(data["label_vectors"])))
                    random.shuffle(shuffle_indexes)
                    shuffle_indexes = shuffle_indexes[:100]
                else:
                    shuffle_indexes = list(range(len(data["label_vectors"])))

                if len(data["label_vectors"]) > 100:
                    random.shuffle(data["label_vectors"])
                    label_array_ = data["label_vectors"][:100]
                    random.shuffle(data["e1_indicators"])
                    e1_indicators_ = data["e1_indicators"][:100]
                    random.shuffle(data["e2_indicators"])
                    e2_indicators_ = data["e2_indicators"][:100]
                    random.shuffle(data["e1_indicators_kg"])
                    e1_indicators_kg_ = data["e1_indicators_kg"][:100]
                    random.shuffle(data["e2_indicators_kg"])
                    e2_indicators_kg_ = data["e2_indicators_kg"][:100]
                    
                else:
                    label_array_ = data["label_vectors"]
                    e1_indicators_ = data["e1_indicators"]
                    e2_indicators_ = data["e2_indicators"]
                    e1_indicators_kg_ = data["e1_indicators_kg"]
                    e2_indicators_kg_ = data["e2_indicators_kg"]
                
                label_array.append(label_array_)
                e1_indicators.append(e1_indicators_)
                e2_indicators.append(e2_indicators_)
                e1_indicators_kg.append(e1_indicators_kg_)
                e2_indicators_kg.append(e2_indicators_kg_)
                ep_masks_ = []
                for e1_indicator, e2_indicator in list(zip(list(e1_indicators_), list(e2_indicators_))):
                    ep_mask_ = np.full((self.max_text_length, self.max_text_length), -1e20)
                    ep_outer = 1 - np.outer(e1_indicator, e2_indicator)
                    ep_mask_ = ep_mask_ * ep_outer
                    ep_masks_.append(ep_mask_)
                ep_masks.append(ep_masks_)

            max_length = 512 
            input_ids = torch.tensor(np.array(input_array)[:, :max_length], dtype=torch.long)
            attention_mask = torch.tensor(np.array(pad_array)[:, :max_length], dtype=torch.long)
            attention_mask_graph = torch.tensor(np.array(pad_array)[:, :max_length], dtype=torch.long)
            position_ids = torch.tensor(np.array(position_ids), dtype=torch.long)
            label_array = torch.tensor(np.array(label_array), dtype=torch.float)
            ep_masks = torch.tensor(np.array(ep_masks)[:, :, :max_length, :max_length], dtype=torch.float)
            e1_indicators = np.array(e1_indicators)
            e2_indicators = np.array(e2_indicators)
            e1_indicators = torch.tensor(e1_indicators[:, :, :max_length], dtype=torch.float)
            e2_indicators = torch.tensor(e2_indicators[:, :, :max_length], dtype=torch.float)
            e1_indicators_kg = np.array(e1_indicators_kg)
            e2_indicators_kg = np.array(e2_indicators_kg)
            e1_indicators_kg = torch.tensor(e1_indicators_kg[:, :, :max_length], dtype=torch.float)
            e2_indicators_kg = torch.tensor(e2_indicators_kg[:, :, :max_length], dtype=torch.float)

            self.num_trained_data += self._bz

            return_data = (
                input_ids,
                attention_mask,
                attention_mask_graph,
                position_ids,
                ep_masks,
                e1_indicators,
                e2_indicators,
                e1_indicators_kg,
                e2_indicators_kg,
                label_array,
                e1s,
                e2s,
                input_kgid,
                graph,
                num_kgid,
            )
            yield self.num_trained_data, return_data
