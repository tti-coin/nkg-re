import numpy as np
import torch
from torch.cuda.amp import GradScaler
from transformers.optimization import AdamW
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_constant_schedule,
)
import os
import sys
import warnings
import json
import dgl
from tqdm import tqdm
from sklearn import metrics
from scipy.special import logsumexp
from module.utils import log1mexp
from torch import optim
import optuna

__all__ = [
    "Trainer",
]


def get_optimizer_grouped_parameters(model, config):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "gcn1.weight" in n
                or "gcn1.bias" in n
                or "gcn2.weight" in n
                or "gcn2.bias" in n
            ],
            "weight_decay": config["weight_decay"],
            "lr": config["gnn_learning_rate"],
        },
    ]
    optimizer_grouped_parameters += [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not (
                    "gcn1.weight" in n
                    or "gcn1.bias" in n
                    or "gcn2.weight" in n
                    or "gcn2.bias" in n
                )
            ],
            "weight_decay": config["weight_decay"],
            "lr": config["bert_learning_rate"],
        },
    ]

    return optimizer_grouped_parameters


class Trainer(object):
    def __init__(
        self,
        data=None,
        model=None,
        logger=None,
        config=None,
        device=None,
        grad_accum_count=1,
    ):

        self.data = data
        self.model = model
        self.logger = logger
        self.config = config
        self.device = device
        # setup optimizer
        self.scaler = GradScaler()
        # divide bert_opt from gcn_opt
        grouped_parameters = get_optimizer_grouped_parameters(
            model=self.model, config=self.config
        )
        if config["opt_divide"] == True:
            self.opt = optim.AdamW(grouped_parameters)
        else:
            self.opt = optim.AdamW(
                self.model.parameters(),
                lr=config["bert_learning_rate"],
                weight_decay=config["weight_decay"],
                eps=1e-8,
            )
        if config["warmup"] >= 0.0:
            self.scheduler = get_linear_schedule_with_warmup(
                self.opt,
                num_warmup_steps=config["warmup"],
                num_training_steps=config["max_num_steps"],
            )
        else:
            self.scheduler = get_constant_schedule(self.opt)

        self.bcelogitloss = torch.nn.BCEWithLogitsLoss()  # y is 1 or 0, x is 1-d logit
        # y is a non-negative integer, x is a multi dimensional logits
        self.celogitloss = torch.nn.CrossEntropyLoss()

    def save_model(self, best_metric_threshold):
        model_state_dict = self.model.state_dict()
        checkpoint = {
            "model": model_state_dict,
            "opt": self.opt,
            "threshold": best_metric_threshold,
        }
        checkpoint_path = os.path.join(self.config["output_path"], "model.pt")
        self.logger.info("Saving checkpoint %s" % checkpoint_path)
        torch.save(checkpoint, checkpoint_path)

    def load_model(self):
        checkpoint_path = os.path.join(self.config["load_path"])
        self.logger.info("Loading best checkpoint %s" % checkpoint_path)
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Model {checkpoint_path} does not exist.")
            return None
        print("before torch.load")
        sys.stdout.flush()
        checkpoint = torch.load(checkpoint_path)
        print("after torch.load")
        sys.stdout.flush()
        self.opt = vars(checkpoint["opt"])
        self.model.load_state_dict(checkpoint["model"])
        model_parameters = dict(
            [(name, params) for name, params in self.model.named_parameters()]
        )
        load_dict = dict(
            [(k, v) for k, v in checkpoint["model"].items() if k in model_parameters]
        )
        self.model.load_state_dict(load_dict, strict=False)

        return checkpoint["threshold"]

    def performance_logging(
        self,
        micro_perf,
        macro_perf,
        categ_acc,
        categ_macro_perf,
        na_acc,
        not_na_perf,
        na_perf,
        per_rel_perf,
        i,
        label="VAL",
    ):
        if len(self.data) == 0:
            data_length = 1
        else:
            data_length = len(self.data)
        self.logger.info(
            f"{i * 100 / data_length} % {label}: Micro P={micro_perf['P']}, Micro R={micro_perf['R']}, Micro F1={micro_perf['F']}, AP={micro_perf['AP']}"
        )
        self.logger.info(
            f"{i * 100 / data_length} % {label}: Macro P={macro_perf['P']}, Macro R={macro_perf['R']}, Macro F1={macro_perf['F']}"
        )
        self.logger.info(
            f"{i * 100 / data_length} % {label}: Categorical Accuracy={categ_acc}, Categorical Macro P={categ_macro_perf['P']}, Categorical Macro R={categ_macro_perf['R']}, Categorical Macro F1 ={categ_macro_perf['F']}"
        )
        self.logger.info(
            f"{i * 100 / data_length} % {label}: not_na Accuracy={na_acc}, not_na P={not_na_perf['P']}, not_na R={not_na_perf['R']}, not_na F1 ={not_na_perf['F']}"
        )
        self.logger.info(
            f"{i * 100 / data_length} % {label} na P={na_perf['P']}, na R={na_perf['R']}, na F1 ={na_perf['F']}"
        )
        for rel_name, (pp, rr, ff, tt) in per_rel_perf.items():
            self.logger.info(
                f"{i * 100 / data_length} % {label}: {rel_name}, P={pp}, R={rr}, F1={ff}, threshold={tt} (threshold not used for multiclass)"
            )

    def _train(self):
        self.logger.debug("This is training")

        if self.config["multi_label"] == True:

            def loss_func(input, target):
                return self.bcelogitloss(input, target)

        else:

            def loss_func(input, target):
                # input: batchsize, num_ep, R+1
                # target: batchsize, num_ep, R
                target = torch.cat(
                    [target, 1 - (target.sum(2, keepdim=True) > 0).float()], dim=2
                )  # (batchsize, R + 1)
                target = target.argmax(2)
                return self.celogitloss(input, target)  # input are logits

        best_metric = -1
        best_metric_threshold = {}
        patience = 0
        rolling_loss = []
        max_step = self.config["epochs"] * len(self.data)
        self.model.zero_grad()

        (
            macro_perf,
            micro_perf,
            categ_acc,
            categ_macro_perf,
            na_acc,
            not_na_perf,
            na_perf,
            per_rel_perf,
        ) = self.test("valid")

        best_metric = micro_perf["F"]
        self.performance_logging(
            micro_perf,
            macro_perf,
            categ_acc,
            categ_macro_perf,
            na_acc,
            not_na_perf,
            na_perf,
            per_rel_perf,
            0,
            label="VAL (a new best)",
        )
        (
            macro_perf,
            micro_perf,
            categ_acc,
            categ_macro_perf,
            na_acc,
            not_na_perf,
            na_perf,
            per_rel_perf,
        ) = self.test("test_ctd", best_metric_threshold=best_metric_threshold)
        sys.stdout.flush()
        self.performance_logging(
            micro_perf,
            macro_perf,
            categ_acc,
            categ_macro_perf,
            na_acc,
            not_na_perf,
            na_perf,
            per_rel_perf,
            0,
            label="TEST CTD",
        )

        (
            macro_perf,
            micro_perf,
            categ_acc,
            categ_macro_perf,
            na_acc,
            not_na_perf,
            na_perf,
            per_rel_perf,
        ) = self.test("test_anno_ctd", best_metric_threshold=best_metric_threshold)
        sys.stdout.flush()
        self.performance_logging(
            micro_perf,
            macro_perf,
            categ_acc,
            categ_macro_perf,
            na_acc,
            not_na_perf,
            na_perf,
            per_rel_perf,
            0,
            label="TEST ANNOTATED CTD",
        )

        (
            macro_perf,
            micro_perf,
            categ_acc,
            categ_macro_perf,
            na_acc,
            not_na_perf,
            na_perf,
            per_rel_perf,
        ) = self.test("test_anno_all", best_metric_threshold=best_metric_threshold)
        sys.stdout.flush()
        self.performance_logging(
            micro_perf,
            macro_perf,
            categ_acc,
            categ_macro_perf,
            na_acc,
            not_na_perf,
            na_perf,
            per_rel_perf,
            0,
            label="TEST ANNOTATED ALL",
        )

        with open(self.config["hkg_data_path"] + "/entid2kgid.json") as f:
            entid2kgid = json.load(f)

        for i, batch in iter(self.data):
            (
                input_ids,
                attention_mask,
                attention_mask_graph,
                position_ids,
                ep_masks,
                e1_indicator,
                e2_indicator,
                e1_indicator_kg,
                e2_indicator_kg,
                label_array,
                e1s,
                e2s,
                input_kgid,
                graph,
                num_kgid,
            ) = batch
            self.model.train(True)
            input_kgid = list(map(set, input_kgid))
            attention_mask = attention_mask.to(self.device)
            attention_mask_graph = attention_mask_graph.to(self.device)
            position_ids = position_ids.to(self.device)
            input_ids = input_ids.to(self.device)
            ep_masks = ep_masks.to(self.device)
            e1_indicator = e1_indicator.to(self.device)
            e2_indicator = e2_indicator.to(self.device)
            e1_indicator_kg = e1_indicator_kg.to(self.device)
            e2_indicator_kg = e2_indicator_kg.to(self.device)
            label_array = label_array.to(self.device)

            scores = self.model(
                input_ids,
                attention_mask,
                attention_mask_graph,
                position_ids,
                ep_masks,
                e1_indicator,
                e2_indicator,
                e1_indicator_kg,
                e2_indicator_kg,
                graph,
                e1s,
                e2s,
                input_kgid,
                num_kgid,
            )  # batchsize, num_ep, R or batchsize, num_ep, R + 1

            loss = loss_func(scores, label_array.to(self.device))
            # bert_norm = float(bert_norm_.detach().cpu())
            # node_norm = float(node_norm_.detach().cpu())
            sys.stdout.flush()
            """back prop"""

            loss = loss / self.config["grad_accumulation_steps"]
            self.scaler.scale(loss).backward()  # backprop

            for param in self.model.parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any()

            if i % self.config["grad_accumulation_steps"] == 0:
                if self.config["max_grad_norm"] > 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["max_grad_norm"]
                    )  # clip gradient norm
                self.scaler.step(self.opt)
                sys.stdout.flush()
                self.scaler.update()
                self.scheduler.step()
                self.model.zero_grad()
                # emb = torch.randn(num_nodes, self.graph_dim) 

            """End"""
            rolling_loss.append(float(loss.detach().cpu()))
            if i % 10000 == 0:
                self.logger.info(f"{i}-th example loss: {np.mean(rolling_loss)}")
                print(f"{i}-th example loss: {np.mean(rolling_loss)}")
                rolling_loss = []

            # evaluate on dev set (if out-performed, evaluate on test as well)
            if i % self.config["log_interval"] == self.config["log_interval"] - 1:
                self.model.eval()

                # per_rel_perf is a list of length of (number of relation types + 1)
                (
                    macro_perf,
                    micro_perf,
                    categ_acc,
                    categ_macro_perf,
                    na_acc,
                    not_na_perf,
                    na_perf,
                    per_rel_perf,
                ) = self.test("valid")

                yield micro_perf["F"]  # optuna

                self.logger.info(f'val: {micro_perf["F"]}, {not_na_perf["F"]}')
                print(f'val: {micro_perf["F"]}, {not_na_perf["F"]}')

                if micro_perf["F"] > best_metric:

                    best_metric = micro_perf["F"]
                    self.performance_logging(
                        micro_perf,
                        macro_perf,
                        categ_acc,
                        categ_macro_perf,
                        na_acc,
                        not_na_perf,
                        na_perf,
                        per_rel_perf,
                        i,
                        label="VAL (a new best)",
                    )

                    #     best_metric_threshold[rel_name] = tt
                    for rel_name, (pp, rr, ff, tt) in per_rel_perf.items():
                        best_metric_threshold[rel_name] = tt
                    patience = 0
                    self.save_model(best_metric_threshold)

                    # evaluate on test set
                    (
                        macro_perf,
                        micro_perf,
                        categ_acc,
                        categ_macro_perf,
                        na_acc,
                        not_na_perf,
                        na_perf,
                        per_rel_perf,
                    ) = self.test(
                        "test_ctd", best_metric_threshold=best_metric_threshold
                    )
                    sys.stdout.flush()
                    self.performance_logging(
                        micro_perf,
                        macro_perf,
                        categ_acc,
                        categ_macro_perf,
                        na_acc,
                        not_na_perf,
                        na_perf,
                        per_rel_perf,
                        i,
                        label="TEST CTD",
                    )

                    (
                        macro_perf,
                        micro_perf,
                        categ_acc,
                        categ_macro_perf,
                        na_acc,
                        not_na_perf,
                        na_perf,
                        per_rel_perf,
                    ) = self.test(
                        "test_anno_ctd", best_metric_threshold=best_metric_threshold
                    )
                    sys.stdout.flush()
                    self.performance_logging(
                        micro_perf,
                        macro_perf,
                        categ_acc,
                        categ_macro_perf,
                        na_acc,
                        not_na_perf,
                        na_perf,
                        per_rel_perf,
                        i,
                        label="TEST ANNOTATED CTD",
                    )

                    (
                        macro_perf,
                        micro_perf,
                        categ_acc,
                        categ_macro_perf,
                        na_acc,
                        not_na_perf,
                        na_perf,
                        per_rel_perf,
                    ) = self.test(
                        "test_anno_all", best_metric_threshold=best_metric_threshold
                    )
                    sys.stdout.flush()
                    self.performance_logging(
                        micro_perf,
                        macro_perf,
                        categ_acc,
                        categ_macro_perf,
                        na_acc,
                        not_na_perf,
                        na_perf,
                        per_rel_perf,
                        i,
                        label="TEST ANNOTATED ALL",
                    )

                else:
                    self.performance_logging(
                        micro_perf,
                        macro_perf,
                        categ_acc,
                        categ_macro_perf,
                        na_acc,
                        not_na_perf,
                        na_perf,
                        per_rel_perf,
                        i,
                        label="VAL",
                    )

                    patience += 1

            # early stop
            if patience > self.config["patience"]:
                self.logger.info("triggers early stop; ended")
                break
            if i > max_step:
                self.logger.info("exceeds maximum steps; ended")
                break
    
    def train(self):
        return list(self._train())

    def optuna(self):
        if not self.config["optuna_study_name"]:
            self.train(self)
        else:
            def objective(trial):
                if self.config["opt_divide"]:
                    self.config["gnn_learning_rate"] = trial.suggest_float("gnn_learning_rate", 1e-5, 1e-2, log=True)
                self.config["bert_learning_rate"] = trial.suggest_float("bert_learning_rate", 1e-5, 1e-2, log=True)

                max_score = float('inf')
                for step, score in enumerate(self._train()):
                    max_score = max(max_score, score)
                    trial.report(score, step)
                    if trial.shoud_prune():
                        raise optuna.TrialPruned()
                return max_score
            pruner = optuna.pruners.SuccessiveHalvingPruner(
                min_resource="auto", reduction_factor=4, min_early_stopping_rate=0
            )
            study = optuna.create_study(
                study_name=self.config["optuna_study_name"],
                storage=self.config["optuna_storage"],
                load_if_exists=True,
                direction="maximize",
                pruner=pruner,
            )
            study.optimize(objective, n_trials=self.config["optuna_n_trials"])
            # if without SQL, output tsv
            if not self.config["optuna_storage"]:
                study.trials_dataframe().to_csv(self.config["optuna_study_name"]+".tsv", sep="\t")


    def calculate_metrics(self, predictions, predictions_categ, labels):
        # calcuate metrics given prediction and labels
        # predictions: (N, R), does not include NA in R
        # labels: (N, R), one and zeros, does not include NA in R
        # predictions_categ: (N, R), contains predictions for calculating performance of categorical classifier (exclude NA)
        TPs = predictions * labels  # (N, R)
        TP = TPs.sum()
        P = predictions.sum()
        T = labels.sum()

        micro_p = TP / P if P != 0 else 0
        micro_r = TP / T if T != 0 else 0
        micro_f = (
            2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r > 0 else 0
        )

        categ_TPs = predictions_categ * labels
        categ_TP = categ_TPs.sum()
        # exludes instance whose label is NA
        categ_Ps = predictions_categ * (labels.sum(1) > 0)[:, None]

        categ_acc = categ_TP / T if T != 0 else 0

        not_NA_Ps = predictions.sum(1) > 0
        not_NA_Ts = labels.sum(1) > 0
        not_NA_TPs = not_NA_Ps * not_NA_Ts
        not_NA_P = not_NA_Ps.sum()
        not_NA_T = not_NA_Ts.sum()
        not_NA_TP = not_NA_TPs.sum()
        not_NA_prec = not_NA_TP / not_NA_P if not_NA_P != 0 else 0
        not_NA_recall = not_NA_TP / not_NA_T if not_NA_T != 0 else 0
        not_NA_f = (
            2 * not_NA_prec * not_NA_recall / (not_NA_prec + not_NA_recall)
            if not_NA_prec + not_NA_recall > 0
            else 0
        )

        not_NA_acc = (not_NA_Ps == not_NA_Ts).mean()

        NA_Ps = predictions.sum(1) == 0
        NA_Ts = labels.sum(1) == 0
        NA_TPs = NA_Ps * NA_Ts
        NA_P = NA_Ps.sum()
        NA_T = NA_Ts.sum()
        NA_TP = NA_TPs.sum()
        NA_prec = NA_TP / NA_P if NA_P != 0 else 0
        NA_recall = NA_TP / NA_T if NA_T != 0 else 0
        NA_f = (
            2 * NA_prec * NA_recall / (NA_prec + NA_recall)
            if NA_prec + NA_recall > 0
            else 0
        )

        per_rel_p = np.zeros(predictions.shape[1])
        per_rel_r = np.zeros(predictions.shape[1])
        per_rel_f = np.zeros(predictions.shape[1])
        categ_per_rel_p = np.zeros(predictions.shape[1])
        categ_per_rel_r = np.zeros(predictions.shape[1])
        categ_per_rel_f = np.zeros(predictions.shape[1])
        # per relation metrics:
        for i in range(predictions.shape[1]):
            TP_ = TPs[:, i].sum()
            P_ = predictions[:, i].sum()
            T_ = labels[:, i].sum()
            categ_TP_ = categ_TPs[:, i].sum()
            categ_P_ = categ_Ps[:, i].sum()

            # if no such relation in the test data, recall = 0
            per_rel_r[i] = TP_ / T_ if T_ != 0 else 0
            categ_per_rel_r[i] = categ_TP_ / T_ if T_ != 0 else 0

            # if no such relation in the prediction, precision = 0
            per_rel_p[i] = TP_ / P_ if P_ != 0 else 0

            # if no such relation in the prediction, precision = 0
            categ_per_rel_p[i] = categ_TP_ / categ_P_ if categ_P_ != 0 else 0

            per_rel_f[i] = (
                2 * per_rel_p[i] * per_rel_r[i] / (per_rel_p[i] + per_rel_r[i])
                if per_rel_p[i] + per_rel_r[i] > 0
                else 0
            )

            categ_per_rel_f[i] = (
                2
                * categ_per_rel_p[i]
                * categ_per_rel_r[i]
                / (categ_per_rel_p[i] + categ_per_rel_r[i])
                if categ_per_rel_p[i] + categ_per_rel_r[i] > 0
                else 0
            )

        macro_p = per_rel_p.mean()
        macro_r = per_rel_r.mean()
        macro_f = per_rel_f.mean()

        categ_macro_p = categ_per_rel_p.mean()
        categ_macro_r = categ_per_rel_r.mean()
        categ_macro_f = categ_per_rel_f.mean()

        results = {
            "micro_p": micro_p,
            "micro_r": micro_r,
            "micro_f": micro_f,
            "macro_p": macro_p,
            "macro_r": macro_r,
            "macro_f": macro_f,
            "categ_acc": categ_acc,
            "categ_macro_p": categ_macro_p,
            "categ_macro_r": categ_macro_r,
            "categ_macro_f": categ_macro_f,
            "na_acc": not_NA_acc,
            "not_na_p": not_NA_prec,
            "not_na_r": not_NA_recall,
            "not_na_f": not_NA_f,
            "na_p": NA_prec,
            "na_r": NA_recall,
            "na_f": NA_f,
            "per_rel_p": per_rel_p,
            "per_rel_r": per_rel_r,
            "per_rel_f": per_rel_f,
            "categ_per_rel_p": categ_per_rel_p,
            "categ_per_rel_r": categ_per_rel_r,
            "categ_per_rel_f": categ_per_rel_f,
        }

        return results

    def test(self, data_name, best_metric_threshold=None):

        # load data
        if data_name == "valid":
            test_mode = True
            data = self.data.val
        elif data_name == "test_ctd":
            test_mode = True
            data = self.data.test_ctd
        elif data_name == "test_anno_ctd":
            test_mode = True
            data = self.data.test_anno_ctd
        elif data_name == "test_anno_all":
            test_mode = True
            data = self.data.test_anno_all

        self.logger.debug("This is " + data_name)
        # output results to a seperate file
        fout = open(os.path.join(self.config["output_path"], f"{data_name}.json"), "w")
        fout_json = {
            "threshold": {},
            "predictions": [],
            "results": {"macro": {}, "micro": {}, "per_rel": {}},
        }
        threshold_vec = np.zeros(len(self.data.relation_map))
        if best_metric_threshold != None:
            for rel, thres in best_metric_threshold.items():
                relid = self.data.relation_map[rel]
                threshold_vec[relid] = thres
                fout_json["threshold"][rel] = float(thres)
        else:
            warnings.warn(
                "evaluation on test data requires best_metric_threshold, use all zero thresholds instead"
            )

        sys.stdout.flush()

        # Infer scores for valid/test data
        total_num_ep = 0
        for d in data:
            total_num_ep += int(len(d["e1_indicators"]))

        with torch.no_grad():
            if self.config["multi_label"] == True:
                # (num_test, R)
                scores = np.zeros((total_num_ep, len(self.data.relation_map)))
            else:
                # (num_test, R + 1)
                scores = np.zeros((total_num_ep, len(self.data.relation_map) + 1))

            # (num_test, R)
            labels = np.zeros((total_num_ep, len(self.data.relation_map)))
            self.logger.debug(f"length of data: {len(data)}")

            count_index = 0
            # docid2bert_norm = {}
            # docid2node_norm = {}
            for i, dict_ in tqdm(enumerate(data)):
                (
                    input_array,
                    pad_array,
                    label_array,
                    docid,
                    ep_masks,
                    e1_indicators,
                    e2_indicators,
                    e1_indicators_kg,
                    e2_indicators_kg,
                    e1id,
                    e2id,
                ) = ([], [], [], [], [[]], [[]], [[]], [[]], [[]], [[]], [[]])
                input_array.append(dict_["input"])
                pad_array.append(dict_["pad"])
                label_array.append(dict_["label_vectors"])
                label_array = np.array(label_array).squeeze(axis=0)  # (num_ep, R)
                docid.append(dict_["docid"])
                max_length = dict_["input_length"]
                num_ep = int(len(dict_["e1_indicators"]))
                e1s = dict_["e1s"]
                e2s = dict_["e2s"]
                subgraph = dict_["graph"]
                input_kgid = list(set(dict_["input_kgid"]))
                position_ids = np.expand_dims(dict_["position"], axis=0)
                attention_mask = np.expand_dims(dict_["mask"], axis=0)
                attention_mask_graph = np.expand_dims(dict_["mask_graph"], axis=0)
                input_length = dict_["input_length"]
                sep_pos = dict_["sep_pos"]
                num_kgid = dict_["num_kgid"]
                # bert_norm_list = []
                # node_norm_list = []
                with open(self.config["hkg_data_path"] + "/entid2kgid.json") as f:
                    entid2kgid = json.load(f)

                for j in range(len(dict_["e1_indicators"])):
                    # (1, num_ep, R)
                    e1_indicators[0].append(dict_["e1_indicators"][j])
                    e2_indicators[0].append(dict_["e2_indicators"][j])
                    e1_indicators_kg[0].append(dict_["e1_indicators_kg"][j])
                    e2_indicators_kg[0].append(dict_["e2_indicators_kg"][j])
                    e1id[0].append(dict_["e1s"][j])
                    e2id[0].append(dict_["e2s"][j])

                    count_index += 1

                    ep_mask_ = np.full(
                        (self.data.max_text_length, self.data.max_text_length), -1e20
                    )
                    ep_outer = 1 - np.outer(
                        dict_["e1_indicators"][j], dict_["e2_indicators"][j]
                    )
                    ep_mask_ = ep_mask_ * ep_outer
                    ep_masks[0].append(ep_mask_)
                    max_length = 512  # 応急処置
                    if (
                        len(e1_indicators[0]) == 100
                        or j == len(dict_["e1_indicators"]) - 1
                    ):
                        input_ids = torch.tensor(
                            np.array(input_array)[:, :max_length], dtype=torch.long
                        ).to(self.device)

                        attention_mask = torch.tensor(
                            attention_mask[:, :max_length], dtype=torch.long
                        ).to(self.device)
                        attention_mask_graph = torch.tensor(
                            attention_mask_graph[:, :max_length], dtype=torch.long
                        ).to(self.device)
                        position_ids = torch.tensor(
                            position_ids[:, :max_length], dtype=torch.long
                        ).to(
                            self.device
                        )  # attention_mask = torch.tensor(
                        #     np.array(pad_array)[:, :max_length], dtype=torch.long
                        # ).to(self.device)
                        e1_indicator = torch.tensor(
                            np.array(e1_indicators)[:, :, :max_length],
                            dtype=torch.float,
                        ).to(self.device)
                        e2_indicator = torch.tensor(
                            np.array(e2_indicators)[:, :, :max_length],
                            dtype=torch.float,
                        ).to(self.device)
                        e1_indicator_kg = torch.tensor(
                            np.array(e1_indicators_kg)[:, :, :max_length],
                            dtype=torch.float,
                        ).to(self.device)
                        e2_indicator_kg = torch.tensor(
                            np.array(e2_indicators_kg)[:, :, :max_length],
                            dtype=torch.float,
                        ).to(self.device)
                        ep_masks = torch.tensor(
                            np.array(ep_masks)[:, :, :max_length, :max_length],
                            dtype=torch.float,
                        ).to(self.device)
                        num = ep_masks.size(1)
                        assert e1_indicator.shape == e1_indicator_kg.shape == e2_indicator.shape == e2_indicator_kg.shape
                        graph = dgl.graph(subgraph, num_nodes=len(entid2kgid))
                        graph = dgl.add_self_loop(graph)
                        graph = graph.to(self.device)  # graph to GPU
                        graph = [graph]
                        score = self.model(
                            input_ids,
                            attention_mask,
                            attention_mask_graph,
                            position_ids,
                            ep_masks,
                            e1_indicator,
                            e2_indicator,
                            e1_indicator_kg,
                            e2_indicator_kg,
                            graph,
                            e1s,
                            e2s,
                            input_kgid,
                            num_kgid,
                        )  # (1, num_ep, R) or (1, num_ep, R+1)
                        score = score.detach().cpu().numpy().squeeze(axis=0)
                        scores[
                            (count_index - int(score.shape[0])) : count_index, :
                        ] = score


                        ep_masks, e1_indicators, e2_indicators, e1_indicators_kg, e2_indicators_kg = (
                            [[]],
                            [[]],
                            [[]],
                            [[]],
                            [[]],
                        )


                labels[(count_index - num_ep) : count_index, :] = label_array  # (num_ep, R)

                # in test mode, save predictions for each data point (docid, e1, e2)
                if self.config["multi_label"] == True:
                    # (num_ep, R)
                    prediction = score > threshold_vec
                else:
                    prediction = np.zeros_like(score)  # (num_ep, R + 1)
                    prediction[np.arange(score.shape[0]), np.argmax(score, 1)] = 1

                    # (batchsize, R), predicts NA if prediction[:, :-1] is all-zero
                    prediction = prediction[:, :-1]

                predict_triples = []
                label_triples = []
                for j in range(len(prediction)):
                    for k in list(np.where(prediction[j] == 1)[0]):
                        predict_triples.append((e1s[j], self.data.relation_name[k], e2s[j])) # (e1s, rel, e2s)
                    for k in list(np.where(label_array[j] == 1)[0]):
                        label_triples.append((e1s[j], self.data.relation_name[k], e2s[j])) # (e1s, rel, e2s)
                    score_dict = {}
                    for k, scr in enumerate(list(score[j])):
                        if k not in self.data.relation_name:
                            score_dict["NA"] = float(scr)
                        else:
                            score_dict[self.data.relation_name[k]] = float(scr)
                fout_json["predictions"].append(
                    {
                        "docid": docid[0],
                        "label_triples": label_triples,
                        "predict_triples": predict_triples,
                        "scores": score_dict,
                    }
                )

        average_precision = metrics.average_precision_score(
            labels.flatten(), scores.flatten()
        )


        # calculate metrics for valid/test data
        if test_mode == True:
            # in test mode, use existing thresholds, save results.

            if self.config["multi_label"] == True and data_name == "valid":
                # in validation model, tune the thresholds
                for i, rel_name in self.data.relation_name.items():
                    (
                        prec_array,
                        recall_array,
                        threshold_array,
                    ) = metrics.precision_recall_curve(labels[:, i], scores[:, i])
                    prec_array_ = np.where(
                        prec_array + recall_array > 0,
                        prec_array,
                        np.ones_like(prec_array),
                    )
                    f1_array = np.where(
                        prec_array + recall_array > 0,
                        2 * prec_array * recall_array / (prec_array_ + recall_array),
                        np.zeros_like(prec_array),
                    )
                    best_threshold = threshold_array[np.argmax(f1_array)]
                    threshold_vec[i] = best_threshold
                predictions = scores > threshold_vec  # (num_test, R)
                predictions_categ = predictions

            elif self.config["multi_label"] == True:
                predictions = scores > threshold_vec  # (num_test, R)
                predictions_categ = predictions
            else:
                # if multi_class, choose argmax when the model predicts multiple labels
                predictions = np.zeros_like(scores)  # (num_test, R + 1)
                predictions[np.arange(scores.shape[0]), np.argmax(scores, 1)] = 1

                # (num_test, R), predicts NA if prediction[:, :-1] is all-zero
                predictions = predictions[:, :-1]

                predictions_categ = np.zeros_like(scores)[:, :-1]
                predictions_categ[
                    np.arange(scores.shape[0]), np.argmax(scores[:, :-1], 1)
                ] = 1

            results = self.calculate_metrics(predictions, predictions_categ, labels)

            fout_json["results"]["micro"] = {
                "P": float(results["micro_p"]),
                "R": float(results["micro_r"]),
                "F": float(results["micro_f"]),
                "AP": float(average_precision),
            }
            fout_json["results"]["macro"] = {
                "P": float(results["macro_p"]),
                "R": float(results["macro_r"]),
                "F": float(results["macro_f"]),
            }
            fout_json["results"]["categ_acc"] = float(results["categ_acc"])
            fout_json["results"]["categ_macro"] = {
                "P": float(results["categ_macro_p"]),
                "R": float(results["categ_macro_r"]),
                "F": float(results["categ_macro_f"]),
            }
            fout_json["results"]["na_acc"] = float(results["na_acc"])
            fout_json["results"]["not_na"] = {
                "P": float(results["not_na_p"]),
                "R": float(results["not_na_r"]),
                "F": float(results["not_na_f"]),
            }
            fout_json["results"]["na"] = {
                "P": float(results["na_p"]),
                "R": float(results["na_r"]),
                "F": float(results["na_f"]),
            }
            for i, rel_name in self.data.relation_name.items():
                fout_json["results"]["per_rel"][rel_name] = {
                    "P": float(results["per_rel_p"][i]),
                    "R": float(results["per_rel_r"][i]),
                    "F": float(results["per_rel_f"][i]),
                }

        else:
            if self.config["multi_label"] == True:
                # in validation model, tune the thresholds
                for i, rel_name in self.data.relation_name.items():
                    (
                        prec_array,
                        recall_array,
                        threshold_array,
                    ) = metrics.precision_recall_curve(labels[:, i], scores[:, i])
                    prec_array_ = np.where(
                        prec_array + recall_array > 0,
                        prec_array,
                        np.ones_like(prec_array),
                    )
                    f1_array = np.where(
                        prec_array + recall_array > 0,
                        2 * prec_array * recall_array / (prec_array_ + recall_array),
                        np.zeros_like(prec_array),
                    )
                    best_threshold = threshold_array[np.argmax(f1_array)]
                    threshold_vec[i] = best_threshold
                predictions = scores > threshold_vec  # (num_test, R)
                predictions_categ = predictions
            else:
                # if multi_class, choose argmax
                predictions = np.zeros_like(scores)  # (num_test, R + 1)
                predictions[np.arange(scores.shape[0]), np.argmax(scores, 1)] = 1

                # (num_test, R), predicts NA if prediction[:, :-1] is all-zero
                predictions = predictions[:, :-1]

                predictions_categ = np.zeros_like(scores)[:, :-1]
                predictions_categ[
                    np.arange(scores.shape[0]), np.argmax(scores[:, :-1], 1)
                ] = 1

            results = self.calculate_metrics(predictions, predictions_categ, labels)

        macro_perf = {
            "P": results["macro_p"],
            "R": results["macro_r"],
            "F": results["macro_f"],
        }
        micro_perf = {
            "P": results["micro_p"],
            "R": results["micro_r"],
            "F": results["micro_f"],
            "AP": float(average_precision),
        }
        categ_macro_perf = {
            "P": results["categ_macro_p"],
            "R": results["categ_macro_r"],
            "F": results["categ_macro_f"],
        }
        not_na_perf = {
            "P": results["not_na_p"],
            "R": results["not_na_r"],
            "F": results["not_na_f"],
        }
        na_perf = {"P": results["na_p"], "R": results["na_r"], "F": results["na_f"]}
        per_rel_perf = {}
        for i, rel_name in self.data.relation_name.items():
            per_rel_perf[rel_name] = [
                results["per_rel_p"][i],
                results["per_rel_r"][i],
                results["per_rel_f"][i],
                threshold_vec[i],
            ]

        # if test_mode == True:
        fout.write(json.dumps(fout_json, indent="\t"))
        fout.close()

        return (
            macro_perf,
            micro_perf,
            results["categ_acc"],
            categ_macro_perf,
            results["na_acc"],
            not_na_perf,
            na_perf,
            per_rel_perf,
        )
