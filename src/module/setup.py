import time
import os
import sys
import logging
import json
from transformers import AutoTokenizer

from module.data_loader import Dataloader
from module.utils import cuda_if_available
from module.model import Model
from scipy.sparse import load_npz


__all__ = [
    "setup",
]


def setup(config):
    """
    Setup and return the datasets, dataloaders, model, logger, and training loop required for training.
    :param config: config dictionary, config will also be modified
    :return: Tuple of dataset collection, dataloader collection, model, and train looper
    """

    if config["output_path"] == "":
        base_dir_name = "_".join(
            [
                config["data_path"].strip("/").split("/")[-1],
                config["encoder_type"].replace("/", "-"),
                config["model"],
                "multilabel_" + str(config["multi_label"]),
                "lr_" + str(config["bert_learning_rate"])[:6],
                "dim_" + str(config["dim"]),
                "acc_" + str(config["grad_accumulation_steps"]),
                "len_" + str(config["max_text_length"]),
                "decay_" + str(config["weight_decay"])[:6],
                "dp_" + str(config["dropout_rate"]),
                "warmup_" + str(config["warmup"]),
                "seed_" + str(config["seed"])[:6],
            ]
        )
        config["data_path"] = config["data_path"].rstrip("/")
        config["output_path"] = os.path.join(
            os.environ["BIORE_OUTPUT_ROOT"],
            "saved_models",
            os.path.basename(config["data_path"]),
            base_dir_name,
        )

    print("output_path", config["output_path"])
    sys.stdout.flush()
    if not os.path.exists(config["output_path"]):
        os.makedirs(config["output_path"])

    logger = logging.getLogger("BioRE")
    # LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()  # DEBUG, INFO
    LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()  # DEBUG, INFO
    logger.setLevel(LOGLEVEL)
    logging_output = logging.FileHandler(config["output_path"] + "/log")
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    logging_output.setFormatter(formatter)
    logger.addHandler(logging_output)

    logger.info("Program started")
    logger.info(config)

    device = cuda_if_available(use_cuda=config["cuda"], cuda=config["cuda_num"])
    lowercase = True if "uncased" in config["encoder_type"] else False

    # setup tokenizer
    if config["encoder_type"] in ["transformer", "transformer_conv"]:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)
    else:  
        tokenizer = AutoTokenizer.from_pretrained(config["encoder_type"], use_fast=True)

    # add entity marker token into vocab 

    tokenizer_org_size = len(tokenizer)
    print("tokenizer original size", tokenizer_org_size)
    config["tokenizer_org_size"] = tokenizer_org_size

    if os.path.exists(config["data_path"] + "/entid2kgid.json"):
        with open(config["data_path"] + "/entid2kgid.json") as f:
            entid2kgid = json.load(f)
        tokenizer.add_tokens(list(entid2kgid.keys()))
        # tokenizer.add_tokens(["[BLANK]"])
    config["vocabsize"] = len(tokenizer)
    tokenizer_org_size = len(tokenizer) - len(entid2kgid)
    config["tokenizer_org_size"] = tokenizer_org_size
    print(f"tokenizer size: {tokenizer_org_size} --> {len(tokenizer)}")

    csr_path = config["hkg_data_path"] + config["csr_path"]
    sym_csr = load_npz(csr_path)

    upper_limit_index = config["upper_limit_index"]
    num_hop = config["num_hop"]
    baseline = config["baseline"]
    debug_num = config["debug_num"]
    # setup data
    time1 = time.time()

    data = Dataloader(
        config["data_path"],
        tokenizer,
        entid2kgid,
        sym_csr,
        upper_limit_index,
        num_hop,
        device,
        baseline,
        debug_num,
        max_text_length=config["max_text_length"],
        training=(config["mode"] == "train"),
        logger=logger,
        lowercase=lowercase,
    )

    time2 = time.time()
    logger.info("Time spent loading data: %f" % (time2 - time1))
    if config["mode"] == "train":
        logger.info(f"number of data points during training : {len(data)}")
        config["max_num_steps"] = len(data) * config["epochs"]
        if isinstance(config["log_interval"], float):
            config["log_interval"] = len(data) * config["log_interval"]
            logger.info(f"Log interval: {config['log_interval']}")
            print("Log interval: ", config["log_interval"])
        else:
            logger.info(f"Log interval: {config['log_interval']}")
            print("Log interval: ", config["log_interval"])

        if isinstance(config["warmup"], float):
            config["warmup"] = config["max_num_steps"] * config["warmup"]
            logger.info(f"warmup {config['warmup']}")

    # setup model
    model = Model(config)
    model.to(device)
    model.feat = model.feat.to(device)
    # create token embedding for new added markers
    if config["encoder_type"] not in ["transformer_conv", "transformer"]:
        model.encoder.resize_token_embeddings(config["vocabsize"])
        model.encoder.config.hidden_dropout_prob = config["dropout_rate"]

    return data, model, device, logger
