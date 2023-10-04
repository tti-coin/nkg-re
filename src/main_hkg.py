# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import random
import click
import torch
import numpy as np
from module.setup import setup
from module.train import Trainer


# %%
class IntOrPercent(click.ParamType):
    name = "click_union"

    def convert(self, value, param, ctx):
        try:
            float_value = float(value)
            if 0 <= float_value <= 1:
                return float_value
            elif float_value == int(float_value):
                return int(float_value)
            else:
                self.fail(
                    f"expected float between [0,1] or int, got {float_value}",
                    param,
                    ctx,
                )
        except TypeError:
            self.fail(
                "expected string for int() or float() conversion, got " f"{value!r} of type {type(value).__name__}",
                param,
                ctx,
            )
        except ValueError:
            self.fail(f"{value!r} is not a valid integer or float", param, ctx)


@click.command(
    context_settings=dict(show_default=True),
)
@click.option(
    "--debug_num",
    type=int,
    default=False,
)
@click.option(
    "--baseline",
    default=False,
)
@click.option(
    "--mode",
    type=click.Choice(
        ["train", "test"],
        case_sensitive=False,
    ),
    default="test",
)
@click.option(
    "--data_path",
    type=click.Path(),
    default="data/",
    help="directory or file",
)
@click.option(
    "--hkg_data_path",
    type=click.Path(),
    default="data_for_relex/",
    help="directory or file",
)
@click.option(
    "--csr_path",
    type=click.Path(),
    default="/100_sym_csr.npz",
    help="csr file",
)
@click.option(
    "--db_path",
    type=click.Path(),
    default="/kgid2subgraph.db",
    help="db file",
)
@click.option(
    "--output_path",
    type=click.Path(),
    default="",
    help="directory to save model",
)
@click.option(
    "--load_path",
    type=click.Path(),
    default="",
    help="directory to load model",
)
@click.option(
    "--encoder_type",
    type=click.Choice(
        ["microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", "transformer_conv"],
        case_sensitive=False,
    ),
    default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    help="encoder architecture to use",
)
@click.option(
    "--model",
    type=click.Choice(
        ["dot", "biaffine"],
        case_sensitive=False,
    ),
    default="biaffine",
    help="score function",
)
@click.option(
    "--multi_label / --multi_class",
    default=True,
    help="multi_label allows multiple labels during inference; multi_class only allow one label",
)
@click.option(
    "--grad_accumulation_steps",
    type=int,
    default=16,
    help="tricks to have larger batch size with limited memory."
    + " The real batch size = train_batch_size * grad_accumulation_steps",
)
@click.option(
    "--max_text_length",
    type=int,
    default=512,
    help="max doc length of BPE tokens",
)
@click.option(
    "--dim",
    type=int,
    default=128,
    help="dimension of last layer feature before the score function "
    + "(e.g., dimension of biaffine layer, dimension of boxes)",
)
@click.option(
    "--bert_learning_rate",
    type=float,
    default=1e-5,
    help="bert_learning rate",
)
@click.option(
    "--weight_decay",
    type=float,
    default=1e-4,
    help="weight decay",
)
@click.option(
    "--dropout_rate",
    type=float,
    default=0.1,
    help="dropout rate",
)
@click.option(
    "--max_grad_norm",
    type=float,
    default=10.0,
    help="gradient norm clip (default 1.0)",
)
@click.option("--epochs", type=int, default=10, help="number of epochs to train")
@click.option(
    "--patience",
    type=int,
    default=5,
    help="patience parameter for early stopping",
)
@click.option(
    "--log_interval",
    type=IntOrPercent(),
    default=1.0,
    help="interval or percentage (as float in [0,1]) of examples to train before logging training metrics "
    "(default: 0, i.e. every batch)",
)
@click.option(
    "--warmup",
    type=float,
    default=-1.0,
    help="number of examples or percentage of training examples for warm up training "
    "(default: -1.0, no warmup, constant learning rate",
)
@click.option(
    "--cuda / --no_cuda",
    default=True,
    help="enable/disable CUDA (eg. no nVidia GPU)",
)
@click.option(
    "--graph_dim",
    default=768,
)
@click.option(
    "--cuda_num",
    default=0,
)
@click.option(
    "--gnn_model",
    default="gcn",
)
@click.option(
    "--max_edge",
    default=None,
)
@click.option(
    "--seed",
    default=random.randint(1, 10000),
)
@click.option(
    "--opt_divide",
    default=False,
)
@click.option(
    "--gnn_learning_rate",
    type=float,
    default=1e-3,
    help="gnn_learning rate",
)
@click.option(
    "--norm_gnn",
    default=False,
)
@click.option(
    "--num_gcn",
    default=2,
)
@click.option(
    "--upper_limit_index",
    default=100,
)
@click.option(
    "--num_hop",
    default=2,
)
@click.option(
    "--kind_of_token",
    type=click.Choice(
        ["text", "text_graph", "cls_text_graph", "only_graph"],
        case_sensitive=False,
    ),
    default="text",
)
@click.option(
    "--encode_type",
    type=click.Choice(
        ["text", "only_graph"],
        case_sensitive=False,
    ),
    default="text",
)
@click.option(
    "--aggregation",
    type=click.Choice(
        ["concat", "max_pool", "average_pool"],
        case_sensitive=False,
    ),
    default="concat",
)

def main(**config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    print("num_hop: " + str(config["num_hop"]))
    print("seed: " + str(config["seed"]))
    print("upper_limit_index: " + str(config["upper_limit_index"]))
    print("num_gcn: " + str(config["num_gcn"]))

    data, model, device, logger = setup(config) 
    trainer = Trainer(data, model, logger, config, device)

    if config["load_path"] != "":
        best_metric_threshold = trainer.load_model()

    if config["mode"] == "train":
        trainer.train()
    else:
        best_metric_threshold = trainer.load_model()
        trainer.model.eval()
        (
            macro_perf,
            micro_perf,
            categ_acc,
            categ_macro_perf,
            na_acc,
            not_na_perf,
            na_perf,
            per_rel_perf,
        ) = trainer.test("test_ctd", best_metric_threshold=best_metric_threshold)
        trainer.performance_logging(
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
        ) = trainer.test("test_anno_ctd", best_metric_threshold=best_metric_threshold)
        trainer.performance_logging(
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
        ) = trainer.test("test_anno_all", best_metric_threshold=best_metric_threshold)
        trainer.performance_logging(
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

    logger.info("gram finished")


if __name__ == "__main__":
    main()
