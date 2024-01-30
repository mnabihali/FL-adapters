import math
import utils
import re
import json
import flwr as fl
import statistics
from collections import OrderedDict
from dataclasses import field, dataclass
from typing import *
from transformers import (set_seed, Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2Config,
                          TrainingArguments, EarlyStoppingCallback, HfArgumentParser)
from transformers.integrations import TensorBoardCallback
from torch.optim.lr_scheduler import LambdaLR
from os.path import join
from dataset import dataset_load
from transformers import Trainer
from modeling_wav2vec2 import Wav2Vec2ForCTC
from data import DataCollatorCTCWithPadding
from datasets import load_metric
from path import Path
import sys

folder = Path(__file__).abspath()
sys.path.append(folder.parent.parent.parent)


@dataclass
class DataTrainingArguments(TrainingArguments):
    node_id: Optional[int] = field(
        default=500, metadata={"help": "partitions to be split"}
    )
    dataset: Optional[str] = field(
        default="esd", metadata={"help": "dataset name"}
    )
    data_dir: Optional[str] = field(
        default="/data/path/ESD/en/", metadata={"help": "The dir of the dataset."}
    )
    feat_adapter_name: Optional[str] = field(
        default="conv_adapter", metadata={"help": "The type of adapter, should be chosen among in {conv_adapter }."}
    )
    trans_adapter_name: Optional[str] = field(
        default="bottleneck",
        metadata={"help": "The type of adapter, should be chosen among in {conv_adapter, bottleneck, adapterblock}."}
    )
    output_adapter: Optional[bool] = field(
        default=False, metadata={"help": "use adapter after FFN"}
    )
    mh_adapter: Optional[bool] = field(
        default=False, metadata={"help": "use adapter after multi-head attention"}
    )
    prefix_tuning: Optional[bool] = field(
        default=False, metadata={"help": "use prefix-tuning in multi-head attention, implemented by us"}
    )
    prefix_seq_len: Optional[int] = field(
        default=30, metadata={"help": "prefix sequence length"}
    )
    prefix_projection: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Apply a two-layer MLP head over the prefix embeddings"
        }
    )
    prefix_dropout_prob: Optional[bool] = field(
        default=0.1,
        metadata={
            "help": "The dropout probability used in the models"
        }
    )
    feat_enc_adapter: Optional[bool] = field(
        default=False, metadata={"help": "use conv_adapter in feature encoder and Adapterblock in  "}
    )
    lora_adapter: Optional[bool] = field(
        default=False, metadata={"help": "use lora_adapter in feature encoder"}
    )
    fine_tune: Optional[bool] = field(
        default=False, metadata={"help": "if fine-tune wav2vec2 or not"}
    )


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_constant_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif num_warmup_steps <= current_step < num_constant_steps:
            return float(1.0)
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_constant_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant_ratio = 0.4
        self.num_constant_steps = -1

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_constant_steps=self.get_keep_constant_steps(num_training_steps),
                num_training_steps=num_training_steps)
        return self.lr_scheduler

    def get_keep_constant_steps(self, num_training_steps: int):
        keep_constant_steps = (
            self.num_constant_steps if self.num_constant_steps > 0 else math.ceil(
                num_training_steps * (self.constant_ratio + self.args.warmup_ratio))
        )
        return keep_constant_steps


def main(node_id):
    set_seed(1314)
    parser = HfArgumentParser(DataTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    config = Wav2Vec2Config.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english",
                                            vocab_size=len(processor.tokenizer))
    config._name_or_path = ""

    config.adapter_name = args.trans_adapter_name
    config.output_adapter = args.output_adapter
    config.mh_adapter = args.mh_adapter
    config.prefix_tuning = args.prefix_tuning
    config.feat_enc_adapter = args.feat_enc_adapter
    config.lora_adapter = args.lora_adapter
    config.prefix_seq_len = args.prefix_seq_len
    config.prefix_projection = args.prefix_projection
    config.prefix_dropout_prob = args.prefix_dropout_prob
    config.ctc_loss_reduction = "mean"
    config.pad_token_id = processor.tokenizer.pad_token_id

    # load pretrained model
    net = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", config=config,
                                         ignore_mismatched_sizes=True)
    net.freeze_feature_encoder()

    print("------>>> Trainable params(before freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))
    if not args.fine_tune:
        model.freeze_exclude_prompt()
    print("------>>> Trainable params(after  freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True, max_length=200000)
    wer_metric = load_metric("wer")

    partition = dataset_load(node_id)

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    trainer = CustomTrainer(
        model=model,
        data_collator=data_collator,
        args=args,
        compute_metrics=compute_metrics,
        train_dataset=partition,
        eval_dataset=valid_set,
        tokenizer=processor.feature_extractor,
        callbacks=[TensorBoardCallback],
    )

    class ASRCLIENT(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            print("Training Started...")
            if args.fine_tune:
                trainer.train()
            return self.get_parameters(config={}), len(train_dataset), {}

    fl.client.start_client(server_address="127.0.0.1:8080", client=ASRCLIENT().to_client())
