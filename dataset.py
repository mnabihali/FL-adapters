from flwr_datasets import FederatedDataset
import argparse
import warnings
from collections import OrderedDict
from datasets import load_dataset
import flwr as fl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

warnings.filterwarnings("ignore", category=UserWarning)


def dataset_load(node_id):
    vocab_json = None

    fds_train = FederatedDataset(dataset='librispeech_asr', partitioners={"train": 500})
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

    def remove_special_characters(batch):
        batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
        return batch

    def extract_all_chars(batch):
        all_text = " ".join(batch["text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    train_data = fds_train.map(remove_special_characters)

    vocabs_train = train_data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                                  remove_columns=librispeech_train.column_names)
    vocab_list = list(set(vocabs_train["vocab"][0]))

    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open('vocab_librispeech.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    vocab_json = 'vocab_librispeech.json'

    tokenizer = Wav2Vec2CTCTokenizer(vocab_json, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # prepare dataset
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids
            return batch

    train_set = train_data.map(prepare_dataset, remove_columns=train_set.column_names)

    partition = train_set.load_partition(node_id)

    return partition


class DataCollatorCTCWithPadding:

    def __init__(self,
                 processor: Wav2Vec2Processor,
                 padding: Union[bool, str] = True,
                 max_length: Optional[int] = None,
                 max_length_labels: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None,
                 pad_to_multiple_of_labels: Optional[int] = None):
        self.processor = processor
        self.padding = padding
        self.max_length = max_length
        self.max_length_labels = max_length_labels
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_to_multiple_of_labels = pad_to_multiple_of_labels

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            # padding=self.padding,
            padding='max_length',
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            truncation=True
        )

        assert batch["input_values"].size(1) == 200000

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--node-id",
        choices=list(range(1_000)),
        required=True,
        type=int,
        default=20,
        help="Partition of the dataset divided into 1,000 iid partitions created "
             "artificially.",
    )
    node_id = parser.parse_args().node_id
    dataset_load(node_id)

'''
def load_data(node_id):
    """Load IMDB data (training and eval)"""
    fds = FederatedDataset(dataset="imdb", partitioners={"train": 1_000})
    partition = fds.load_partition(node_id)
    # Divide data: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    partition_train_test = partition_train_test.map(tokenize_function, batched=True)
    partition_train_test = partition_train_test.remove_columns("text")
    partition_train_test = partition_train_test.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        partition_train_test["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        partition_train_test["test"], batch_size=32, collate_fn=data_collator
    )

    return trainloader, testloader



#fds = FederatedDataset(dataset="cifar10", partitioners={"train": 100})
#print(fds)
#partition = fds.load_partition(0, "train")
#partition2 = fds.load_partition(1, "train")
#partition9 = fds.load_partition(9, "train")
#print(partition9)
'''
