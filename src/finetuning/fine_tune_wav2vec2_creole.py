# -*- coding: utf-8 -*-
from argparse import ArgumentParser, RawTextHelpFormatter

parser = ArgumentParser(description="Fine-tuning a pretrained Wav2Vec2.0 model.", formatter_class=RawTextHelpFormatter)

parser.add_argument('--train_tsv', type=str, required=True,
                   help="Train .tsv file.")
parser.add_argument('--test_tsv', type=str, required=True,
                   help="Test .tsv file.")
parser.add_argument('--output_dir', type=str, required=True,
                   help="Output directory to store the fine-tuned model.")

arguments = parser.parse_args()

# ------------ Libraries ------------ #
from datasets import load_dataset, load_metric
import pandas as pd
from IPython.display import display, HTML
import json
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
import random
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import torchaudio
import librosa
import preprocessing_creole as preprocessing


# ------------ Load dataset ------------ #
train_data = load_dataset('csv', data_files=[arguments.train_tsv], delimiter='\t')
test_data = load_dataset('csv', data_files=[arguments.test_tsv], delimiter='\t')

train_data = train_data['train']
test_data = test_data['train']

# ----------- Preprocessing ----------- #

train_data = train_data.map(preprocessing.remove_special_characters_v3)
test_data = test_data.map(preprocessing.remove_special_characters_v3)

# train_data.to_csv('preprocessed_data_mauritian.csv', index=False, sep='\t')


# ------------ Vocabulary ------------ #
def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


vocab_train = train_data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                             remove_columns=train_data.column_names)
vocab_test = test_data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                           remove_columns=test_data.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open(arguments.output_dir + 'vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

# ------------ Tokenizer, Feature extractor & Processor ------------ #
tokenizer = Wav2Vec2CTCTokenizer(arguments.output_dir + "vocab.json",
                                 unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                             return_attention_mask=True)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

processor.save_pretrained(arguments.output_dir)


# ----------- Prepare dataset ----------- #
def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
            len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch


# print("TARGET TEXT: ", train_data['sentence'][0])


# ------------ Preprocessing dataset audio ------------ #
def speech_file_to_array_fn(batch):
    # change the path to match the directory of audio clips
    speech_array, sampling_rate = torchaudio.load("/home/getalp/macairec/creole/data/clips/"+batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch


def resample(batch):
    if batch["sampling_rate"] != 16000:
        batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 44_100, 16_000)
    batch["sampling_rate"] = 16_000
    return batch


train_data = train_data.map(speech_file_to_array_fn, remove_columns=train_data.column_names)
test_data = test_data.map(speech_file_to_array_fn, remove_columns=test_data.column_names)

# train_data = train_data.map(resample, num_proc=4)
# test_data = test_data.map(resample, num_proc=4)

train_data = train_data.map(prepare_dataset, remove_columns=train_data.column_names, batch_size=8, num_proc=4,
                            batched=True)
test_data = test_data.map(prepare_dataset, remove_columns=test_data.column_names, batch_size=8, num_proc=4,
                          batched=True)


# ------------ Dataclass ------------ #
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# metrics to choose
wer_metric = load_metric("wer")
cer_metric = load_metric("cer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    print("WER: ", wer)
    return {"CER": cer}


# ------------ Definition of the model, training args ------------ #
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", #"LeBenchmark/wav2vec2-FR-7K-large",
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.075,
    layerdrop=0.1,
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

model.config.ctc_zero_infinity = True

model.freeze_feature_extractor()

training_args = TrainingArguments(
    output_dir=arguments.output_dir,
    # output_dir="./wav2vec2-large-xlsr-turkish-demo",
    logging_dir=arguments.output_dir,
    group_by_length=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=60,
    # fp16=True,
    save_steps=100,
    eval_steps=50,
    logging_steps=50,
    learning_rate=3e-4,
    warmup_steps=500,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=processor.feature_extractor,
)

trainer.train()
