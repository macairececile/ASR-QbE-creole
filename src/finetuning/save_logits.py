# -*- coding: utf-8 -*-
# ----------- Libraries ----------- #
import argparse
import difflib
import itertools
from os import listdir
from os.path import isfile, join
import librosa
import torchaudio
from datasets import load_dataset
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
import torch
import numpy as np
import csv
import json
import preprocessing_creole as preprocessing
from pathlib import Path


# ----------- Functions ----------- #
def load_model(model_path):
    # Call the fine-tuned model
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to("cuda")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    return model, processor, tokenizer


def load_data(file):
    na_test = load_dataset('csv', data_files=[file], delimiter='\t')
    na_test = na_test['train']
    return na_test


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load("/home/getalp/macairec/creole/data/clips/" + batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch


def pipeline(model_dir, test_file):
    def prepare_dataset(batch):
        # check that all files have the correct sampling rate
        assert (
                len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

        with processor.as_target_processor():
            batch["labels"] = processor(batch["target_text"]).input_ids
        return batch

    # load model
    model, processor, tokenizer = load_model(model_dir)
    # load the dataset
    data = load_data(test_file)

    na_test = data.map(preprocessing.remove_special_characters_v3)
    na_test_ref = na_test.map(speech_file_to_array_fn, remove_columns=na_test.column_names)
    na_test_ref = na_test_ref.map(prepare_dataset, remove_columns=na_test_ref.column_names, batch_size=8,
                                  num_proc=4,
                                  batched=True)
    return model, processor, tokenizer, na_test, na_test_ref


def save_logits(arguments):
    # Preprocessing the data
    logits_dir = Path(arguments.model) / "logits_maur/"
    logits_dir.mkdir(exist_ok=True, parents=True)
    model, processor, tokenizer, na_test, na_test_ref = pipeline(arguments.model, arguments.test)
    for i in range(len(na_test_ref)):
        input_dict = processor(na_test_ref["input_values"][i], return_tensors="pt", padding=True, sampling_rate=16000)
        logits = model(input_dict.input_values.to("cuda")).logits
        torch.save(logits, str(logits_dir) + '/logits_' + str(i) + '.pt')


# ----------- Arguments ----------- #
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='sub-command help')

save_pred = subparsers.add_parser("save",
                                  help="Generate predictions from fine-tuned model and store them in csv file.")
save_pred.add_argument('--test', type=str, required=True,
                       help="Test .tsv file.")
save_pred.add_argument('--model', type=str, required=True,
                       help="Directory where the fine-tuned model is stored.")
save_pred.set_defaults(func=save_logits)

arguments = parser.parse_args()
arguments.func(arguments)
