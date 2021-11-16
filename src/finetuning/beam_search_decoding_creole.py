# -*- coding: utf-8 -*-
# ----------- Libraries ----------- #
from argparse import ArgumentParser, RawTextHelpFormatter
import difflib
from datasets import load_dataset, load_metric
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
import torch
import numpy as np
import json
from ctcdecode import CTCBeamDecoder
import operator
import re
import torchaudio
import librosa
import preprocessing_creole as prep

cer_metric = load_metric('cer')
wer_metric = load_metric('wer')


# ----------- Load the data, the model, tokenizer, processor, process the data ----------- #
def load_model(model_path):
    # Call the fine-tuned model
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    return processor, tokenizer


def load_data(file):
    na_test = load_dataset('csv', data_files=[file], delimiter='\t')
    na_test = na_test['train']
    return na_test


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(
        "/home/getalp/macairec/Bureau/Creole/guadeloupean/clips/" + batch["path"])
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
    processor, tokenizer = load_model(model_dir)
    # load the dataset
    data = load_data(test_file)

    na_test = data.map(prep.remove_special_characters_v3)
    na_test_ref = na_test.map(speech_file_to_array_fn, remove_columns=na_test.column_names)
    na_test_ref = na_test_ref.map(prepare_dataset, remove_columns=na_test_ref.column_names, batch_size=8,
                                  num_proc=4,
                                  batched=True)
    return processor, tokenizer, na_test, na_test_ref


# ----------- Beam search decoding ----------- #
# Decoding with https://github.com/parlance/ctcdecode library
def beam_search_decoder_lm(processor, tokenizer, logits, lm, alpha, beta):
    vocab = tokenizer.convert_ids_to_tokens(range(0, processor.tokenizer.vocab_size))
    space_ix = vocab.index('|')
    vocab[space_ix] = ' '

    ctcdecoder = CTCBeamDecoder(vocab,
                                model_path=lm,
                                alpha=alpha,
                                beta=beta,
                                cutoff_top_n=300,
                                cutoff_prob=1.0,
                                beam_width=100,
                                num_processes=4,
                                blank_id=processor.tokenizer.pad_token_id,
                                log_probs_input=True
                                )

    beam_results, beam_scores, timesteps, out_lens = ctcdecoder.decode(logits)

    # beam_results - Shape: BATCHSIZE x N_BEAMS X N_TIMESTEPS A batch containing the series
    # of characters (these are ints, you still need to decode them back to your text) representing
    # results from a given beam search. Note that the beams are almost always shorter than the
    # total number of timesteps, and the additional data is non-sensical, so to see the top beam
    # (as int labels) from the first item in the batch, you need to run beam_results[0][0][:out_len[0][0]].
    beam_string = "".join(vocab[n] for n in beam_results[0][0][:out_lens[0][0]])

    # timesteps : BATCHSIZE x N_BEAMS : the timestep at which the nth output character has peak probability.
    # Can be used as alignment between the audio and the transcript.
    alignment = list()
    for i in range(0, out_lens[0][0]):
        alignment.append([beam_string[i], int(timesteps[0][0][i])])
    return beam_string


def decoding_lm(args):
    processor, tokenizer, na_test, na_test_ref = pipeline(args.model, args.test)
    preds_lm = []
    refs = []
    for i in range(len(na_test)):
        # load the saved logits to generate the prediction of the model
        logits = torch.load(args.model + '/logits/logits_' + str(i) + '.pt', map_location=torch.device('cpu'))
        preds_lm.append(
            beam_search_decoder_lm(processor, tokenizer, logits, args.lm, args.alpha, args.beta))
        refs.append(na_test['sentence'][i])
        # ------ save all LM predictions in a csv file ------ #
    df_lm = pd.DataFrame({'Reference': refs, 'Prediction': preds_lm})
    df_lm.to_csv(args.model + 'results_decode_lm.csv', index=False, sep='\t')


# ----------- Arguments ----------- #
parser = ArgumentParser(description="Generate predictions with a kenLM language model from a Wav2Vec2.0 model and save them.", formatter_class=RawTextHelpFormatter)

parser.add_argument('--test', type=str, required=True,
                               help="Test .tsv file.")
parser.add_argument('--model', type=str, required=True,
                               help="Directory where the fine-tuned model is stored.")
parser.add_argument('--lm', type=str, required=True,
                               help="Word Ken language model.")
parser.add_argument('--alpha', type=int, required=True,
                               help="alpha lm parameter.")
parser.add_argument('--beta', type=int, required=True,
                               help="beta lm parameter.")

parser.set_defaults(func=decoding_lm)
args = parser.parse_args()
args.func(args)
