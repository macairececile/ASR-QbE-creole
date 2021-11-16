# -*- coding: utf-8 -*-
# ----------- Libraries ----------- #
from argparse import ArgumentParser, RawTextHelpFormatter
from os import listdir
from os.path import isfile, join
import librosa
import torchaudio
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
import torch
import numpy as np
import re
import itertools
import swalign
from collections import OrderedDict, defaultdict
import operator
from sklearn.metrics import precision_score, recall_score
import io
import sys
from contextlib import redirect_stdout
from io import StringIO

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\\n\-\_\'\…]'
a_vowels = '[\à\â]'
e_vowels = '[\ë\ê]'
e_vowels_v3 = '[\ë\ê\è\é]'


# ----------- Functions ----------- #
def get_files_from_directory(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files


def preprocessing(sentence):
    """Version (a, e, i, o, u)"""
    sentence = re.sub(chars_to_ignore_regex, '', sentence).lower()
    sentence = re.sub(a_vowels, 'a', sentence)
    sentence = re.sub(e_vowels_v3, 'e', sentence)
    sentence = re.sub('î', 'i', sentence)
    sentence = re.sub('ù', 'u', sentence)
    sentence = re.sub('ò', 'o', sentence)
    return sentence


def get_data(transcript):
    data = pd.read_csv(transcript, sep='\t')
    data['sentence'] = data['sentence'].apply(preprocessing)
    return data.to_dict(orient='records')


def check_double(word, top_align, match=2, mismatch=-1):
    scoring = swalign.NucleotideScoringMatrix(match, mismatch)
    sw = swalign.LocalAlignment(scoring)
    best_score = top_align[0][2]
    for i, j in enumerate(top_align):
        print("For the "+str(i)+' eme ref: ')
        sentence = j[1]
        s = sentence[:j[3].q_pos] + sentence[j[3].q_end:]
        alignment = sw.align(word, s)
        score = alignment.score
        while score == best_score:
            alignment.dump()
            s = s[:alignment.q_pos] + s[alignment.q_end:]
            alignment = sw.align(word, s)
            score = alignment.score
        else:
            print("Nothing else for this sequence.")


def keyword_spotting(word, trans, match=2, mismatch=-1):
    scoring = swalign.NucleotideScoringMatrix(match, mismatch)
    sw = swalign.LocalAlignment(scoring)

    scores = []
    refs = []
    queries = []

    for i in trans:
        alignment = sw.align(word, i['sentence'])
        score_per_sentence = [i['path'], i['sentence'], alignment.score, alignment]
        scores.append(score_per_sentence)
    sort_alignments = sorted(scores, key=lambda tup: tup[2], reverse=True)
    top_alignments = [i for i in sort_alignments if i[2] == sort_alignments[0][2]]

    for align in top_alignments:
        refs.append(align[3].orig_ref)
        queries.append(align[3].orig_query[align[3].q_pos:align[3].q_end])
        align[3].dump()
        output = '\nPath of the file: ' + align[0] + '\nComplete sentence: ' + align[
            1] + '\n--------------------------\n'
        print(output)

    check_double(word, top_alignments)

    print("----------------------------\n----------------------------\n")


def keyword_spotting_wav2vec2(args):
    # Preprocessing the data
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to("cuda")
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    files = get_files_from_directory(args.sound)
    for f in files:
        speech_array, sampling_rate = torchaudio.load(args.sound+f)
        speech = speech_array[0].numpy()
        input_dict = processor(speech, return_tensors="pt", padding=True, sampling_rate=16000)
        logits = model(input_dict.input_values.to("cuda")).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        prediction = ' '+processor.decode(pred_ids)+' '
        print("Prediction of the model: ", prediction)

        transcripts = get_data(args.transcript)
        keyword_spotting(prediction, transcripts)


# ----------- Arguments ----------- #
parser = ArgumentParser(description="Keyword spotting with Wav2Vec2.0.", formatter_class=RawTextHelpFormatter)

parser.add_argument("--sound", required=True,
                        help="Wav file for keyword spotting.")
parser.add_argument("--model", required=True,
                        help="Directory where the fine-tuned model is stored.")
parser.add_argument("--transcript", required=True,
                        help="File where the transcriptions of the corpus are stored.")
parser.set_defaults(func=keyword_spotting_wav2vec2)
args = parser.parse_args()
args.func(args)
