import os
import xml.etree.ElementTree as et
from os import listdir
from os.path import join, isfile
import sox
from pathlib import Path
import csv
import pandas as pd
import argparse
from pydub import AudioSegment
import librosa
from sklearn.utils import shuffle
import textgrids as tg
import shutil
import glob
import json
import statistics
import codecs


def get_files_from_directory(path):
    """
    Get all files from directory
    :param path: path where transcripts + wav files are stored
    :return: files
    """
    files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".wav")]
    return files


def stats(args):
    df = pd.read_csv(args.path+'all.tsv', sep='\t')
    time_secs = []
    duration_per_words = {}
    for index, row in df.iterrows():
        duration = librosa.get_duration(filename=args.path + 'all_wavs/' + row["path"])
        time_secs.append(duration)
        if row["sentence"] not in duration_per_words.keys():
            duration_per_words[row['sentence']] = [duration]
        else:
            duration_per_words[row['sentence']].append(duration)
    for k,v in duration_per_words.items():
        duration_per_words[k] = statistics.mean(v)
    # with open(args.path + 'duration_per_words_creole.txt', 'w') as file:
    #     file.write(json.dumps(duration_per_words))
    print(duration_per_words)
    df['time'] = time_secs
    print("Mean duration of audio files in the corpus: ", df['time'].mean())
    print(df.describe())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    stats_words = subparsers.add_parser("stats",
                                  help="Stats")
    stats_words.add_argument('--path', type=str, required=True,
                       help="Path of the corpus with mp3 files.")
    stats_words.set_defaults(func=stats)
    args = parser.parse_args()
    args.func(args)
