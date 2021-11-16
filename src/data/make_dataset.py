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
import pympi


def get_files_from_directory(path):
    """
    Get all files from directory
    :param path: path where transcripts + wav files are stored
    :return: files
    """
    files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".wav")]
    return files


def convert_mp3towav(args):
    input = args.path_input
    output = args.path_output
    files = get_files_from_directory(input)
    for f in files:
        sound = AudioSegment.from_mp3(input + f)
        sound.export(output + f[:-4] + '.WAV', format="wav")


def extract_information(tsf_file):
    """
    Extract the transcription timecodes and ID from an xml file at the sentence level
    :param tsf_file: transcription file
    :return: information (start audio, end audio, transcription)
    """
    information = {}
    tree = et.parse(tsf_file)
    root = tree.getroot()  # lexical resources
    sections = root.find('Episode').findall('Section')
    for h,d in enumerate(sections):
        if d.attrib['type'] == 'report':
            endTime = d.attrib.get('endTime')
            turn = d.findall('Turn')
            for k, l in enumerate(turn):
                endTime2 = l.attrib.get('endTime')
                syncs = l.findall('Sync')
                for i, j in enumerate(syncs):
                    id = str(h) + str(k) + str(i)
                    if i < len(syncs) - 1:
                        info = [j.attrib.get('time'), syncs[i + 1].attrib.get('time'), id]
                    elif k == len(turn) - 1 and i == len(syncs) - 1:
                        info = [j.attrib.get('time'), endTime, id]
                    else:
                        info = [j.attrib.get('time'), endTime2, id]
                    information[j.tail] = info
    return information


def extract_words(textgrid):
    information = {}
    grid = tg.TextGrid(textgrid)  # extract TextGrid
    for i, j in enumerate(grid['words']):  # extract timecodes
        if j.text != "_":
            info = [j.xmin, j.xmax, i]
            information[j.text] = info
    return information


def create_data_words_yazid(args):
    files = get_files_from_directory(args.path)

    tsv = open(args.path + 'all.tsv', 'wt')
    tsv_writer = csv.writer(tsv, delimiter='\t')
    tsv_writer.writerow(['path', 'sentence'])

    for f in files:
        word = f.split('_')[2]
        if word != '':
            tsv_writer.writerow([f, word])


def extract_information_eaf(eaf_file):
    information = {}
    eaf = pympi.Elan.Eaf(eaf_file)
    tier_name = list(eaf.get_tier_names())[0]
    for i, j in enumerate(eaf.get_annotation_data_for_tier(tier_name)):
        information[j[2]] = [j[0] / 1000, j[1] / 1000, i]
    return information


def create_audio_tsv(args):
    """
    Create audios at the sentence level and create a tsv file which links each new audio file with the corresponding sentence
    :param args: path
    """
    files_process = []
    path = args.path
    files = get_files_from_directory(path)

    tsv = open(path + 'all.tsv', 'wt')
    tsv_writer = csv.writer(tsv, delimiter='\t')
    tsv_writer.writerow(['path', 'sentence'])

    for f in files:
        name_xml = f[:-4] + '.trs'
        # name_eaf = f[:-4] + '.eaf'
        # name_textgrid = f[:-4] + '.TextGrid'
        try:
            info = extract_information(path + name_xml)
            # info = extract_words(path + name_textgrid)
            # info = extract_information_eaf(path + name_eaf)
            wav_dir = Path(path) / "clips/"
            wav_dir.mkdir(exist_ok=True, parents=True)
            files_process.append(f)
            wav_file = path + f

            for k, v in info.items():
                tfm = sox.Transformer()
                tfm.trim(float(v[0]), float(v[1]))
                tfm.compand()
                output_wav = f"{f[:-4]}_{v[2]}.wav"  # fstring
                tfm.build_file(wav_file, str(wav_dir) + '/' + output_wav)
                tsv_writer.writerow([output_wav, k])
        except:
            print('No xml file in this format: ', name_xml)
            # print('No xml file in this format: ', name_textgrid)
    tsv.close()


def create_dataset(args):
    """
    Create train/val/test tsv files (ratio 80 / 10 / 10)
    :param args: path
    """
    path = args.path

    corpus = pd.read_csv(path + 'all.tsv', sep='\t')
    corpus = shuffle(corpus)

    size_corpus = corpus.shape[0]

    split = [int(size_corpus * 0.9), int(size_corpus * 0.1)]

    train = corpus.iloc[:split[0]]
    # val = corpus.iloc[split[0]:split[0] + split[1]]
    test = corpus.iloc[split[0]:split[0] + split[1]]

    train.to_csv(path + 'train.tsv', index=False, sep='\t')
    # val.to_csv(path + 'valid.tsv', index=False, sep='\t')
    test.to_csv(path + 'test.tsv', index=False, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    convert = subparsers.add_parser("convert",
                                    help="Convert mp3 audio to wav format.")
    convert.add_argument('--path_input', type=str, required=True,
                         help="Path of the corpus with mp3 files.")
    convert.add_argument('--path_output', type=str, required=True,
                         help="Path to store the converted audio files.")
    convert.set_defaults(func=convert_mp3towav)

    create_audio = subparsers.add_parser("create_audio",
                                         help="Create audio per sentences from xml and wav and store the info in a tsv file. "
                                              "Make sure the transcriptions are in /trans and wav in /wav")
    create_audio.add_argument('--path', type=str, required=True,
                              help="path of the corpus with wav and transcription files.")
    create_audio.set_defaults(func=create_audio_tsv)

    create_words = subparsers.add_parser("create_words",
                                         help="Create audio per sentences from xml and wav and store the info in a tsv file. "
                                              "Make sure the transcriptions are in /trans and wav in /wav")
    create_words.add_argument('--path', type=str, required=True,
                              help="path of the corpus with wav and transcription files.")
    create_words.set_defaults(func=create_data_words_yazid)

    split_dataset = subparsers.add_parser("create_dataset",
                                          help="Create dataset - train/val/test tsv files.")
    split_dataset.add_argument('--path', required=True, help="path of the corpus with wav and transcription files")
    split_dataset.set_defaults(func=create_dataset)

    args = parser.parse_args()
    args.func(args)
