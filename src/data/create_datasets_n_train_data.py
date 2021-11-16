from os import listdir
from os.path import join, isfile
import pandas as pd
import librosa
from argparse import ArgumentParser, RawTextHelpFormatter


def get_files_from_directory(path):
    """
    Get all files from directory
    :param path: path where transcripts + wav files are stored
    :return: files
    """
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files


def make_train_split(args):
    for n in range(0, 150, args.time):
        data = pd.read_csv(args.tsv_file, sep='\t')
        path = data['path']
        t = 0
        for i, j in enumerate(args.path):
            if t < time:
                t += librosa.get_duration(filename=args.path + j) / 60
            else:
                data[:i].to_csv('train_' + str(args.time) + '.csv', sep='\t', index=False)
                break


parser = ArgumentParser(description="Create datasets with n train data.", formatter_class=RawTextHelpFormatter)

parser.add_argument("--tsv_file", required=True,
                        help="Path of the .tsv data file.")
parser.add_argument("--path", required=True,
                        help="Path of the audio files.")
parser.add_argument("--time", required=True,
                        help="Time length")
parser.set_defaults(func=get_duration)
args = parser.parse_args()
args.func(args)
