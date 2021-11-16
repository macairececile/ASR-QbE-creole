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


def get_duration(args):
    duration = 0
    data = pd.read_csv(args.tsv_file, sep='\t')
    path = data['path']
    for i in path:
        t = librosa.get_duration(filename=args.path_audio + i) / 60
        duration += t
    print("Duration associated to a data file in minutes: ", duration)


if __name__ == "__main__":
    parser = ArgumentParser(description="Length of a data set in seconds.", formatter_class=RawTextHelpFormatter)

    parser.add_argument("--tsv_file", required=True,
                        help="Path of the .tsv data file.")
    parser.add_argument("--path_audio", required=True,
                        help="Path of the audio files.")
    parser.set_defaults(func=get_duration)
    args = parser.parse_args()
    args.func(args)