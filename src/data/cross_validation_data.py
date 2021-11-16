from os import listdir
from os.path import join, isfile
import pandas as pd
import librosa
from argparse import ArgumentParser, RawTextHelpFormatter


def make_split(args):
    data = pd.read_csv(args.tsv_file, sep='\t')
    path = data['path']
    t = 0
    el = 0
    for i, j in enumerate(path[el:]):
        if t < args.test_length:
            t += librosa.get_duration(filename=args.clips + j) / 60
        else:
            data[el:el+i].to_csv(args.output_name+'_test.csv', sep='\t', index=False)
            train = data[0:el].append(data[el+i:len(data)])
            train.to_csv(args.output_name+'_train.csv', sep='\t', index=False)
            print("Num stop element: ", el+i)
            break


if __name__ == '__main__':
    parser = ArgumentParser(description="Create cross-validation data files.", formatter_class=RawTextHelpFormatter)

    parser.add_argument('--tsv_file', type=str, required=True,
                            help="Data file.")
    parser.add_argument('--clips', type=str, required=True,
                            help="Directory of the audio clips.")
    parser.add_argument('--test_length', type=int, required=True, help="Duration in seconds of files in the test set.")
    parser.add_argument('--output_name', type=str, required=True, help="Name of the split file.")
    parser.set_defaults(func=make_split)

    args = parser.parse_args()
    args.func(args)