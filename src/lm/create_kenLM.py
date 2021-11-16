# ----------- Libraries ----------- #
from argparse import ArgumentParser, RawTextHelpFormatter
import subprocess
from datasets import load_dataset
import re

chars_to_ignore_regex = '[\-\,\?\.\!\;\:\"\“\%\‘\”\\n\-\_\'\…\[\]]'
a_vowels = '[\à\â]'
e_vowels = '[\ë\ê]'
e_vowels_v3 = '[\ë\ê\è\é]'


def remove_special_characters_v3(s):
    """Version (a, e, i, o, u)"""
    s = re.sub(chars_to_ignore_regex, '', s).lower()
    s = re.sub(a_vowels, 'a', s)
    s = re.sub(e_vowels_v3, 'e', s)
    s = re.sub('î', 'i', s)
    s = re.sub('ù', 'u', s)
    s = re.sub('ò', 'o', s)
    s = re.sub('\(\)', '', s)
    return s


def load_data(file):
    dataset = load_dataset('csv', data_files=[file], delimiter='\t')
    dataset = dataset['train']['sentence']
    return dataset


def preprocessing_data_LM(file, output):
    """Get the train text sentences and store it in a .txt file"""
    data = load_data(file)
    data = list(map(remove_special_characters_v3, data))
    with open(output + 'train_kenLM.txt', 'w') as f:
        for el in data:
            f.write(el + '\n')


def create_language_model(arguments):
    """Create 2-, 3-, and 4-gram kenLM language model"""
    preprocessing_data_LM(arguments.data, arguments.output_path)
    for i in range(2, 5):
        command = './kenlm/build/bin/lmplz -o ' + str(
            i) + ' < ' + arguments.output_path + 'train_kenLM.txt > ' + arguments.output_path + 'lm_' + str(i) + '.arpa'
        subprocess.check_output(command, shell=True)


if __name__ == '__main__':
    parser = ArgumentParser(description="Create 2,3,4-kenLM language model.", formatter_class=RawTextHelpFormatter)

    parser.add_argument('--data', type=str, required=True,
                            help="Data file.")
    parser.add_argument('--output_path', type=str, required=True,
                            help="Name of the file to store language model data.")
    parser.set_defaults(func=create_language_model)

    arguments = parser.parse_args()
    arguments.func(arguments)

