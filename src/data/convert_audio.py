import subprocess
from argparse import ArgumentParser, RawTextHelpFormatter
from os import listdir
from os.path import join, isfile
from pathlib import Path
from pydub import AudioSegment

def get_files_from_directory(path):
    """
    Get all files from directory
    :param path: path where transcripts + wav files are stored
    :return: files
    """
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files


def convert_mono(args):
    wav_dir = Path(args.path) / "wav_convert"
    wav_dir.mkdir(exist_ok=True, parents=True)
    output_path = wav_dir
    files = get_files_from_directory(args.path)
    for i in files:
        sound = AudioSegment.from_wav(args.path+i)
        sound = sound.set_channels(1)
        sound.export(str(output_path)+'/'+i, format="wav")


def convert_wav(args):
    wav_dir = Path(args.path) / "wav_convert"
    wav_dir.mkdir(exist_ok=True, parents=True)
    output_path = str(wav_dir) + '/'
    files = get_files_from_directory(args.path)
    for i in files:
        subprocess.check_output("sox "+args.path+i+" -r 16000 -c 1 -b 16 "+output_path+i, shell=True)



if __name__ == "__main__":
    parser = ArgumentParser(description="Convert audio file in 16kHz", formatter_class=RawTextHelpFormatter)

    parser.add_argument("--path", required=True,
                        help="Path of the audio files to convert.")
    parser.set_defaults(func=convert_wav)
    args = parser.parse_args()
    args.func(args)
