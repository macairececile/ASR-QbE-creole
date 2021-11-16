# Automatic Speech Recognition and Query By Example for Creole Languages Documentation

This repository contains the code and results to reproduce the experiments conducted in the article entitled "Automatic Speech Recognition and Query By Example for Creole Languages Documentation" written by Cécile Macaire, Didier Schwab, Benjamin Lecouteux, and Emmanuel Schang.

## Content

The repository is organized as follow:
* **/src**, containing the source code.
* **/results**, including the results for fine-tuning and the query-by-example experiments.

## Getting started

## Experiments

### Fine-tuning Wav2Vec2.0 on Creole languages

### Query-by-example

The query-by-example is based on the swalign python library. From the prediction of a fine-tuned model, the query-by-example algorithm will give the top alignments, with the path of the file and the complete sentence where the keyword is spotted.

```bash
python --sound path_of_the_audio_segments --model path_of_the_fine_tuned_model --transcript transcription_file
```

## Acknowledgments
