import re
import statistics
from copy import copy
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, plot_confusion_matrix
import Levenshtein as lev
import statistics as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import operator
import argparse
from datasets import load_metric
import json
from pathlib import Path

wer_metric = load_metric("wer")
cer_metric = load_metric("cer")


def compute_levenshtein_distance(dataframe):
    """Evaluation of edition distance between words per sentence by using the levenshtein distance."""
    dist = []
    lev_all = []
    num_words = []
    dataframe['Ref_words'] = dataframe['Reference'].apply(lambda x: x.split(' '))
    dataframe['Pred_words'] = dataframe['Prediction'].apply(lambda x: x.split(' '))
    for index, row in dataframe.iterrows():
        num_words.append(len(row["Pred_words"]) - len(row["Ref_words"]))
        lev_all.append(lev.ratio(row["Reference"], row["Prediction"]))
        lev_d = []
        for i, j in enumerate(row["Ref_words"]):
            if i < len(row["Pred_words"]):
                lev_d.append(lev.ratio(row["Ref_words"][i], row["Pred_words"][i]))
            else:
                lev_d.append(lev.ratio(row["Ref_words"][i], ''))
        dist.append(lev_d)
    dataframe['Diff_num_words'] = num_words
    dataframe['Lev_distance'] = lev_all
    print('WER_huggingface : ',
          wer_metric.compute(predictions=dataframe["Reference"], references=dataframe["Prediction"]))
    print('CER_huggingface : ',
          cer_metric.compute(predictions=dataframe["Reference"], references=dataframe["Prediction"]))
    print('WER_lev : ', 1 - st.mean([j for a in dist for j in a]))
    print('CER_lev : ', 1 - st.mean(lev_all))
    dataframe['Lev_distance_words'] = dist
    dataframe['Average_lev_dist_words'] = dataframe['Lev_distance_words'].apply(lambda x: st.mean(x))
    return dataframe


def compute_characters(dataframe, out_path):
    f_score = []
    ref_g = []
    hyp_g = []
    ref_words = []
    hyp_words = []
    precision = []
    recall = []
    f = open(out_path + "latex.txt", "w")
    S_insert = 0
    S_delete = 0
    for index, row in dataframe.iterrows():
        edit = lev.editops(row['Prediction'], row['Reference'])
        ind = 0
        ind_bis = 0
        hyp = [i for i in row['Prediction']]
        ref = [i for i in row['Reference']]
        hyp_out = copy(hyp)
        ref_out = copy(ref)
        for i, j in enumerate(edit):
            if edit[i][0] == 'insert':
                if ref_out[edit[i][2]] == ' ':
                    S_insert += 1
                hyp.insert(edit[i][1] + ind_bis, '*')
                ind_bis += 1
                ref_out[edit[i][2]] = '\\hl{' + ref_out[edit[i][2]] + '}'
            elif edit[i][0] == 'delete':
                if hyp_out[edit[i][1]] == ' ': S_delete += 1
                hyp_out[edit[i][1]] = '\\hl{' + hyp_out[edit[i][1]] + '}'
                if len(ref) > edit[i][1] + ind:
                    ref.insert(edit[i][2] + ind, '*')
                    ind += 1
                else:
                    ref.append('*')
            elif edit[i][0] == 'replace':
                hyp_out[edit[i][1]] = '\\hl{' + hyp_out[edit[i][1]] + '}'
                ref_out[edit[i][2]] = '\\hl{' + ref_out[edit[i][2]] + '}'
        ref_out = ''.join(ref_out).replace('\\hl{ }', ' ')
        hyp_out = ''.join(hyp_out).replace('\\hl{ }', ' ')
        f.write('Ref: & ' + ref_out + ' \\\ \n' + 'Hyp: & ' + hyp_out + ' \\\ \n\\midrule \n')
        ref = ['S' if x == ' ' else x for x in ref]
        hyp = ['S' if x == ' ' else x for x in hyp]
        ref_g.append(ref)
        hyp_g.append(hyp)
        ref_words.append(''.join(ref))
        hyp_words.append(''.join(hyp))
        f_score.append(round(f1_score(ref, hyp, average="macro"), 3))
        precision.append(round(precision_score(ref, hyp, average="macro", zero_division=1), 3))
        recall.append(round(recall_score(ref, hyp, average="macro", zero_division=1), 3))
    # print('S_insert: ', S_insert)
    # print('S_delete: ', S_delete)
    f.close()
    dataframe['Ref_char'] = ref_g
    dataframe['Pred_char'] = hyp_g
    dataframe['F_score_char'] = f_score
    dataframe['Precision_char'] = precision
    dataframe['Recall_char'] = recall
    return dataframe, ref_g, hyp_g, ref_words, hyp_words


def compute_words_analysis(ref_w, hyp_w):
    words_error_freq = {}
    ref_cut_by_words = []
    hyp_cut_by_words = []
    for i, m in enumerate(ref_w):
        for j, k in enumerate(m):
            if hyp_w[i][j] == '*' and k == 'S':
                hyp_w[i] = hyp_w[i][:j] + 'S' + hyp_w[i][j + 1:]
            elif hyp_w[i][j] == 'S' and k == '*':
                ref_w[i] = ref_w[i][:j] + 'S' + ref_w[i][j + 1:]
    for e, r in enumerate(ref_w):
        id_S = [match.start() for match in re.finditer('S', r)]
        id_S += [match.start() for match in re.finditer('S', hyp_w[e])]
        id_S = sorted(id_S)
        for g, h in enumerate(id_S):
            if g == 0:
                ref_cut_by_words.append(r[0:h])
                hyp_cut_by_words.append(hyp_w[e][0:h])
            elif g > 0 and g < len(id_S) - 1:
                ref_cut_by_words.append(r[h + 1:id_S[g + 1]])
                hyp_cut_by_words.append(hyp_w[e][h + 1:id_S[g + 1]])
            else:
                ref_cut_by_words.append(r[h + 1:len(r)])
                hyp_cut_by_words.append(hyp_w[e][h + 1:len(r)])
    ref_cut_by_words = list(filter(None, ref_cut_by_words))
    hyp_cut_by_words = list(filter(None, hyp_cut_by_words))
    for id, el in enumerate(ref_cut_by_words):
        if el not in words_error_freq.keys():
            if hyp_cut_by_words[id] != el:
                words_error_freq[el] = [1, 1]
            else:
                words_error_freq[el] = [0, 1]
        else:
            if hyp_cut_by_words[id] != el:
                words_error_freq[el][0] += 1
                words_error_freq[el][1] += 1
            else:
                words_error_freq[el][1] += 1
    for k, v in words_error_freq.items():
        if v[0] == 0:
            words_error_freq[k] = 0
        else:
            words_error_freq[k] = v[0] / v[1] * 100
    words_error_freq = {k: v for k, v in sorted(words_error_freq.items(), key=lambda item: item[1])}
    df = pd.DataFrame.from_dict(words_error_freq, orient="index")
    df.to_csv("error_rate_per_words.csv", sep='\t')



def confusion_matrix_phoneme(ref, pred, phoneme, out_path):
    ref_all = [x for y in ref for x in y]  # découpage en charactères
    hyp_all = [x for y in pred for x in y]
    select_ref = []
    select_hyp = []
    for i, j in enumerate(ref_all):
        if j == phoneme:
            select_ref.append(ref_all[i])
            select_hyp.append(hyp_all[i])
    plt.rcParams.update({'font.size': 25, 'xtick.labelsize': 'small', 'ytick.labelsize': 'small'})
    labels = list(set(select_ref) | set(select_hyp))
    cm = confusion_matrix(select_ref, select_hyp, labels=labels)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(cm, annot=True, ax=ax, cmap="OrRd", fmt="d")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='left', minor=False)
    ax.set_yticklabels(labels, rotation=0)
    plt.xlabel('Predicted labels', labelpad=20)
    plt.ylabel('Referenced labels', labelpad=20)
    plt.tight_layout()
    plt.show()

    conf_matrix = pd.crosstab(np.array(ref_all), np.array(hyp_all), rownames=['Reference'], colnames=['Hypothesis'],
                              margins=True)
    results = {}
    d = conf_matrix.to_dict(orient='records')
    ind = list(conf_matrix.index.values)
    for i, j in enumerate(ind):
        if j != 'All':
            del d[i]['All']
            if j in d[i].keys():
                del d[i][j]
            results[j] = d[i]
    best_wrong_associations = {}
    for k, v in results.items():
        best_wrong_associations[k] = max(v.items(), key=operator.itemgetter(1))[0]
    with open(out_path + 'best_wrong_predictions.txt', 'w') as file:
        file.write(json.dumps(best_wrong_associations))


def process_csv(path):
    df = pd.read_csv(path, sep='\t')
    df['Reference'] = df['Reference'].apply(lambda x: x.replace('|', ' '))
    df['Reference'] = df['Reference'].apply(lambda x: x.rstrip())
    df['Prediction'] = df['Prediction'].apply(lambda x: x.replace('[UNK]', '*'))
    df['Prediction'] = df['Prediction'].apply(lambda x: x.replace('…', ''))
    return df


def eval_lev(args):
    out_path = Path('./analysis/')
    out_path.mkdir(exist_ok=True, parents=True)
    out_path = out_path.__str__() + '/'
    data = process_csv(args.input_file)
    results = compute_levenshtein_distance(data)
    results.to_csv(out_path + 'results_analysis_lev_dist.csv', sep='\t', index=False)


def eval_char(args):
    out_path = Path('./analysis/')
    out_path.mkdir(exist_ok=True, parents=True)
    out_path = out_path.__str__() + '/'
    data = process_csv(args.input_file)
    data2, refs, preds, ref_w, hyp_w = compute_characters(data, out_path)
    confusion_matrix_phoneme(refs, preds, args.phone, out_path)
    # data2.to_csv(out_path + 'results_analysis_char.csv', sep='\t', index=False)


def eval_words(input_file):
    out_path = Path('./analysis/')
    out_path.mkdir(exist_ok=True, parents=True)
    out_path = out_path.__str__() + '/'
    data = process_csv(input_file)
    data2, refs, preds, ref_w, hyp_w = compute_characters(data, out_path)
    compute_words_analysis(ref_w, hyp_w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    lev_dist = subparsers.add_parser("lev_dist",
                                     help="Compute the levenshtein distance between each reference and its corresponding prediction (all sentence and sentence splitted by words")
    lev_dist.add_argument('--input_file', type=str, required=True,
                          help="CSV result file with Reference and Prediction columns.")
    lev_dist.set_defaults(func=eval_lev)

    eval_phon = subparsers.add_parser("eval_char",
                                      help="Analysis of character similarities between references and predictions.")
    eval_phon.add_argument('--input_file', type=str, required=True,
                           help="CSV result file with Reference and Prediction columns.")
    eval_phon.add_argument('--phone', type=str, required=False,
                           help="Phoneme to print the confusion matrix.")
    eval_phon.set_defaults(func=eval_char)

    eval_w = subparsers.add_parser("eval_words",
                                      help="Analysis of character similarities between references and predictions.")
    eval_w.add_argument('--input_file', type=str, required=True,
                           help="CSV result file with Reference and Prediction columns.")
    eval_w.set_defaults(func=eval_words)

    args = parser.parse_args()
    args.func(args)