from collections import Counter, defaultdict
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from evaluate import language_groups
import pandas as pd


def convert_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "{}h {:2.2f}m {:2.2f}s".format(h, m, s)


def get_fine_to_coarse_labels(label_groups):
    coarse_label_mapping = dict()
    for language, lcodes in label_groups.items():
        for lcode in lcodes:
            coarse_label_mapping[lcode] = language
    return coarse_label_mapping


def load_data(filepath, partition, version=4.0):
    """ Load data from file """
    print("\n================= Loading {} =================\n".format(partition))
    label_groups = language_groups(version=version)
    coarse_label_mapping = get_fine_to_coarse_labels(label_groups)
    label_statistics = Counter()
    data = {
        'sentence': [],
        'label': [],
        'coarse_label': []
    }
    with open(filepath, 'r') as f:
        for line in f:
            sentence, label = line.split('\t')
            sentence, label = sentence.strip(), label.strip()
            data['sentence'].append(sentence)
            data['label'].append(label)
            data['coarse_label'].append(coarse_label_mapping[label])
            label_statistics[label] += 1
    data = pd.DataFrame.from_dict(data)
    print("# examples in <{}>: {}".format(partition, data.shape[0]))
    print("# examples per label: {}".format(label_statistics))
    if partition == "train":
        id_to_label = list(label_statistics.keys())
        label_to_id = dict([(k, i) for i, k in enumerate(id_to_label)])
        mappings = dict()
        mappings['label'] = (id_to_label, label_to_id)
        id_to_coarse = label_groups.keys()
        coarse_to_id = dict([(k, i) for i, k in enumerate(id_to_coarse)])
        mappings['coarse_label'] = (id_to_coarse, coarse_to_id)
        mappings['label_groups'] = label_groups
        print("# categories: {}\n{}".format(len(id_to_label), id_to_label))
        return data, mappings
    return data


if __name__ == "__main__":
    data, mappings = load_data("data/DSLCC-v2017/DSL-TRAIN.txt.tok", partition="train")
    # data = load_data("data/DSL-DEV.txt", partition="dev")
    # data = load_data("data/DSL-TEST-GOLD.txt", partition="test")
    print("\n")
    print("Sentence {}".format(data["sentence"][0]))
    print("Label {}".format(data["label"][0]))
