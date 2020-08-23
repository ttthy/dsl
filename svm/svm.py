from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from evaluate import language_groups, breakdown_evaluation
import numpy as np
import time
import svm.utils as utils
import argparse


def build_model(data, label_column='label', model_class=LinearSVC):
    vectorizer = TfidfVectorizer(ngram_range=(2, 5), analyzer="char_wb")
    model = model_class()
    start = time.time()
    feat_vectors = vectorizer.fit_transform(data["sentence"])
    model.fit(feat_vectors, data[label_column])
    print("Fitting time: {}".format(utils.convert_time(time.time()-start)))
    return vectorizer, model


def get_data_subset(data, coarse_label):
    return data.loc[data["coarse_label"] == coarse_label]


def predict(vectorizer, model, data):
    start = time.time()
    feat_vectors = vectorizer.transform(data["sentence"])
    pred = model.predict(feat_vectors)
    print("Predict time : {}".format(utils.convert_time(time.time()-start)))
    return pred


def predict_per_group(coarse_label, data, model_groups):
    data_subset = get_data_subset(data, coarse_label)
    vectorizer, model = model_groups[coarse_label]
    pred = predict(vectorizer, model, data_subset)
    return pred


def print_scores(gold, pred):
    start = time.time()
    f_macro = f1_score(gold, pred, average="macro")
    acc = accuracy_score(gold, pred)
    breakdown_evaluation(pred, gold)
    print('F1 macro = {:.3f}, Acc = {:.3f}'.format(f_macro, acc))
    print()
    print("Predict time : {}".format(utils.convert_time(time.time()-start)))


def run_charngram(train_data, dev_data, test_data):
    print("\n================= CharNGram SVC =================")
    charngram = build_model(train_data)
    print("\n====== Dev")
    dev_pred = predict(charngram[0], charngram[1], dev_data)
    print_scores(dev_data["label"], dev_pred)
    print("\n==== Test")
    test_pred = predict(charngram[0], charngram[1], test_data)
    print_scores(test_data["label"], test_pred)


def run_coarse_to_fine_models(train_data, dev_data, test_data):
    print("\n================= 2-step SVC =================")
    print("\n======= Building coarse SVC =======")
    coarse_charngram = build_model(train_data, label_column="coarse_label")
    coarse_dev_pred = predict(coarse_charngram[0], coarse_charngram[1], dev_data)
    coarse_test_pred = predict(coarse_charngram[0], coarse_charngram[1], test_data)
    print("\n======= Building fine-grained SVCs =======")
    svc_groups = dict()
    dev_pred, dev_gold = [], []
    test_pred, test_gold = [], []
    for coarse_label in mappings['label_groups']:
        data = get_data_subset(train_data, coarse_label)
        svc_groups[coarse_label] = build_model(data)
        data_subset = dev_data.loc[np.where(coarse_dev_pred == coarse_label)]
        finegrain_dev_pred = predict(svc_groups[coarse_label][0], 
                                     svc_groups[coarse_label][1], 
                                     data_subset)
        dev_pred.extend(finegrain_dev_pred)
        dev_gold.extend(data_subset["label"])
        data_subset = test_data.loc[np.where(coarse_test_pred == coarse_label)]
        finegrain_test_pred = predict(svc_groups[coarse_label][0], 
                                      svc_groups[coarse_label][1], data_subset)
        test_pred.extend(finegrain_test_pred)
        test_gold.extend(data_subset["label"])
    print('\n============== Dev ==============\n')
    print_scores(dev_gold, dev_pred)
    print('\n============== Test ==============\n')
    print_scores(test_gold, test_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('system', type=str)
    args = parser.parse_args()
    config = {
        "train_path": "data/DSLCC-v2017/DSL-TRAIN.txt.tok",
        "dev_path": "data/DSLCC-v2017/DSL-DEV.txt.tok",
        "test_path": "data/DSLCC-v2017/DSL-TEST-GOLD.txt.tok"
    }
    train_data, mappings = utils.load_data(
            config["train_path"], "train")
    dev_data = utils.load_data(config["dev_path"], "dev")
    test_data = utils.load_data(config["test_path"], "test")
    if args.system == "charngram":
        run_charngram(train_data, dev_data, test_data)
    elif args.system == "2step":
        run_coarse_to_fine_models(train_data, dev_data, test_data)
    else:
        NotImplementedError()
