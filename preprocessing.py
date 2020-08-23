from collections import Counter
import spacy
import argparse
from tqdm import tqdm


def tokenise_raw_data(infile, outfile, spacy_model):
    """ Load data from file """
    with open(infile, 'r') as f, open(outfile, 'w') as w:
        for line in tqdm(f):

            try:
                x, y = line.split('\t')
                x, y = x.strip(), y.strip()
                x = " ".join([tok.text for tok in spacy_model(x)])
                w.write("{}\t{}\n".format(x, y))
            except ValueError:
                x = line.strip()
                x = " ".join([tok.text for tok in spacy_model(x)])
                w.write("{}\n".format(x))


if __name__ == "__main__":
    spacy_model = spacy.load("pt_core_news_sm")
    partition = ["TRAIN", "DEV", "TEST-UNLABELLED"]
    abs_path = "data/DSL-"
    for p in partition:
        infilepath = "{}{}.txt".format(abs_path, p)
        outfilepath = "{}{}.txt.tok".format(abs_path, p)
        tokenise_raw_data(infilepath, outfilepath, spacy_model)
