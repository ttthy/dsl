from collections import Counter
from nltk.tokenize import word_tokenize
import argparse
from tqdm import tqdm


def tokenise_raw_data(infile, outfile):
    """ Load data from file """
    with open(infile, 'r') as f, open(outfile, 'w') as w:
        for line in tqdm(f):

            try:
                x, y = line.split('\t')
                x, y = x.strip(), y.strip()
                x = " ".join(word_tokenize(x))
                w.write("{}\t{}\n".format(x, y))
            except ValueError:
                x = line.strip()
                x = " ".join(word_tokenize(x))
                w.write("{}\n".format(x))


if __name__ == "__main__":
    partition = ["TRAIN", "DEV", "TEST-UNLABELLED"]
    abs_path = "data/DSL-"
    for p in partition:
        infilepath = "{}{}.txt".format(abs_path, p)
        outfilepath = "{}{}.txt.tok".format(abs_path, p)
        tokenise_raw_data(infilepath, outfilepath)
