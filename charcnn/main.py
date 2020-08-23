import pdb
import os
import argparse
from torchtext import data
import torch
from charcnn.dataset import load_dsl_data
import charcnn.utils as utils
from charcnn.model import BOW, CharCNN, CharMultiCNN
import charcnn.runner as runner


if __name__ == "__main__":
    args = utils.get_args()
    
    print('\n========= Load data =========\n')
    text_field = data.Field(tokenize=list)
    label_field = data.LabelField()
    vocabs = {'text': text_field, 'label': label_field}
    train_iter, dev_iter, test_iter = load_dsl_data(text_field, label_field)
    args.n_chars = len(text_field.vocab)
    args.n_classes = len(label_field.vocab)
    print("\n>>> No. of characters: {}\n>>> No. of classes: {}".format(
        args.n_chars, args.n_classes))
    
    config = {
        'n_chars': args.n_chars,
        'n_classes': args.n_classes,
        'char_emb_dim': args.char_emb_dim,
        'hidden_dim': args.hidden_dim,
        'padding_idx': text_field.vocab.stoi[text_field.pad_token],
        'dropout': args.dropout
    }
    if args.model == 'bow':
        model = BOW(config)
    elif args.model == 'cnn':
        model = CharCNN(config)
    elif args.model == 'multicnn':
        model = CharMultiCNN(config)
    model.summary()
    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        runner.train(train_iter, dev_iter, model, args, vocabs)
    
    best_dev_path = os.path.join(args.save_dir, "best_dev.pth")
    if os.path.isfile(best_dev_path):
        model.load_state_dict(torch.load(best_dev_path))
        print("\n================== Start evaluation")
        print("\n======= Dev set")
        runner.evaluate(dev_iter, model, args, vocabs)
        print("\n======= Test set")
        runner.evaluate(test_iter, model, args, vocabs)

