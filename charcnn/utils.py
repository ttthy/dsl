import argparse
import os
import datetime
import numpy as np
import random
import torch


def cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x


def set_seet_every_where(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        print("\n========= Use cuda")
        torch.cuda.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--char_emb_dim', type=int, default=64,
                        help='Dimension of character embeddings')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Dimension of hidden layer')
    parser.add_argument('--n_classes', type=int, default=14,
                        help='Number of classes')
    parser.add_argument('--n_chars', type=int, default=70,
                        help='Number of characters')

    parser.add_argument('--lr', type=float, default=0.001, 
                        help='learning rate [default: 0.001]')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='number of training epochs [default: 100]')

    parser.add_argument('--early_stop', type=int, default=5, 
                        help='iteration numbers to stop without performance increasing')
    parser.add_argument('--dropout', type=float, default=0.5,
                         help='the probability of dropout [default: 0.5]')

    parser.add_argument('--save_dir', type=str, default=None, 
                        help='directory to save model')

    parser.add_argument('--mode', type=str, default='train', help='Running mode: train/test')
    parser.add_argument('--model', type=str, default='bow', help='Model class: bow/cnn')

    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = os.path.join('checkpoints', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(args.save_dir)
    elif not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    set_seet_every_where(42)
    return args
