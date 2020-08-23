import os
import sys
import pdb
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from charcnn.utils import cuda


def train(train_iter, dev_iter, model, args, vocabs):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, amsgrad=True)

    best_acc = 0
    patience = args.early_stop

    model.train()
    print("\n============== Start training ==============\n")
    for epoch in range(1, args.epochs+1):
        print("\n===== Epoch {}".format(epoch))
        for batch in train_iter:
            batch_input, batch_label = batch.text, batch.label
            batch_input.t_()
            batch_input, batch_label = cuda(batch_input), cuda(batch_label)
            
            optimizer.zero_grad()
            logit = model(batch_input)

            loss = F.cross_entropy(logit, batch_label)
            loss.backward()
            optimizer.step()

        dev_acc = evaluate(dev_iter, model, args, vocabs)
        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(
                model.state_dict(), 
                os.path.join(args.save_dir, 'best_dev.pth'))
            patience = args.early_stop
        else:
            patience -= 1
            print('Early stop after {} epochs'.format(patience))
            if patience == 0:
                break
        

def evaluate(data_iter, model, args, vocabs):
    model.eval()
    n_correct, avg_loss = 0, 0 
    with torch.no_grad():
        for batch in data_iter:
            batch_input, batch_label = batch.text, batch.label
            batch_input.t_()
            batch_input, batch_label = cuda(batch_input), cuda(batch_label)
            logit = model(batch_input)
            values, indices = torch.max(logit, 1)
            loss = F.cross_entropy(logit, batch_label, size_average=False)
            avg_loss += loss.detach().data.item()
            n_correct += (indices.view(batch_label.size()).data == batch_label.data).sum().detach().data.item()
    n_samples = len(data_iter.dataset)
    avg_loss /= n_samples
    accuracy = 100. * n_correct / n_samples
    print("\nEvaluation: loss: {:.5f} acc: {:.4f} ({}/{})" .format(
        avg_loss, accuracy, n_correct, n_samples))
    return accuracy

