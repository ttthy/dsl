import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        params_memory = sum([sys.getsizeof(p) for p in model_parameters])
        print("""
---------------- Model ----------------
Named modules: {}
Trainable parameters: {}, {:.5f} MB, {}
Model: \n{}
        """.format(
            [k for k, v in self.named_parameters()],
            params, params*32*1.25e-7, params_memory,
            self
        ))
    

class BOW(BaseModel):
    def __init__(self, config):
        super(BOW, self).__init__()
        self.embedding = nn.EmbeddingBag(
            config['n_chars'], config['char_emb_dim'], 
            scale_grad_by_freq=True, mode='mean')
        self.fc = nn.Sequential(
            nn.Linear(config['char_emb_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['n_classes'])
        )
        self.padding_idx = config['padding_idx']
        self.dropout = config['dropout']
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)
    
    def forward(self, batch_input):
        # [B, N, C] --> [B, ]
        x = self.embedding(batch_input)
        x = F.dropout(x, self.dropout)
        logit = self.fc(x)
        return logit


class CharCNN(BaseModel):
    def __init__(self, config):
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(
            config['n_chars'], config['char_emb_dim'], padding_idx=config['padding_idx'])
        self.conv = nn.Conv1d(config['char_emb_dim'], config['hidden_dim'], 
                               kernel_size=5, stride=1)
        self.fc = nn.Linear(config['hidden_dim'], config['n_classes'])
        self.dropout = config['dropout']

    def forward(self, batch_input):
        # [B, N, D]
        x = self.embedding(batch_input)
        # [B, D, N]
        x = x.transpose(1, 2)
        # [B, H, L], L from N
        x = F.relu(self.conv(x))
        x = F.max_pool1d(x, x.size(-1)).squeeze(-1)
        x = F.dropout(x, self.dropout)
        logit = self.fc(x)
        return logit


class CharMultiCNN(BaseModel):
    def __init__(self, config):
        super(CharMultiCNN, self).__init__()
        self.embedding = nn.Embedding(
            config['n_chars'], config['char_emb_dim'], padding_idx=config['padding_idx'])
        kernels = list(range(2, 6))
        self.convs = nn.ModuleList([
            nn.Conv1d(config['char_emb_dim'], config['char_emb_dim'], 
                      kernel_size=ks, stride=1)
            for ks in kernels
        ])
        self.fc = nn.Linear(len(kernels)*config['char_emb_dim'], config['n_classes'])
        self.dropout = config['dropout']

    def forward(self, batch_input):
        # [B, N, D]
        emb_x = self.embedding(batch_input)
        # [B, D, N]
        emb_x = emb_x.transpose(1, 2)
        # N_kernel*[B, D, L], L from N
        conv_x = [F.relu(conv(emb_x)) for conv in self.convs]
        # N_kernel*[B, D]
        pool_x = [F.max_pool1d(x, x.size(-1)).squeeze(-1) for x in conv_x]
        # [B, N_kernel*D]
        fc_x = torch.cat(pool_x, dim=1)
        fc_x = F.dropout(fc_x, self.dropout)
        # [B, N_classes]
        logit = self.fc(fc_x)
        return logit
