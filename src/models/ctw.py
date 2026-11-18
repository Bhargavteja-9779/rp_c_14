"""Re-implementation of CTW: Confident Time-Warping for time-series label-noise learning
(Ma et al., IJCAI 2023), following the official code (github.com/qianlima-lab/CTW,
src/models/model.py, src/models/MultiTaskClassification.py, src/utils/training_helper_CTW.py and
small_loss_criterion_EPS in src/utils/utils.py) and the hyper-parameters of its UCR experiments
(run_sym30.sh: 300 epochs, lr 1e-3, embedding 32, mean_loss_len 10, gamma 0.3, arg_interval 1).

Differences from the official code, all forced or neutral:
* our splits and noise realisations are used instead of the code's 5-fold split and noise injection,
  so that CTW sees exactly the same data as every other method;
* only the final-epoch model is evaluated (as in the official code); per-epoch test evaluation, which the
  official code performs for logging only, is skipped;
* runs on CPU.
Data layout is (N, L, C) as in the official code.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k, stride, padding, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(cin, cout, k, stride=stride, padding=padding), nn.BatchNorm1d(cout),
                                 nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class ConvTBlock(nn.Module):
    def __init__(self, cin, cout, k, stride, padding, output_padding, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.ConvTranspose1d(cin, cout, k, stride=stride, padding=padding,
                                                    output_padding=output_padding),
                                 nn.BatchNorm1d(cout), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


def conv_out_len(seq_len, k, stride, padding, dilation, stack):
    for _ in range(stack):
        seq_len = int((seq_len + 2 * padding - dilation * (k - 1) - 1) / stride + 1)
    return seq_len


class Encoder(nn.Module):
    def __init__(self, cin, filters, emb, k, stride, padding, dropout):
        super().__init__()
        layers = []
        for i, c in enumerate(filters):
            layers.append(ConvBlock(cin if i == 0 else filters[i - 1], c, k, stride, padding, dropout))
        self.network = nn.Sequential(*layers)
        self.conv1x1 = nn.Conv1d(filters[-1], emb, 1)

    def forward(self, x):                      # x: (N, L, C)
        x = self.network(x.transpose(2, 1))
        x = F.max_pool1d(x, kernel_size=x.shape[2])
        return self.conv1x1(x)                 # (N, emb, 1)


class Decoder(nn.Module):
    def __init__(self, emb, filters, seq_len, cout, k, stride, padding, dropout):
        super().__init__()
        filters = filters[::-1]
        nb = len(filters)
        self.compressed_len = conv_out_len(seq_len, k, stride, padding, 1, nb)
        if stride > 1:
            op, seq = [], seq_len
            for _ in range(nb):
                op.append(seq % 2)
                seq = conv_out_len(seq, k, stride, padding, 1, 1)
            op = [1 - x for x in op[::-1]] if k % 2 == 1 else op[::-1]
        else:
            op = [0] * nb
        layers = [ConvTBlock(emb if i == 0 else filters[i - 1], filters[i], k, stride, padding, op[i], dropout)
                  for i in range(nb)]
        self.network = nn.Sequential(*layers)
        self.upsample = nn.Linear(1, self.compressed_len)
        self.conv1x1 = nn.Conv1d(filters[-1], cout, 1)
        self.seq_len = seq_len

    def forward(self, h):
        x = self.conv1x1(self.network(self.upsample(h)))
        if x.shape[2] != self.seq_len:          # guard; the official padding rule normally makes this exact
            x = F.interpolate(x, size=self.seq_len, mode="linear", align_corners=False)
        return x.transpose(2, 1)


class Classifier(nn.Module):
    def __init__(self, d_in, n_class, d_hidd=128, dropout=0.2):
        super().__init__()
        self.dense1, self.norm, self.act = nn.Linear(d_in, d_hidd), nn.BatchNorm1d(d_hidd), nn.ReLU()
        self.dropout, self.dense2 = nn.Dropout(dropout), nn.Linear(d_hidd, n_class)

    def forward(self, x):
        h = self.dense1(x)
        if len(x) > 1:                          # official code skips BatchNorm for single-sample batches
            h = self.norm(h)
        return self.dense2(self.dropout(self.act(h)))


class CTWNet(nn.Module):
    def __init__(self, cin, seq_len, n_class, filters=(128, 128, 256, 256), emb=32, k=4, stride=2, padding=2,
                 dropout=0.2, d_hidd=128):
        super().__init__()
        self.encoder = Encoder(cin, list(filters), emb, k, stride, padding, dropout)
        self.decoder = Decoder(emb, list(filters), seq_len, cin, k, stride, padding, dropout)
        self.classifier = Classifier(emb, n_class, d_hidd, dropout)

    def forward(self, x):
        return self.classifier(self.encoder(x).squeeze(-1))


def _select(loss_all, idx, epoch, labels, n_class, mean_loss_len=10, gamma=0.3):
    """small_loss_criterion_EPS: class-wise z-scored running loss, keep samples at or below the mean."""
    cur = loss_all[idx, epoch]
    if mean_loss_len > epoch:
        lm = gamma * cur + (1 - gamma) * loss_all[idx, :epoch].mean(axis=1)
    elif mean_loss_len < 2:
        lm = cur
    else:
        lm = gamma * cur + (1 - gamma) * loss_all[idx, epoch - mean_loss_len + 1:epoch].mean(axis=1)
    std_loss, order = [], []
    bidx = np.arange(len(idx))
    for c in range(n_class):
        m = labels == c
        if m.sum() > 1:
            v = lm[m]
            sd = v.std()
            std_loss.append((v - v.mean()) / sd if sd > 0 else np.zeros_like(v))   # official: nan when sd == 0
            order.append(bidx[m])
        elif m.sum() == 1:
            std_loss.append(np.zeros(1)); order.append(bidx[m])
    std_loss, order = np.concatenate(std_loss), np.concatenate(order)
    kk = int(std_loss.size * (std_loss <= std_loss.mean()).mean())
    sel = order[np.argsort(std_loss, kind="stable")[:kk]]
    return sel


def train_ctw(X, y, n_class, seed=0, epochs=300, lr=1e-3, device="cpu"):
    """X: (N, L, C) float32, y: noisy integer labels.  Returns the trained network (final epoch)."""
    import tsaug
    torch.manual_seed(seed); np.random.seed(seed)
    n, L, Cin = X.shape
    bs = min(n // 10, 128)
    if bs < 2:
        bs = 2
    if n % bs == 1:
        bs -= 1
    warmup = 30 if n <= 1000 else (15 if n <= 3000 else 10)
    net = CTWNet(Cin, L, n_class).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-4)
    ce = nn.CrossEntropyLoss(reduction="none")
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    loss_all = np.zeros((n, epochs))
    tw = tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3)
    g = torch.Generator().manual_seed(seed)
    for e in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            if len(idx) == 1 and e <= warmup:
                continue
            x, yy = Xt[idx].to(device), yt[idx].to(device)
            h = net.encoder(x)
            out = net.classifier(h.squeeze(-1))
            rec = F.mse_loss(net.decoder(h), x)
            l = ce(out, yy)
            loss_all[idx.numpy(), e] = l.detach().cpu().numpy()
            if e <= warmup:
                loss = l.sum() + rec
            else:
                sel = _select(loss_all, idx.numpy(), e, yy.cpu().numpy(), n_class)
                mask = torch.zeros(len(idx)); mask[torch.as_tensor(sel, dtype=torch.long)] = 1.0
                loss = (mask.to(device) * l).sum() + rec
                if len(sel) > 1:
                    xa = torch.tensor(tw.augment(x[sel].detach().cpu().numpy()), dtype=torch.float32).to(device)
                    loss = loss + ce(net.classifier(net.encoder(xa).squeeze(-1)), yy[sel]).sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
    return net


@torch.no_grad()
def predict(net, X, device="cpu", bs=256):
    net.eval()
    out = [net(torch.tensor(X[i:i + bs], dtype=torch.float32).to(device)).cpu().numpy() for i in range(0, len(X), bs)]
    return np.concatenate(out).argmax(1)
