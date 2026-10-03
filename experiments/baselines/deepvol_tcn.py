#!/usr/bin/env python3
"""Daily-return TCN baseline inspired by DeepVol (Moreno-Pino & Zohren, 2024).

The input is a 22-day window of squared daily returns, used as variance proxies.
These are not intraday realised-variance observations. The network uses dilated
causal convolutions and residual connections, with two blocks of three layers
by default. It is fitted with QLIKE loss and Adam on expanding past-data windows.

After constructing the variance forecasts, the code fits one Student-t degrees
of freedom to all residuals with available forecasts, including the evaluation
period. The resulting log densities, VaR and ES are retrospective calibrations;
they should not be described as fully out-of-sample density or tail forecasts.

Example:
    from experiments.baselines.deepvol_tcn import run_deepvol
    out = run_deepvol(r, burn=500, levels=(0.99, 0.975))

Outputs h, logscore, var<level> and es<level> align with the input returns;
unissued forecasts remain NaN. nu contains the fitted tail parameter.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar

WINDOW = 22  # Number of past daily squared returns.


class _DilatedBlock(nn.Module):
    def __init__(self, channels, dilation, kernel_size=3):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.act = nn.ReLU()
        self.res = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        y = nn.functional.pad(x, (self.pad, 0))  # Left padding keeps the convolution causal.
        y = self.act(self.conv(y))
        return x + self.res(y)


class DeepVolTCN(nn.Module):
    """Dilated causal convolutions for a window of squared daily returns.

    Adapted from Moreno-Pino & Zohren (2024, Eq. 24-27); see the module notes
    for differences from the intraday setting.
    """

    def __init__(self, in_channels=1, channels=32, kernel_size=3, num_blocks=2, num_layers=3):
        super().__init__()
        self.inp = nn.Conv1d(in_channels, channels, 1)
        blocks = []
        for _ in range(num_blocks):
            for l in range(num_layers):
                blocks.append(_DilatedBlock(channels, dilation=2**l, kernel_size=kernel_size))
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Conv1d(channels, 1, 1)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.out(h)[:, :, -1].squeeze(
            -1
        )  # last time-step = tomorrow's log-variance forecast


def qlike_loss(log_pred, real_var, eps=1e-8):
    pred_var = torch.exp(log_pred)
    return torch.mean(torch.log(pred_var + eps) + real_var / (pred_var + eps))


def _make_windows(rv, window):
    log_rv = np.log(np.maximum(rv, 1e-12))
    N = len(rv) - window
    X = np.stack([log_rv[i : i + window] for i in range(N)])
    y = rv[window : window + N]
    return X.astype(np.float32), y.astype(np.float32)


def run_deepvol(
    r,
    burn=500,
    levels=(0.99, 0.975),
    window=WINDOW,
    epochs=60,
    lr=1e-3,
    refit_every=250,
    seed=0,
    device=None,
):
    """Fit expanding-window variance forecasts, then calibrate a Student-t tail.

    Refit the network every refit_every observations. Tail calibration uses
    all available forecast residuals, including the evaluation period.
    Return aligned h, logscore, var<level> and es<level> arrays, plus fitted nu.
    """
    from experiments.baselines.univol_eval import t_logscore, t_var_es

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    r = np.asarray(r, dtype=float)
    T = len(r)
    rv = r**2
    X_all, y_all = _make_windows(rv, window)  # Window i predicts squared return rv[window + i].

    h_path = np.full(T, np.nan)
    for t0 in range(max(burn, window + 50), T, refit_every):
        train_end = t0 - window
        if train_end < 50:
            continue
        model = DeepVolTCN().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        Xtr = torch.tensor(X_all[:train_end]).unsqueeze(1).to(device)
        ytr = torch.tensor(y_all[:train_end]).to(device)
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = qlike_loss(model(Xtr), ytr)
            loss.backward()
            opt.step()
        model.eval()
        t_end = min(t0 + refit_every, T)
        idx_lo, idx_hi = t0 - window, t_end - window
        Xte = torch.tensor(X_all[idx_lo:idx_hi]).unsqueeze(1).to(device)
        with torch.no_grad():
            pred = model(Xte).cpu().numpy()
        h_path[t0:t_end] = np.exp(pred)

    ok = ~np.isnan(h_path)
    out = {'h': h_path, 'logscore': np.full(T, np.nan)}
    for lv in levels:
        out[f'var{lv:g}'] = np.full(T, np.nan)
        out[f'es{lv:g}'] = np.full(T, np.nan)
    if ok.sum() < 50:
        out['nu'] = float('nan')
        return out

    def neg_ll(nu):
        return -float(np.mean(t_logscore(r[ok], h_path[ok], nu)))

    nu = float(minimize_scalar(neg_ll, bounds=(2.1, 60.0), method='bounded').x)
    out['logscore'][ok] = t_logscore(r[ok], h_path[ok], nu)
    for lv in levels:
        var_ok, es_ok = t_var_es(h_path[ok], nu, lv)
        out[f'var{lv:g}'][ok] = var_ok
        out[f'es{lv:g}'][ok] = es_ok
    out['nu'] = nu
    return out


if __name__ == '__main__':
    import time

    rng = np.random.default_rng(0)
    r = rng.standard_normal(1500) * 0.01
    t0 = time.time()
    out = run_deepvol(r, burn=600, refit_every=250, epochs=60)
    dt = time.time() - t0
    ok = ~np.isnan(out['h'])
    print(
        f'{dt:.1f}s for T=1500 (smoke test); mean logscore {np.nanmean(out["logscore"]):.4f}; '
        f'mean h {np.nanmean(out["h"][ok]):.6f}; fitted nu {out["nu"]:.2f}'
    )
