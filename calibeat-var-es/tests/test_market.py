"""Market-leg tests: recursions, fits, causality of every black box and wrapper, end to end."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calibeat import market as M
from calibeat.fz import fz0, empirical_var_es


def sim_garch_t(T, seed, w=0.02, a=0.08, b=0.90, nu=5.0):
    rng = np.random.default_rng(seed); z = rng.standard_t(nu, T) * np.sqrt((nu - 2) / nu)
    r = np.empty(T); h = w / (1 - a - b)
    for t in range(T):
        r[t] = np.sqrt(h) * z[t]; h = w + a * r[t] ** 2 + b * h
    return r


def test_recursions_match_explicit_loops():
    r = sim_garch_t(500, 1); h = M.ewma_variance(r, 0.94); ref = np.empty(500); ref[0] = h[0]
    for t in range(1, 500):
        ref[t] = 0.94 * ref[t - 1] + 0.06 * r[t - 1] ** 2
    assert np.allclose(h, ref, rtol=1e-12, atol=1e-12)
    g = M.garch_variance(r, 0.02, 0.08, 0.9, 1.3); ref[0] = 1.3
    for t in range(1, 500):
        ref[t] = 0.02 + 0.08 * r[t - 1] ** 2 + 0.9 * ref[t - 1]
    assert np.allclose(g, ref, rtol=1e-12, atol=1e-12)


def test_garch_t_fit_recovers_planted_parameters():
    r = sim_garch_t(6000, 2)
    _, (w, a, b, nu), _ = M.fit_garch_t(r)
    assert abs(a - 0.08) < 0.03 and abs(b - 0.90) < 0.03 and 3.5 < nu < 7.5, (w, a, b, nu)


def test_gas_fit_is_at_least_the_constant_pair_on_iid_data():
    rng = np.random.default_rng(3); r = rng.standard_t(5, 3000) * np.sqrt(0.6)
    _, (a, b, beta, gamma), f = M.fit_gas(r, 0.05)
    q, es = empirical_var_es(r, 0.05); const = float(np.mean(fz0(r[50:], q, es, 0.05)))
    assert f <= const + 0.01, (f, const)
    assert abs(a - q) / abs(q) < 0.25 and b < a < 0, (a, b, q, es)


def test_every_black_box_and_wrapper_is_causal(tmp_path):
    T, t0 = 2000, 1500; r = sim_garch_t(T, 4); r2 = r.copy(); r2[t0 + 1:] = sim_garch_t(T, 5)[t0 + 1:]
    net = M.ewma_variance(r, 0.9); net2 = M.ewma_variance(r2, 0.9)
    for rr, cache in ((r, dict(h_ewma=M.ewma_variance(r), h_net=net)), (r2, dict(h_ewma=M.ewma_variance(r2), h_net=net2))):
        cache['h_garch'], cache['nu_garch'], _ = M.garch_t_paths(rr); cache['tag'] = rr
    c1 = dict(h_ewma=M.ewma_variance(r), h_net=net); c1['h_garch'], c1['nu_garch'], _ = M.garch_t_paths(r)
    c2 = dict(h_ewma=M.ewma_variance(r2), h_net=net2); c2['h_garch'], c2['nu_garch'], _ = M.garch_t_paths(r2)
    for fc in ['ewma_gauss', 'hist250', 'garch_gauss', 'garch_t', 'gas_fz0', 'volpfn_t8']:
        v1, e1 = M.forecast(fc, r, 0.05, c1); v2, e2 = M.forecast(fc, r2, 0.05, c2)
        assert np.allclose(v1[:t0 + 1], v2[:t0 + 1]) and np.allclose(e1[:t0 + 1], e2[:t0 + 1]), fc
        assert not np.allclose(v1[t0 + 1:], v2[t0 + 1:]), fc + ' should react to the future segment'
        if fc == 'ewma_gauss':
            for cb in M.CALIBEATERS:
                if cb.startswith(('switch_', 'guard_')):
                    kind, tail = cb.split('_', 1); kw = dict(window=1000, margin=0.001) if kind == 'guard' else {}
                    a1 = M.calibeat('ratio_' + tail, r, v1, e1, 0.05); a2 = M.calibeat('ratio_' + tail, r2, v2, e2, 0.05)
                    w1 = M.switch(fz0(r, v1, e1, .05), fz0(r, *a1, .05), v1, e1, *a1, **kw); w2 = M.switch(fz0(r2, v2, e2, .05), fz0(r2, *a2, .05), v2, e2, *a2, **kw)
                else:
                    w1 = M.calibeat(cb, r, v1, e1, 0.05); w2 = M.calibeat(cb, r2, v2, e2, 0.05)
                assert np.allclose(w1[0][:t0 + 1], w2[0][:t0 + 1]) and np.allclose(w1[1][:t0 + 1], w2[1][:t0 + 1]), cb


def test_switch_follows_the_dominant_expert_and_is_causal():
    T = 3000; rng = np.random.default_rng(6); y = rng.standard_t(5, T)
    v_good = np.full(T, -2.0); e_good = np.full(T, -2.8); v_bad = np.full(T, -1.0); e_bad = np.full(T, -1.3)
    sg = fz0(y, v_good, e_good, .05); sb = fz0(y, v_bad, e_bad, .05)
    v, e = M.switch(sb, sg, v_bad, e_bad, v_good, e_good)           # original bad, wrapped good -> switch to wrapped
    assert np.all(v[:M.WARM + M.SWITCH_MIN] == v_bad[:M.WARM + M.SWITCH_MIN]) and np.mean(v[1000:] == v_good[1000:]) > 0.95
    v, e = M.switch(sg, sb, v_good, e_good, v_bad, e_bad)           # original good, wrapped bad -> stay
    assert np.mean(v == v_good) > 0.98
    s = fz0(y, v, e, .05); assert np.mean(s[500:]) <= np.mean(sg[500:]) + 0.01
    # a guarded switch with a windowed comparison tracks a leader change: wrapped better first, original better later
    flip = 1500; s_o = np.where(np.arange(T) < flip, sb, sg); s_w = np.where(np.arange(T) < flip, sg, sb)
    v, e = M.switch(s_o, s_w, v_bad, e_bad, v_good, e_good, window=300, margin=0.0)
    assert np.mean(v[800:flip] == v_good[800:flip]) > 0.9 and np.mean(v[flip + 400:] == v_bad[flip + 400:]) > 0.9
    v_inf, _ = M.switch(s_o, s_w, v_bad, e_bad, v_good, e_good)
    assert np.mean(v_inf[flip + 400:] == v_bad[flip + 400:]) < np.mean(v[flip + 400:] == v_bad[flip + 400:])


def _write_market(root, name, T, seed, extract=None):
    r = sim_garch_t(T, seed) / 100.; dates = pd.bdate_range('2000-01-03', periods=T).strftime('%Y-%m-%d').to_numpy().astype('U10')
    split = np.zeros(T, np.int8); split[T - 500:] = 1
    np.savez_compressed(root / f'{name}.npz', date=dates, returns=r, rv5=r ** 2, rk_parzen=r ** 2, split=split)
    if extract is not None:
        h = M.ewma_variance(r * 100, 0.9); h[:100] = np.nan
        np.savez_compressed(extract / f'{name}.npz', date=dates, r=r * 100, h=h, h_lat=h, logscore=h, owner=np.zeros(T))


def test_market_cell_and_summarise_end_to_end(tmp_path):
    data = tmp_path / 'data'; extract = tmp_path / 'extract'; out = tmp_path / 'out'; data.mkdir(); extract.mkdir()
    _write_market(data, 'AAA', 1600, 11, extract); _write_market(data, 'BBB', 1600, 12)
    for n in ['AAA', 'BBB']:
        res = M.market_cell((n, str(data), str(extract), [0.05], 1, str(out))); assert res['rows'] > 0, res
    rows = json.loads((out / 'cells' / 'AAA.json').read_text())['rows']
    assert sorted({r['forecaster'] for r in rows}) == sorted(M.FORECASTERS)
    rows_b = json.loads((out / 'cells' / 'BBB.json').read_text())['rows']
    assert 'volpfn_t8' not in {r['forecaster'] for r in rows_b}          # no extract for BBB
    for r in rows:                                                      # hindsight on the window never loses to the original
        assert r['hindsight_window'] <= r['original'] + 1e-9, r['forecaster']
    M.summarise(out, [0.05])
    assert (out / 'REPORT.txt').exists() and (out / 'decisions.json').exists()
    dec = json.loads((out / 'decisions.json').read_text()); assert any(d['rule'] == 'M2_no_harm' for d in dec)
