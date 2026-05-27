#!/usr/bin/env python3
"""Harmonized response-derived DIF design for BBH.

This script aligns family-label, strict MADNESS, and SRRC (EP-1b) analyses to the same
model split and item split:
  - model split: cluster-aware 70/30 discovery/evaluation split, stratified by
    Llama/Qwen/Other where possible;
  - item split: domain-stratified 70/30 group-discovery / DIF-test split;
  - response-derived groups learned only from discovery models x group-discovery
    items;
  - MH-DIF run only on discovery models x DIF-test items;
  - consequences evaluated only on held-out evaluation models.

The implementation is intentionally conservative: it preserves the local
standard MH/ETS/BH machinery and documents where it is a simplified replication.
"""
from __future__ import annotations

import hashlib, json, math, os, re, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, cdist
from scipy.stats import norm

import argparse

DEFAULT_DATA_DIR = Path('data/processed_bbh')
DEFAULT_OUT = Path('outputs_recomputed')

_parser = argparse.ArgumentParser(description='Run harmonized BBH LLM DIF analysis from processed BBH files.')
_parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR, help='Directory containing processed BBH CSV inputs.')
_parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT, help='Output directory for recomputed harmonized analysis.')
_parser.add_argument('--seed', type=int, default=None, help='Override BBH_HARMONIZED_SEED.')
_parser.add_argument('--score-bins', type=int, default=None, help='Number of score bins for MH matching strata.')
_parser.add_argument('--min-group-per-stratum', type=int, default=None, help='Minimum reference/focal models per score stratum.')
_parser.add_argument('--anchor-q', type=float, default=None, help='Fraction of lowest-|delta_MH| items used as anchors.')
_parser.add_argument('--madness-min-valid-items-floor', type=int, default=None, help='Minimum decoded item floor for strict MADNESS model retention; default 50, smaller for fixtures.')
_parser.add_argument('--min-cluster-size', type=int, default=None, help='Minimum cluster size for SRRC candidate selection; default 100, smaller for fixtures.')
_args, _unknown = _parser.parse_known_args()

DATA_DIR = _args.data_dir
OUT = _args.out_dir
OUT.mkdir(parents=True, exist_ok=True)

RESP = DATA_DIR / 'BBH_responses.csv.gz'
MODEL_META = DATA_DIR / 'BBH_model_metadata.csv.gz'
ITEM_META = DATA_DIR / 'BBH_item_metadata.csv.gz'
SIDECAR = DATA_DIR / 'BBH_decoded_predictions_main_decodable.csv.gz'

SEED = _args.seed if _args.seed is not None else int(os.environ.get('BBH_HARMONIZED_SEED', '20260526'))
DISCOVERY_FRAC = float(os.environ.get('BBH_HARMONIZED_DISCOVERY_FRAC', '0.70'))
ITEM_DISCOVERY_FRAC = float(os.environ.get('BBH_HARMONIZED_ITEM_DISCOVERY_FRAC', '0.70'))
N_BINS = _args.score_bins if _args.score_bins is not None else int(os.environ.get('BBH_HARMONIZED_SCORE_BINS', '20'))
MIN_GROUP_PER_STRATUM = _args.min_group_per_stratum if _args.min_group_per_stratum is not None else int(os.environ.get('BBH_HARMONIZED_MIN_GROUP_PER_STRATUM', '10'))
ANCHOR_Q = _args.anchor_q if _args.anchor_q is not None else float(os.environ.get('BBH_HARMONIZED_ANCHOR_Q', '0.20'))
CHUNK = int(os.environ.get('BBH_HARMONIZED_CHUNK', '1000000'))
MADNESS_MIN_VALID_ITEMS_FLOOR = _args.madness_min_valid_items_floor if _args.madness_min_valid_items_floor is not None else int(os.environ.get('BBH_HARMONIZED_MADNESS_MIN_VALID_ITEMS_FLOOR', '50'))
K_LIST = [2, 3, 4, 5, 6, 7, 8]
N_INIT = 8
MAX_ITER = 100
MIN_CLUSTER_SIZE = _args.min_cluster_size if _args.min_cluster_size is not None else int(os.environ.get('BBH_HARMONIZED_MIN_CLUSTER_SIZE', '100'))


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def simple_base_cluster(hf_model_id: str, model_id: str) -> str:
    s = (hf_model_id or '').strip() or (model_id or '').replace('__', '/')
    owner, _, name = s.partition('/')
    n = name.lower()
    pats = [
        r'(llama[-_ ]?\d+(?:\.\d+)?[-_ ]?\d+b)',
        r'(meta[-_ ]?llama[-_ ]?\d+(?:\.\d+)?[-_ ]?\d+b)',
        r'(qwen(?:\d+(?:\.\d+)?)?[-_ ]?\d+(?:\.\d+)?b)',
        r'(qwen[-_ ]?\d+(?:\.\d+)?[-_ ]?\d+(?:\.\d+)?b)',
    ]
    for pat in pats:
        m = re.search(pat, n)
        if m:
            return re.sub(r'[^a-z0-9.]+', '_', m.group(1)).strip('_')
    n2 = re.sub(r'(?i)([-_ ]?(instruct|chat|dpo|sft|rlhf|lora|awq|gptq|gguf|merge|merged|tuned|finetune|finetuned)).*$', '', name)
    n2 = n2.strip('-_ ') or name
    return f'{owner}/{n2}'


def bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    tmp = np.empty(n, dtype=float); tmp[order] = adj
    out[ok] = tmp
    return out


def ets_category(delta: np.ndarray, p_adj: np.ndarray) -> np.ndarray:
    a = (~np.isfinite(p_adj)) | (p_adj > 0.05) | (np.abs(delta) < 1.0)
    c = (~a) & (np.abs(delta) >= 1.5)
    return np.where(a, 'A', np.where(c, 'C', 'B'))


def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(MODEL_META)
    meta['split_family'] = np.where(meta['architecture_family'].isin(['Llama', 'Qwen']), meta['architecture_family'], 'Other')
    meta['cluster_proxy'] = [simple_base_cluster(h, i) for h, i in zip(meta.get('hf_model_id', ''), meta['id'])]
    item = pd.read_csv(ITEM_META)
    return meta, item


def split_models(meta: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for fam, df in meta.groupby('split_family', sort=True):
        clusters = df.groupby('cluster_proxy')['id'].apply(list).reset_index()
        clusters['n'] = clusters['id'].map(len)
        clusters = clusters.sample(frac=1.0, random_state=SEED + {'Llama': 1, 'Qwen': 2, 'Other': 3}.get(fam, 4)).reset_index(drop=True)
        target = DISCOVERY_FRAC * clusters['n'].sum()
        acc = 0; discovery_clusters = set()
        for _, row in clusters.iterrows():
            if acc < target:
                discovery_clusters.add(row['cluster_proxy']); acc += row['n']
        x = df.copy()
        x['model_split'] = np.where(x['cluster_proxy'].isin(discovery_clusters), 'discovery', 'evaluation')
        parts.append(x)
    out = pd.concat(parts, ignore_index=True)
    return out


def split_items(item: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for dom, df in item.groupby('domain', sort=True):
        stable_dom_seed = int(hashlib.md5(dom.encode('utf-8')).hexdigest()[:8], 16) % 10000
        x = df.sample(frac=1.0, random_state=SEED + stable_dom_seed).reset_index(drop=True).copy()
        n_disc = int(round(ITEM_DISCOVERY_FRAC * len(x)))
        n_disc = min(max(n_disc, 1), len(x)-1) if len(x) > 1 else len(x)
        x['item_split'] = ['group_discovery' if i < n_disc else 'dif_test' for i in range(len(x))]
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def load_response_matrix(models: list[str], items: list[str]) -> np.ndarray:
    midx = {m: i for i, m in enumerate(models)}; iidx = {it: j for j, it in enumerate(items)}
    mat = np.full((len(models), len(items)), -1, dtype=np.int8)
    total = kept = 0
    for ci, ch in enumerate(pd.read_csv(RESP, usecols=['id', 'item', 'resp'], chunksize=CHUNK)):
        total += len(ch)
        ch = ch[ch['id'].isin(midx) & ch['item'].isin(iidx)]
        if len(ch):
            mi = ch['id'].map(midx).to_numpy(np.int32)
            ii = ch['item'].map(iidx).to_numpy(np.int32)
            mat[mi, ii] = ch['resp'].to_numpy(np.int8)
            kept += len(ch)
        if ci % 5 == 0:
            log(f'response chunks={ci+1} total={total:,} kept={kept:,}')
    return mat


def qcut_strata(scores: np.ndarray) -> np.ndarray:
    s = pd.Series(scores)
    return pd.qcut(s, q=N_BINS, labels=False, duplicates='drop').to_numpy(dtype=int)


def mh_dif(mat: np.ndarray, row_idx: np.ndarray, item_idx: np.ndarray, labels: np.ndarray, ref: str, focal: str, out_prefix: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run total-score then anchor-refined MH-DIF. Returns anchor result, diag, total result."""
    def one_pass(score_items: np.ndarray, pass_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        y_score = mat[np.ix_(row_idx, score_items)]
        valid_score = y_score >= 0
        scores = np.where(valid_score, y_score, 0).sum(axis=1) / np.maximum(valid_score.sum(axis=1), 1)
        strata = qcut_strata(scores)
        lab = labels.copy()
        diag_rows = []
        keep_strata = []
        for st in sorted(set(strata.tolist())):
            n_ref = int(np.sum((strata == st) & (lab == ref)))
            n_foc = int(np.sum((strata == st) & (lab == focal)))
            usable = n_ref >= MIN_GROUP_PER_STRATUM and n_foc >= MIN_GROUP_PER_STRATUM
            diag_rows.append({'pass': pass_name, 'stratum': int(st), f'n_{ref}': n_ref, f'n_{focal}': n_foc, 'usable': usable})
            if usable: keep_strata.append(st)
        keep_mask = np.isin(strata, keep_strata) & np.isin(lab, [ref, focal])
        rows = []
        for jj in item_idx:
            num = den = A_sum = B_sum = C_sum = D_sum = 0.0
            n_strata = 0
            col = mat[row_idx, jj]
            for st in keep_strata:
                m_ref = keep_mask & (strata == st) & (lab == ref) & (col >= 0)
                m_foc = keep_mask & (strata == st) & (lab == focal) & (col >= 0)
                if not (m_ref.any() and m_foc.any()):
                    continue
                A = float(col[m_ref].sum()); B = float(m_ref.sum() - A)
                C = float(col[m_foc].sum()); D = float(m_foc.sum() - C)
                if min(A, B, C, D) == 0:
                    A += 0.5; B += 0.5; C += 0.5; D += 0.5
                N = A + B + C + D
                num += A * D / N; den += B * C / N
                A_sum += A; B_sum += B; C_sum += C; D_sum += D
                n_strata += 1
            alpha = num / den if den > 0 else np.nan
            delta = -2.35 * math.log(alpha) if alpha and alpha > 0 and math.isfinite(alpha) else np.nan
            se = math.sqrt(1/A_sum + 1/B_sum + 1/C_sum + 1/D_sum) if min(A_sum, B_sum, C_sum, D_sum) > 0 else np.nan
            z = math.log(alpha) / se if se and alpha and alpha > 0 and math.isfinite(alpha) else np.nan
            p = 2 * norm.sf(abs(z)) if math.isfinite(z) else np.nan
            rows.append({'item_col': int(jj), 'alpha_mh': alpha, 'delta_mh': delta, 'se_log_or': se, 'z': z, 'p': p, 'n_strata': n_strata, 'A': A_sum, 'B': B_sum, 'C': C_sum, 'D': D_sum})
        res = pd.DataFrame(rows)
        res['p_adj'] = bh_adjust(res['p'].to_numpy())
        res['ets_cat'] = ets_category(res['delta_mh'].to_numpy(), res['p_adj'].to_numpy())
        res['direction'] = np.where(res['delta_mh'] > 0, focal, ref)
        diag = pd.DataFrame(diag_rows)
        return res, diag

    total, diag_total = one_pass(item_idx, 'total_score_pass')
    anchor_n = max(1, int(math.ceil(ANCHOR_Q * len(total))))
    anchor_items = total.assign(abs_delta=total['delta_mh'].abs()).sort_values('abs_delta').head(anchor_n)['item_col'].to_numpy(int)
    anchor, diag_anchor = one_pass(anchor_items, 'anchor20_pass')
    total.to_csv(out_prefix.with_name(out_prefix.name + '_total_score_pass.csv'), index=False)
    anchor.to_csv(out_prefix.with_name(out_prefix.name + '_anchor20_pass.csv'), index=False)
    pd.concat([diag_total, diag_anchor]).to_csv(out_prefix.with_name(out_prefix.name + '_strata_diagnostics.csv'), index=False)
    return anchor, diag_anchor, total


def summarize_dif(name: str, r: pd.DataFrame, diag: pd.DataFrame, item_lookup: pd.DataFrame) -> dict:
    rr = r.merge(item_lookup, left_on='item_col', right_on='item_col', how='left')
    c = rr['ets_cat'].value_counts(); n = len(rr); C = rr[rr['ets_cat'] == 'C']
    return {
        'analysis': name, 'pass': 'anchor20_pass', 'items': int(n),
        'A_count': int(c.get('A', 0)), 'B_count': int(c.get('B', 0)), 'C_count': int(c.get('C', 0)),
        'A_pct': float(100*c.get('A', 0)/n) if n else None,
        'B_pct': float(100*c.get('B', 0)/n) if n else None,
        'C_pct': float(100*c.get('C', 0)/n) if n else None,
        'median_abs_delta_C': float(C['delta_mh'].abs().median()) if len(C) else None,
        'usable_strata': int(diag['usable'].sum()) if 'usable' in diag else None,
        'score_bins': int(diag.shape[0]),
    }


def kmeans(X: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed); n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=np.float32)
    centers[0] = X[rng.integers(n)]
    d2 = cdist(X, centers[:1], 'sqeuclidean').min(axis=1)
    for c in range(1, k):
        probs = d2 / d2.sum() if d2.sum() > 0 else np.ones(n)/n
        centers[c] = X[rng.choice(n, p=probs)]
        d2 = np.minimum(d2, cdist(X, centers[c:c+1], 'sqeuclidean').ravel())
    lab = np.full(n, -1, dtype=np.int32)
    for _ in range(MAX_ITER):
        new = cdist(X, centers, 'sqeuclidean').argmin(axis=1).astype(np.int32)
        if np.array_equal(new, lab): break
        lab = new
        for c in range(k):
            centers[c] = X[lab == c].mean(axis=0) if (lab == c).any() else X[rng.integers(n)]
    inertia = float(((X - centers[lab])**2).sum())
    return lab, centers, inertia


def ari(a: np.ndarray, b: np.ndarray) -> float:
    da = {v: i for i, v in enumerate(sorted(set(a.tolist())))}; db = {v: i for i, v in enumerate(sorted(set(b.tolist())))}
    M = np.zeros((len(da), len(db)), dtype=float)
    for x, y in zip(a, b): M[da[x], db[y]] += 1
    n = M.sum(); comb = lambda x: x*(x-1)/2
    s = comb(M).sum(); r = comb(M.sum(1)).sum(); c = comb(M.sum(0)).sum(); t = comb(n); e = r*c/t if t else 0; mx = .5*(r+c)
    return float((s-e)/(mx-e)) if mx != e else 0.0


def ability_r2(labels: np.ndarray, score: np.ndarray) -> float:
    o = score.mean(); total = ((score-o)**2).sum(); between = 0.0
    for g in np.unique(labels):
        s = score[labels == g]; between += len(s) * (s.mean()-o)**2
    return float(between/total) if total > 0 else np.nan


def ep1b_groups(mat: np.ndarray, disc_rows: np.ndarray, eval_rows: np.ndarray, group_items: np.ndarray, item_domains: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Learn SRRC groups on discovery models and assign evaluation models to frozen centroids.

    SRRC residualizes domain-level accuracy profiles against total score, normalizes
    residual shape, selects a stable k-means solution on discovery models only,
    and assigns held-out evaluation models by nearest frozen centroid.
    """
    domains = sorted(set(item_domains[group_items].tolist()))
    dom_to_i = {d: i for i, d in enumerate(domains)}

    def features(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        D = np.zeros((len(rows), len(domains)), dtype=np.float64); C = np.zeros_like(D)
        sub = mat[np.ix_(rows, group_items)]
        for local_j, global_j in enumerate(group_items):
            di = dom_to_i[item_domains[global_j]]
            y = sub[:, local_j]
            ok = y >= 0
            D[ok, di] += y[ok]; C[ok, di] += 1
        F = D / np.maximum(C, 1)
        total = np.where(sub >= 0, sub, 0).sum(axis=1) / np.maximum((sub >= 0).sum(axis=1), 1)
        return F, total

    Fd, sd = features(disc_rows)
    Zd = Fd - sd[:, None]
    Xpoly = np.column_stack([np.ones_like(sd), sd, sd**2, sd**3])
    beta = np.linalg.lstsq(Xpoly, Zd, rcond=None)[0]
    Rd = Zd - Xpoly @ beta
    norm = np.linalg.norm(Rd, axis=1, keepdims=True); norm[norm < 1e-8] = 1
    Rshape = Rd / norm
    mu = Rshape.mean(axis=0, keepdims=True); sigma = Rshape.std(axis=0, keepdims=True); sigma[sigma < 1e-8] = 1
    Xd = ((Rshape - mu) / sigma).astype(np.float32)
    metrics = []
    best = None
    for k in K_LIST:
        labs = []; centers = []; inert = []
        for s in range(N_INIT):
            lab, cen, inn = kmeans(Xd, k, SEED + 100*k + s)
            labs.append(lab); centers.append(cen); inert.append(inn)
        bi = int(np.argmin(inert)); lab = labs[bi]
        st = [ari(lab, labs[i]) for i in range(len(labs)) if i != bi]
        sizes = Counter(lab.tolist())
        row = {'k': k, 'inertia': inert[bi], 'stability_ari_mean': float(np.mean(st)), 'stability_ari_min': float(np.min(st)), 'ability_r2': ability_r2(lab, sd), 'min_group_size': min(sizes.values()), 'max_group_size': max(sizes.values()), 'group_sizes': dict(sizes)}
        row['selection_score'] = row['stability_ari_mean'] - 0.85*row['ability_r2']
        metrics.append(row)
        if row['min_group_size'] >= MIN_CLUSTER_SIZE and (best is None or row['selection_score'] > best[0]['selection_score']):
            best = (row, lab, centers[bi])
    if best is None: raise RuntimeError('SRRC (EP-1b) found no usable grouping')
    bestrow, lab_raw, centers_raw = best
    means = {g: sd[lab_raw == g].mean() for g in np.unique(lab_raw)}
    order = sorted(means, key=means.get, reverse=True)
    remap = {old: f'EPn{i+1}' for i, old in enumerate(order)}
    disc_labels = np.array([remap[x] for x in lab_raw])

    # Frozen eval assignment: same residualization/standardization, nearest fitted center, then same remap.
    Fe, se = features(eval_rows)
    Ze = Fe - se[:, None]
    Xe_poly = np.column_stack([np.ones_like(se), se, se**2, se**3])
    Re = Ze - Xe_poly @ beta
    ne = np.linalg.norm(Re, axis=1, keepdims=True); ne[ne < 1e-8] = 1
    Xe = (((Re / ne) - mu) / sigma).astype(np.float32)
    eval_raw = cdist(Xe, centers_raw, 'sqeuclidean').argmin(axis=1)
    eval_labels = np.array([remap[x] for x in eval_raw])
    info = {'selected': bestrow, 'all_candidate_metrics': metrics, 'domains': domains, 'remap': {str(k): v for k, v in remap.items()}, 'grouping_note': 'domain residual shape after cubic score residualization on discovery models x group-discovery items; eval assigned by nearest frozen centroid'}
    return disc_labels, eval_labels, info


def load_sidecar_matrices(models: list[str], selected_items: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], dict]:
    midx = {m: i for i, m in enumerate(models)}; iidx = {it: j for j, it in enumerate(selected_items)}
    pred = np.full((len(models), len(selected_items)), -1, dtype=np.int16)
    resp = np.full((len(models), len(selected_items)), -1, dtype=np.int8)
    kopt = np.full(len(selected_items), -1, dtype=np.int16)
    mismatch = total = kept = 0
    usecols = ['id', 'item', 'target_idx', 'pred_idx', 'option_count', 'resp', 'decode_status']
    for ci, ch in enumerate(pd.read_csv(SIDECAR, usecols=usecols, chunksize=CHUNK)):
        total += len(ch)
        ch = ch[ch['id'].isin(midx) & ch['item'].isin(iidx) & ch['decode_status'].eq('ok')].copy()
        if ch.empty: continue
        for col in ['target_idx', 'pred_idx', 'option_count', 'resp']:
            ch[col] = pd.to_numeric(ch[col], errors='coerce')
        ch = ch.dropna(subset=['target_idx', 'pred_idx', 'option_count', 'resp'])
        bad = (ch['pred_idx'].astype(int).eq(ch['target_idx'].astype(int))) != (ch['resp'].astype(int).eq(1))
        mismatch += int(bad.sum())
        ch = ch[~bad]
        if ch.empty: continue
        mi = ch['id'].map(midx).to_numpy(np.int32); ii = ch['item'].map(iidx).to_numpy(np.int32)
        pred[mi, ii] = ch['pred_idx'].to_numpy(np.int16)
        resp[mi, ii] = ch['resp'].to_numpy(np.int8)
        # option count should be item-level; repeated assignment is fine.
        kopt[ii] = ch['option_count'].to_numpy(np.int16)
        kept += len(ch)
        if ci % 5 == 0: log(f'sidecar chunks={ci+1} total={total:,} kept={kept:,}')
    # Do not require a complete model x item rectangle here. Requiring all models to
    # decode every group-discovery item can collapse the strict MADNESS item set to
    # zero. Missing cells remain -1 and are excluded from same-wrong pair counts by
    # treating them as not jointly wrong.
    item_valid_counts = (pred >= 0).sum(axis=0)
    keep_item_mask = (kopt >= 2) & (item_valid_counts >= 2)
    pred = pred[:, keep_item_mask]; resp = resp[:, keep_item_mask]; kopt = kopt[keep_item_mask]
    kept_items = [it for it, ok in zip(selected_items, keep_item_mask) if ok]
    model_valid_counts_all = (pred >= 0).sum(axis=1)
    min_model_items = max(MADNESS_MIN_VALID_ITEMS_FLOOR, int(0.80 * len(kept_items)))
    keep_model_mask = model_valid_counts_all >= min_model_items
    pred = pred[keep_model_mask]; resp = resp[keep_model_mask]
    kept_models = [m for m, ok in zip(models, keep_model_mask) if ok]
    model_valid_counts = model_valid_counts_all[keep_model_mask]
    info = {
        'sidecar_rows_seen': int(total),
        'sidecar_rows_kept_after_strict_filter': int(kept),
        'dropped_pred_resp_mismatch_cells': int(mismatch),
        'models_retained_for_sparse_same_wrong': len(kept_models),
        'model_retention_rule': f'at least {min_model_items} valid decoded group-discovery items (max(configured floor={MADNESS_MIN_VALID_ITEMS_FLOOR}, 80% of retained decoded items))',
        'group_discovery_decodable_items_retained': len(kept_items),
        'min_valid_group_discovery_items_per_retained_model': int(model_valid_counts.min()) if len(model_valid_counts) else 0,
        'median_valid_group_discovery_items_per_retained_model': float(np.median(model_valid_counts)) if len(model_valid_counts) else 0,
        'models_dropped_for_low_decoded_coverage': int((~keep_model_mask).sum()),
    }
    return pred, resp, kopt, kept_models, kept_items, info


def pairwise_same_wrong_z(pred: np.ndarray, resp: np.ndarray, kopt: np.ndarray) -> np.ndarray:
    n, m = pred.shape
    wrong = (resp == 0).astype(np.float64)
    p = (1.0 / np.maximum(kopt.astype(np.float64) - 1.0, 1.0))[None, :]
    log(f'MADNESS matrix multiply for n={n:,}, items={m:,}')
    exp = (wrong * p) @ wrong.T
    var = (wrong * (p * (1.0 - p))) @ wrong.T
    obs = np.zeros((n, n), dtype=np.float64)
    max_opt = int(kopt.max()) if len(kopt) else 0
    for o in range(max_opt):
        M = ((pred == o) & (resp == 0)).astype(np.float64)
        if M.any(): obs += M @ M.T
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (obs - exp) / np.sqrt(var)
    z[~np.isfinite(z)] = 0
    np.fill_diagonal(z, 0)
    return z.astype(np.float32)


def strict_madness_groups(model_ids: list[str], discovery_ids: list[str], eval_ids: list[str], group_items: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Learn strict MADNESS same-wrong groups and freeze them for evaluation assignment.

    Discovery models are clustered by Ward-style hierarchical clustering on a
    dissimilarity derived from same-wrong z-similarity over group-discovery items.
    Evaluation models are assigned by higher mean z-similarity to frozen groups.
    """
    selected_models = discovery_ids + eval_ids
    pred, resp, kopt, kept_models, kept_items, info = load_sidecar_matrices(selected_models, group_items)
    kept_idx = {m: i for i, m in enumerate(kept_models)}
    disc_pos = np.array([kept_idx[m] for m in discovery_ids if m in kept_idx], dtype=int)
    eval_pos = np.array([kept_idx[m] for m in eval_ids if m in kept_idx], dtype=int)
    disc_kept_models = [m for m in discovery_ids if m in kept_idx]
    eval_kept_models = [m for m in eval_ids if m in kept_idx]
    z_disc = pairwise_same_wrong_z(pred[disc_pos], resp[disc_pos], kopt)
    maxv = float(np.max(z_disc)) if z_disc.size else 1.0
    D = maxv - ((z_disc + z_disc.T)/2); np.fill_diagonal(D, 0)
    link = linkage(squareform(D.astype(np.float64), checks=False), method='ward')
    raw = fcluster(link, t=2, criterion='maxclust')
    counts = Counter(raw.tolist()); largest = counts.most_common(1)[0][0]
    remap = {largest: 'G0'}
    for g in sorted(counts):
        if g != largest: remap[g] = 'G1'
    disc_labels_kept = np.array([remap[x] for x in raw])

    # Frozen eval assignment: compute eval-to-discovery z and assign to group with higher mean z.
    eval_labels_kept = []
    if len(eval_pos):
        all_pos = np.concatenate([disc_pos, eval_pos])
        # For simplicity and exactness, reuse full pairwise on discovery+eval kept subset, then slice eval-vs-discovery.
        z_all = pairwise_same_wrong_z(pred[all_pos], resp[all_pos], kopt)
        nd = len(disc_pos)
        z_ed = z_all[nd:, :nd]
        for row in z_ed:
            means = {g: float(row[disc_labels_kept == g].mean()) for g in ['G0', 'G1']}
            eval_labels_kept.append('G1' if means.get('G1', -np.inf) > means.get('G0', -np.inf) else 'G0')
    disc_label_map = dict(zip(disc_kept_models, disc_labels_kept))
    eval_label_map = dict(zip(eval_kept_models, eval_labels_kept))
    disc_labels = np.array([disc_label_map.get(m, 'UNASSIGNED') for m in discovery_ids])
    eval_labels = np.array([eval_label_map.get(m, 'UNASSIGNED') for m in eval_ids])
    info.update({'method': 'strict same-wrong Ward clustering; eval assignment by higher mean z-similarity to frozen discovery groups', 'discovery_models_assigned': int(np.sum(disc_labels != 'UNASSIGNED')), 'evaluation_models_assigned': int(np.sum(eval_labels != 'UNASSIGNED')), 'discovery_group_counts': dict(Counter(disc_labels.tolist())), 'evaluation_group_counts': dict(Counter(eval_labels.tolist())), 'group_discovery_items_used_for_madness': kept_items})
    return disc_labels, eval_labels, info


def consequence(mat: np.ndarray, eval_rows: np.ndarray, item_idx: np.ndarray, eval_labels: np.ndarray, ref: str, focal: str, dif: pd.DataFrame, name: str, n_draws: int = 500, m: int = 50) -> list[dict]:
    """Compute held-out focal-minus-reference accuracy gaps for DIF-defined item pools.

    The full-pool estimate uses all available held-out responses in the item pool;
    draw summaries repeatedly sample up to m items to show gap variability.
    """
    rng = np.random.default_rng(SEED + 909)
    dif = dif.copy()
    pools = {
        'All DIF-test items': set(item_idx.tolist()),
        'Invariant A': set(dif.loc[dif.ets_cat == 'A', 'item_col'].astype(int)),
        f'B/C favor {focal}': set(dif.loc[(dif.ets_cat.isin(['B','C'])) & (dif.direction == focal), 'item_col'].astype(int)),
        f'B/C favor {ref}': set(dif.loc[(dif.ets_cat.isin(['B','C'])) & (dif.direction == ref), 'item_col'].astype(int)),
        f'C favor {focal}': set(dif.loc[(dif.ets_cat == 'C') & (dif.direction == focal), 'item_col'].astype(int)),
        f'C favor {ref}': set(dif.loc[(dif.ets_cat == 'C') & (dif.direction == ref), 'item_col'].astype(int)),
    }
    rows = []
    valid_rows = eval_rows[np.isin(eval_labels, [ref, focal])]
    valid_labels = eval_labels[np.isin(eval_labels, [ref, focal])]
    for pool, items in pools.items():
        items = np.array(sorted(items), dtype=int)
        if len(items) == 0 or len(valid_rows) == 0:
            rows.append({'analysis': name, 'pool': pool, 'pool_items': int(len(items)), 'eval_models': int(len(valid_rows)), 'gap_pp_full_pool': np.nan, 'draw_m': m, 'draws': 0, 'gap_pp_draw_mean': np.nan, 'gap_pp_draw_sd': np.nan})
            continue
        sub = mat[np.ix_(valid_rows, items)]
        ref_vals = sub[valid_labels == ref]; foc_vals = sub[valid_labels == focal]
        gap = 100 * (foc_vals[foc_vals >= 0].mean() - ref_vals[ref_vals >= 0].mean())
        draw_gaps = []
        if len(items) >= 1:
            draw_m = min(m, len(items))
            for _ in range(n_draws):
                sample = rng.choice(items, size=draw_m, replace=False)
                ss = mat[np.ix_(valid_rows, sample)]
                rv = ss[valid_labels == ref]; fv = ss[valid_labels == focal]
                draw_gaps.append(100 * (fv[fv >= 0].mean() - rv[rv >= 0].mean()))
        rows.append({'analysis': name, 'pool': pool, 'pool_items': int(len(items)), 'eval_models': int(len(valid_rows)), f'n_{ref}': int(np.sum(valid_labels == ref)), f'n_{focal}': int(np.sum(valid_labels == focal)), 'gap_pp_full_pool': float(gap), 'draw_m': int(min(m, len(items))), 'draws': len(draw_gaps), 'gap_pp_draw_mean': float(np.mean(draw_gaps)), 'gap_pp_draw_sd': float(np.std(draw_gaps, ddof=1)) if len(draw_gaps) > 1 else np.nan}
        )
    return rows


def main() -> None:
    t0 = time.time()
    meta, item = load_metadata()
    meta = split_models(meta)
    item = split_items(item)
    models = meta['id'].tolist(); items = item['item'].tolist()
    model_to_row = {m: i for i, m in enumerate(models)}; item_to_col = {it: j for j, it in enumerate(items)}
    item['item_col'] = item['item'].map(item_to_col)
    meta['model_row'] = meta['id'].map(model_to_row)
    item_domains = item.set_index('item_col').sort_index()['domain'].to_numpy()
    group_items = item.loc[item.item_split == 'group_discovery', 'item_col'].to_numpy(int)
    dif_items = item.loc[item.item_split == 'dif_test', 'item_col'].to_numpy(int)
    item_lookup = item[['item_col', 'item', 'domain', 'item_split']].copy()
    log(f'models={len(models):,}; group_items={len(group_items):,}; dif_items={len(dif_items):,}')
    mat = load_response_matrix(models, items)

    # Diagnostics exports.
    meta.to_csv(OUT/'model_split_assignments.csv', index=False)
    item.to_csv(OUT/'item_split_assignments.csv', index=False)
    split_diag = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
        'model_counts': meta.groupby(['model_split','split_family']).size().unstack(fill_value=0).to_dict(),
        'cluster_counts': meta.drop_duplicates(['cluster_proxy']).groupby('model_split').size().to_dict(),
        'item_counts': item.groupby(['item_split','domain']).size().unstack(fill_value=0).to_dict(),
        'total_models': len(models), 'total_items': len(items),
        'discovery_models': int((meta.model_split == 'discovery').sum()), 'evaluation_models': int((meta.model_split == 'evaluation').sum()),
        'group_discovery_items': int(len(group_items)), 'dif_test_items': int(len(dif_items)),
        'common_dif_test_item_set': 'All analyses use this same DIF-test item split for MH-DIF.'
    }

    disc_rows_all = meta.loc[meta.model_split == 'discovery', 'model_row'].to_numpy(int)
    eval_rows_all = meta.loc[meta.model_split == 'evaluation', 'model_row'].to_numpy(int)
    disc_ids = meta.loc[meta.model_split == 'discovery', 'id'].tolist()
    eval_ids = meta.loc[meta.model_split == 'evaluation', 'id'].tolist()

    dif_summaries = []; consequence_rows = []

    # A. Family baseline: external Llama/Qwen labels, no learned grouping.
    fam_disc = meta[(meta.model_split == 'discovery') & (meta.architecture_family.isin(['Llama','Qwen']))].copy()
    fam_eval = meta[(meta.model_split == 'evaluation') & (meta.architecture_family.isin(['Llama','Qwen']))].copy()
    fam_labels_disc = fam_disc['architecture_family'].to_numpy(str)
    fam_anchor, fam_diag, fam_total = mh_dif(mat, fam_disc['model_row'].to_numpy(int), dif_items, fam_labels_disc, 'Llama', 'Qwen', OUT/'family_llama_qwen_mh')
    fam_anchor.merge(item_lookup, on='item_col', how='left').to_csv(OUT/'family_llama_qwen_mh_anchor20_pass_with_items.csv', index=False)
    dif_summaries.append(summarize_dif('family_llama_qwen', fam_anchor, fam_diag, item_lookup))
    consequence_rows += consequence(mat, fam_eval['model_row'].to_numpy(int), dif_items, fam_eval['architecture_family'].to_numpy(str), 'Llama', 'Qwen', fam_anchor, 'family_llama_qwen')

    # B. Strict MADNESS same-wrong grouping: discovery-only Ward-style clustering on same-wrong z-similarity.
    madness_disc_labels_all, madness_eval_labels_all, madness_info = strict_madness_groups(models, disc_ids, eval_ids, item.loc[item.item_split == 'group_discovery', 'item'].tolist())
    pd.DataFrame({'id': disc_ids, 'model_row': disc_rows_all, 'madness_group': madness_disc_labels_all}).to_csv(OUT/'strict_madness_discovery_group_assignments.csv', index=False)
    pd.DataFrame({'id': eval_ids, 'model_row': eval_rows_all, 'madness_group': madness_eval_labels_all}).to_csv(OUT/'strict_madness_evaluation_group_assignments.csv', index=False)
    m_keep = madness_disc_labels_all != 'UNASSIGNED'
    mad_anchor, mad_diag, mad_total = mh_dif(mat, disc_rows_all[m_keep], dif_items, madness_disc_labels_all[m_keep], 'G0', 'G1', OUT/'strict_madness_ward_mh')
    mad_anchor.merge(item_lookup, on='item_col', how='left').to_csv(OUT/'strict_madness_ward_mh_anchor20_pass_with_items.csv', index=False)
    dif_summaries.append(summarize_dif('strict_madness_ward', mad_anchor, mad_diag, item_lookup))
    e_keep = madness_eval_labels_all != 'UNASSIGNED'
    consequence_rows += consequence(mat, eval_rows_all[e_keep], dif_items, madness_eval_labels_all[e_keep], 'G0', 'G1', mad_anchor, 'strict_madness_ward')

    # C. SRRC (EP-1b).
    ep_disc_labels, ep_eval_labels, ep_info = ep1b_groups(mat, disc_rows_all, eval_rows_all, group_items, item_domains)
    pd.DataFrame({'id': disc_ids, 'model_row': disc_rows_all, 'ep1b_group': ep_disc_labels}).to_csv(OUT/'ep1b_discovery_group_assignments.csv', index=False)
    pd.DataFrame({'id': eval_ids, 'model_row': eval_rows_all, 'ep1b_group': ep_eval_labels}).to_csv(OUT/'ep1b_evaluation_group_assignments.csv', index=False)
    ep_anchor, ep_diag, ep_total = mh_dif(mat, disc_rows_all, dif_items, ep_disc_labels, 'EPn2', 'EPn1', OUT/'ep1b_mh')
    ep_anchor.merge(item_lookup, on='item_col', how='left').to_csv(OUT/'ep1b_mh_anchor20_pass_with_items.csv', index=False)
    dif_summaries.append(summarize_dif('ep1b', ep_anchor, ep_diag, item_lookup))
    consequence_rows += consequence(mat, eval_rows_all, dif_items, ep_eval_labels, 'EPn2', 'EPn1', ep_anchor, 'ep1b')

    # Consequence gap calculation: focal-minus-reference held-out accuracy gaps by DIF-defined item pool.
    # Group summaries and consequence exports.
    group_rows = []
    for name, labels in [('strict_madness_discovery', madness_disc_labels_all), ('strict_madness_evaluation', madness_eval_labels_all), ('ep1b_discovery', ep_disc_labels), ('ep1b_evaluation', ep_eval_labels)]:
        for g, n in Counter(labels.tolist()).items():
            group_rows.append({'analysis_split': name, 'group': g, 'models': int(n)})
    pd.DataFrame(group_rows).to_csv(OUT/'response_derived_group_sizes.csv', index=False)
    pd.DataFrame(dif_summaries).to_csv(OUT/'dif_abc_summary_common_dif_items.csv', index=False)
    pd.DataFrame(consequence_rows).to_csv(OUT/'heldout_evaluation_consequence_table.csv', index=False)

    split_diag['strict_madness'] = madness_info
    split_diag['ep1b'] = ep_info
    split_diag['runtime_seconds'] = time.time() - t0
    (OUT/'split_diagnostics.json').write_text(json.dumps(split_diag, indent=2, sort_keys=True, default=str))

    source_map = {
        'script': str(Path(__file__).resolve()),
        'inputs': {'responses': str(RESP), 'model_metadata': str(MODEL_META), 'item_metadata': str(ITEM_META), 'decoded_sidecar': str(SIDECAR)},
        'outputs': {
            'split_diagnostics': str(OUT/'split_diagnostics.json'),
            'model_split_assignments': str(OUT/'model_split_assignments.csv'),
            'item_split_assignments': str(OUT/'item_split_assignments.csv'),
            'strict_madness_groups': [str(OUT/'strict_madness_discovery_group_assignments.csv'), str(OUT/'strict_madness_evaluation_group_assignments.csv')],
            'ep1b_groups': [str(OUT/'ep1b_discovery_group_assignments.csv'), str(OUT/'ep1b_evaluation_group_assignments.csv')],
            'dif_summary': str(OUT/'dif_abc_summary_common_dif_items.csv'),
            'consequence_table': str(OUT/'heldout_evaluation_consequence_table.csv'),
            'family_item_dif': str(OUT/'family_llama_qwen_mh_anchor20_pass_with_items.csv'),
            'madness_item_dif': str(OUT/'strict_madness_ward_mh_anchor20_pass_with_items.csv'),
            'ep1b_item_dif': str(OUT/'ep1b_mh_anchor20_pass_with_items.csv'),
        },
        'notes': [
            'All MH-DIF analyses use discovery models and the common domain-stratified DIF-test item set.',
            'Response-derived groups are learned only from discovery models x group-discovery items.',
            'Strict MADNESS uses only group-discovery items with complete decoded option data in the sidecar.',
            'Evaluation consequences use held-out evaluation models only; response-derived evaluation labels use frozen assignment rules.',
            'Anchor refinement is implemented using the lowest-|delta_MH| 20% items from the total-score pass.',
            'This remains a simplified replication using standard pooled MH SE, not cluster-robust PH/Rao-Scott variance.'
        ]
    }
    (OUT/'SOURCE_MAP.json').write_text(json.dumps(source_map, indent=2, sort_keys=True))
    log(f'done: {OUT}; runtime={time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
