#!/usr/bin/env python3
"""Regenerate manuscript Figure 1 from preserved consequence-gap outputs.

The plot uses held-out draw means (`gap_pp_draw_mean`) as points and draw
standard deviations (`gap_pp_draw_sd`) as error bars, matching the manuscript
figure rather than the full-pool diagnostic column.
"""
from __future__ import annotations
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PANELS = [
    {
        'analysis': 'family_llama_qwen',
        'title': 'Family baseline\n(Qwen − Llama)',
        'pools': {
            'All': 'All DIF-test items',
            'Invariant': 'Invariant A',
            'B/C focal': 'B/C favor Qwen',
            'B/C reference': 'B/C favor Llama',
        },
    },
    {
        'analysis': 'strict_madness_ward',
        'title': 'Strict MADNESS\n(G1 − G0)',
        'pools': {
            'All': 'All DIF-test items',
            'Invariant': 'Invariant A',
            'B/C focal': 'B/C favor G1',
            'B/C reference': 'B/C favor G0',
        },
    },
    {
        'analysis': 'ep1b',
        'title': 'SRRC\n(EPn1 − EPn2)',
        'pools': {
            'All': 'All DIF-test items',
            'Invariant': 'Invariant A',
            'B/C focal': 'B/C favor EPn1',
            'B/C reference': 'B/C favor EPn2',
        },
    },
]

POINT_COLOR = '#2f4b7c'
ERR_COLOR = '#4c78a8'
REF_COLOR = '#999999'
ZERO_COLOR = '#333333'


def get_row(df: pd.DataFrame, analysis: str, pool: str) -> pd.Series:
    """Return the unique consequence row for an analysis/pool pair."""
    rows = df[(df['analysis'] == analysis) & (df['pool'] == pool)]
    if len(rows) != 1:
        raise ValueError(f'Expected one row for analysis={analysis!r}, pool={pool!r}; found {len(rows)}')
    return rows.iloc[0]


def main() -> None:
    ap = argparse.ArgumentParser(description='Plot manuscript consequence-gap Figure 1.')
    ap.add_argument('--input', type=Path, default=Path('outputs/final/heldout_evaluation_consequence_table.csv'))
    ap.add_argument('--out-dir', type=Path, default=Path('figures'))
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2), sharey=True)
    x = np.arange(4)
    xlabels = ['All', 'Invariant', 'B/C\nfocal', 'B/C\nreference']

    all_values = []
    for ax, panel in zip(axes, PANELS):
        ys = []
        yerrs = []
        for label, pool in panel['pools'].items():
            row = get_row(df, panel['analysis'], pool)
            ys.append(float(row['gap_pp_draw_mean']))
            yerrs.append(float(row['gap_pp_draw_sd']))
        all_values.extend(ys)

        ax.axhline(0, color=ZERO_COLOR, linewidth=0.8)
        ax.axhline(ys[0], color=REF_COLOR, linewidth=0.9, linestyle='--', zorder=0)
        ax.errorbar(
            x,
            ys,
            yerr=yerrs,
            fmt='o',
            color=POINT_COLOR,
            ecolor=ERR_COLOR,
            elinewidth=1.2,
            capsize=3,
            markersize=5,
        )
        ax.set_title(panel['title'], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.grid(axis='y', color='#dddddd', linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('Held-out accuracy gap (pp)')
    finite = np.array([v for v in all_values if np.isfinite(v)])
    if len(finite):
        pad = max(2.0, 0.15 * (finite.max() - finite.min()))
        axes[0].set_ylim(finite.min() - pad, finite.max() + pad)

    fig.tight_layout(w_pad=1.0)
    fig.savefig(args.out_dir / 'consequence_gap.png', dpi=300)
    fig.savefig(args.out_dir / 'consequence_gap.pdf')
    print(f'wrote {args.out_dir / "consequence_gap.png"} and {args.out_dir / "consequence_gap.pdf"}')


if __name__ == '__main__':
    main()
