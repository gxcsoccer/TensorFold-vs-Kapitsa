"""Render the comparison figures from raw result files."""
from __future__ import annotations

import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "analysis", "figs")
os.makedirs(FIGS, exist_ok=True)

C_K = "#2e7d32"
C_T12 = "#1565c0"
C_T06 = "#ef6c00"


def last(pattern):
    files = sorted(glob.glob(os.path.join(RES, pattern)))
    return json.load(open(files[-1])) if files else None


def probe(name):
    f = os.path.join(RES, f"probe-{name}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def cached_ttft(d):
    turns = d["phases"].get("cached", {}).get("turns", [])
    vals = [t["ttft_ms"] for t in turns if t.get("ttft_ms")]
    return st.median(vals) if vals else None


def cold_ttft(d):
    return d["phases"].get("cached", {}).get("first", {}).get("ttft_ms")


def fig_decode():
    rows = [
        ("27B Kapitsa\nplain", probe("kapitsa-27b-plain-s1")["decode"]["tps_median"], C_K),
        ("27B Kapitsa\nDFlash2", probe("kapitsa-27b-dflash")["decode"]["tps_median"], C_K),
        ("27B TF 1.2.2\nplain", last("*tf12-27b.json")["phases"]["decode"]["greedy"]["decode_tps_median"], C_T12),
        ("27B TF 1.2.2\nDFlash2", probe("tf12-27b-drafter")["decode"]["tps_median"], C_T12),
        ("27B TF 0.6\nplain", probe("tf06-27b-nodrafts")["decode"]["tps_median"], C_T06),
        ("27B TF 0.6\nauto DFlash2", last("*tf06-27b.json")["phases"]["decode"]["greedy"]["decode_tps_median"], C_T06),
        ("35B Kapitsa\nplain", last("*kapitsa-35b.json")["phases"]["decode"]["greedy"]["decode_tps_median"], C_K),
        ("35B TF 0.6\ndefault", last("*tf06-35b.json")["phases"]["decode"]["greedy"]["decode_tps_median"], C_T06),
        ("35B TF 0.6\n--no-drafts", probe("tf06-35b-nodrafts")["decode"]["tps_median"], C_T06),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = np.arange(len(rows))
    bars = ax.bar(x, [r[1] for r in rows], color=[r[2] for r in rows], width=0.68)
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{r[1]:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylabel("decode tok/s (greedy, single stream)")
    ax.set_title("Single-stream decode speed — same weights, engine only")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "decode.png"), dpi=160)
    plt.close(fig)


def fig_concurrency():
    k27 = last("*kapitsa-27b.json")
    t12 = last("*tf12-27b.json")
    t12p1 = last("*tf12-27b-p1.json")
    t06 = last("*tf06-27b.json")
    k35 = last("*kapitsa-35b.json")
    t0635 = last("*tf06-35b.json")

    def series(d, single=None):
        dec = single if single is not None else d["phases"]["decode"]["greedy"]["decode_tps_median"]
        c = d["phases"]["concurrency"]
        return [dec, c["2"]["aggregate_tps"], c["4"]["aggregate_tps"], c["8"]["aggregate_tps"]]

    xs = [1, 2, 4, 8]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax1.plot(xs, series(k27), marker="o", label="Kapitsa 27B", color=C_K)
    ax1.plot(xs, series(t12), marker="o", label="TF 1.2.2 27B", color=C_T12)
    ax1.plot(xs, series(t12p1, single=t12["phases"]["decode"]["greedy"]["decode_tps_median"]), marker="o", linestyle="--", label="TF 1.2.2 27B --parallel 1", color=C_T12)
    ax1.plot(xs, series(t06), marker="o", label="TF 0.6.6 27B (auto drafter)", color=C_T06)
    ax1.set_title("27B: aggregate throughput vs concurrent requests")
    ax1.set_xlabel("concurrent requests")
    ax1.set_ylabel("aggregate decode tok/s")
    ax1.set_xticks(xs)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=7.5)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(xs, series(k35), marker="o", label="Kapitsa 35B", color=C_K)
    ax2.plot(xs, series(t0635), marker="o", label="TF 0.6.6 35B", color=C_T06)
    ax2.set_title("35B MoE: aggregate throughput vs concurrent requests")
    ax2.set_xlabel("concurrent requests")
    ax2.set_xticks(xs)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "concurrency.png"), dpi=160)
    plt.close(fig)


def fig_prefill():
    groups = [
        ("Kapitsa 27B", last("*kapitsa-27b.json"), C_K),
        ("TF 1.2.2 27B", last("*tf12-27b.json"), C_T12),
        ("TF 0.6.6 27B", last("*tf06-27b.json"), C_T06),
        ("Kapitsa 35B", last("*kapitsa-35b.json"), C_K),
        ("TF 0.6.6 35B", last("*tf06-35b.json"), C_T06),
    ]
    x = np.arange(len(groups))
    w = 0.36
    v4 = [g[1]["phases"]["prefill"]["4096"]["prefill_tps"] for g in groups]
    v16 = [g[1]["phases"]["prefill"]["16384"]["prefill_tps"] for g in groups]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    b1 = ax.bar(x - w / 2, v4, w, label="4K prompt", color="#455a64")
    b2 = ax.bar(x + w / 2, v16, w, label="16K prompt", color="#90a4ae")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 12, f"{b.get_height():.0f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel("prefill tok/s (prompt tokens / TTFT)")
    ax.set_title("Prefill throughput")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "prefill.png"), dpi=160)
    plt.close(fig)


def fig_cache():
    groups = [
        ("Kapitsa 27B", last("*kapitsa-27b.json"), C_K),
        ("TF 1.2.2 27B", last("*tf12-27b.json"), C_T12),
        ("TF 0.6.6 27B", last("*tf06-27b.json"), C_T06),
        ("Kapitsa 35B", last("*kapitsa-35b.json"), C_K),
        ("TF 0.6.6 35B", last("*tf06-35b.json"), C_T06),
    ]
    x = np.arange(len(groups))
    w = 0.36
    cold = [cold_ttft(g[1]) for g in groups]
    warm = [cached_ttft(g[1]) for g in groups]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    b1 = ax.bar(x - w / 2, cold, w, label="cold 4K prompt", color="#c62828")
    b2 = ax.bar(x + w / 2, warm, w, label="cached follow-up", color="#2e7d32")
    ax.set_yscale("log")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.12, f"{b.get_height():.0f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel("TTFT (ms, log scale)")
    ax.set_title("Time to first token: cold prefill vs cached follow-up")
    ax.legend()
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "cache.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_decode()
    fig_concurrency()
    fig_prefill()
    fig_cache()
    print("wrote", os.listdir(FIGS))
