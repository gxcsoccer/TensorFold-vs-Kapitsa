"""Aggregate raw result JSON into Markdown tables for the report."""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def load(pattern):
    files = sorted(glob.glob(os.path.join(RES, pattern)))
    return json.load(open(files[-1])) if files else None


def pick(phases, key, *path, default=None):
    cur = phases.get(key)
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return default if cur is None else cur


def merge(label_base):
    """Merge the core run and the features re-run for the same label."""
    core = load(f"*{label_base}.json")
    feat = load(f"*{label_base}-feat.json")
    if not core:
        return None
    out = {**core}
    if feat:
        out["phases"] = {**core.get("phases", {}), **feat.get("phases", {})}
    return out


def fmt(v, nd=1):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def perf_table(runs):
    rows = []
    for name, d in runs.items():
        if not d:
            continue
        p = d["phases"]
        dec = p.get("decode", {})
        rows.append({
            "engine": name,
            "load_s": pick(p, "startup", "load_s"),
            "mem_idle_gib": round((pick(p, "startup", "footprint_idle", "phys_footprint") or 0) / 2**30, 1),
            "mem_peak_gib": round((d.get("meta", {}).get("footprint_end", {}) or {}).get("phys_footprint_peak", 0) / 2**30, 1),
            "decode_greedy": pick(dec, "greedy", "decode_tps_median"),
            "decode_sampled": pick(dec, "sampled", "decode_tps_median"),
            "ttft_ms": pick(dec, "greedy", "ttft_ms_median"),
            "prefill_4k": pick(p, "prefill", "4096", "prefill_tps"),
            "prefill_16k": pick(p, "prefill", "16384", "prefill_tps"),
            "cached_ttft_ms": st.median([t["ttft_ms"] for t in p.get("cached", {}).get("turns", []) if t.get("ttft_ms")])
            if p.get("cached", {}).get("turns") else None,
            "conc2": pick(p, "concurrency", "2", "aggregate_tps"),
            "conc4": pick(p, "concurrency", "4", "aggregate_tps"),
            "conc8": pick(p, "concurrency", "8", "aggregate_tps"),
            "conc8_ttft_ms": pick(p, "concurrency", "8", "ttft_ms_median"),
            "longctx_ttft_s": pick(p, "longctx", "ttft_s"),
            "longctx_ok": pick(p, "longctx", "correct"),
            "quality": f"{p.get('quality', {}).get('passed')}/{p.get('quality', {}).get('total')}",
            "fidelity": p.get("fidelity", {}).get("combined_sha"),
        })
    return rows


def md_table(rows, cols):
    if not rows:
        return "_no data_"
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def main():
    print("# 机器可读汇总（由 `analysis/aggregate.py` 生成）\n")
    print("> `image` 列反映各引擎的**默认** server flags：TensorFold 需要 `--vision` 才接受图像，"
          "Kapitsa 默认开启；TF 0.6.6 在本机因缺 torchvision 无法启用。详见 README §5。\n")
    runs = {
        "Kapitsa · 27B": merge("kapitsa-27b"),
        "TF 1.2.2 · 27B": merge("tf12-27b"),
        "TF 0.6.6 · 27B": merge("tf06-27b"),
        "Kapitsa · 35B": merge("kapitsa-35b"),
        "TF 0.6.6 · 35B": merge("tf06-35b"),
    }
    rows = perf_table(runs)
    cols = ["engine", "load_s", "mem_idle_gib", "mem_peak_gib", "decode_greedy", "decode_sampled",
            "ttft_ms", "prefill_4k", "prefill_16k", "cached_ttft_ms", "conc2", "conc4", "conc8",
            "conc8_ttft_ms", "longctx_ttft_s", "longctx_ok", "quality", "fidelity"]
    print("## Performance\n")
    print(md_table(rows, cols))

    print("\n## Feature checks\n")
    feat_rows = []
    for name, d in runs.items():
        if not d:
            continue
        p = d["phases"]
        ep = p.get("endpoints", {})
        feat_rows.append({
            "engine": name,
            "models": ep.get("/v1/models", {}).get("status"),
            "metrics": ep.get("/metrics", {}).get("status"),
            "completions": ep.get("/v1/completions", {}).get("status"),
            "responses": ep.get("/v1/responses", {}).get("status"),
            "messages": ep.get("/v1/messages", {}).get("status"),
            "image": ep.get("image", {}).get("status"),
            "tools": p.get("tools", {}).get("nonstream", {}).get("pass"),
            "tools_stream": p.get("tools", {}).get("stream", {}).get("pass"),
            "json_schema": p.get("structured", {}).get("json_schema", {}).get("valid"),
            "json_object": p.get("structured", {}).get("json_object", {}).get("valid"),
            "thinking_off_tokens": pick(p, "thinking", "off", "tokens"),
            "thinking_on_tokens": pick(p, "thinking", "on", "tokens"),
            "cancel_ok": p.get("cancel", {}).get("next_request_ok"),
        })
    print(md_table(feat_rows, ["engine", "models", "metrics", "completions", "responses", "messages",
                               "image", "tools", "tools_stream", "json_schema", "json_object",
                               "thinking_off_tokens", "thinking_on_tokens", "cancel_ok"]))

    print("\n## Probes\n")
    prows = []
    for f in sorted(glob.glob(os.path.join(RES, "probe-*.json"))):
        d = json.load(open(f))
        prows.append({
            "label": d.get("label"),
            "load_s": d.get("load_s"),
            "prefill_4k": (d.get("prefill") or {}).get("prefill_tps"),
            "decode_tps": (d.get("decode") or {}).get("tps_median"),
            "mem_peak_gib": round(((d.get("footprint_peak") or {}).get("phys_footprint_peak") or 0) / 2**30, 1),
            "cmd": " ".join(d.get("cmd", [])[1:])[:110],
        })
    print(md_table(prows, ["label", "load_s", "prefill_4k", "decode_tps", "mem_peak_gib", "cmd"]))

    print("\n## Quality per task\n")
    ids = None
    for name, d in runs.items():
        if not d:
            continue
        q = d["phases"].get("quality", {})
        ids = [r["id"] for r in q.get("rows", [])] or ids
    if ids:
        print("| engine | " + " | ".join(ids) + " |")
        print("|" + "|".join(["---"] * (len(ids) + 1)) + "|")
        for name, d in runs.items():
            if not d:
                continue
            q = {r["id"]: r["ok"] for r in d["phases"].get("quality", {}).get("rows", [])}
            print("| " + name + " | " + " | ".join("✅" if q.get(i) else "❌" for i in ids) + " |")


if __name__ == "__main__":
    main()
