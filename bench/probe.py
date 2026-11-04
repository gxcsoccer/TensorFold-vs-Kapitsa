"""Fast A/B probe: startup, one 4K cold prefill and N short decodes.

Used for ablations where only the relative change matters (drafter on/off,
prompt rows, slots, cache). Full protocol numbers come from run.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchlib import Server, GpuLock, stream_chat, cold_prompt, now
import run as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--engine", required=True, choices=["kapitsa", "tf12", "tf06"])
    ap.add_argument("--model", default="27b")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--slots", type=int)
    ap.add_argument("--extra", default="")
    ap.add_argument("--env", default="")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=256)
    args = ap.parse_args()

    R.resolve_models()
    spec = dict(engine=args.engine, model=args.model, ctx=args.ctx, slots=args.slots,
                port=8300 + (abs(hash(args.label)) % 60), name="local-model",
                extra=args.extra.split() if args.extra else [],
                env=dict(kv.split("=", 1) for kv in args.env.split(",") if "=" in kv))
    srv = R.build_server(args.label, spec)
    out = {"label": args.label, "spec": {**spec, "model": args.model}, "cmd": srv.cmd,
           "env": spec["env"], "ts": now()}
    with GpuLock():
        try:
            srv.start()
            out["load_s"] = round(srv.t_ready - srv.t_start, 2)
            out["footprint"] = srv.footprint()
            for _ in range(2):
                stream_chat(srv.port, R.chat_body([{"role": "user", "content": "Say ready."}], 8))
            # 4K cold prefill
            text = cold_prompt(4096, uuid.uuid4().hex[:10])
            t0 = time.perf_counter()
            r = stream_chat(srv.port, R.chat_body([{"role": "user", "content": text}], 8, 0.0))
            pt = (r.get("usage") or {}).get("prompt_tokens")
            out["prefill"] = {"prompt_tokens": pt,
                              "ttft_s": round(r["ttft"], 3) if r.get("ttft") else None,
                              "prefill_tps": round(pt / r["ttft"], 1) if pt and r.get("ttft") else None}
            # decodes
            rates, ttfts = [], []
            prompt = "Write a Python function that merges two sorted lists. Be concise."
            for _ in range(args.runs):
                rr = stream_chat(srv.port, R.chat_body([{"role": "user", "content": prompt}], args.tokens, 0.0))
                v = R.tps(rr)
                if v:
                    rates.append(v)
                if rr.get("ttft"):
                    ttfts.append(rr["ttft"])
            out["decode"] = {"n": len(rates),
                             "tps_median": round(st.median(rates), 2) if rates else None,
                             "ttft_ms_median": round(st.median(ttfts) * 1000, 1) if ttfts else None,
                             "tokens": args.tokens}
            # drafter evidence from the log tail
            try:
                log = open(srv.log_path, errors="replace").read()
                out["log_lines"] = [ln for ln in log.splitlines()
                                    if re.search(r"draft|Drafter|DFlash|MTP|dflash|prompt rows|rows when|speculat", ln, re.I)][-8:]
            except Exception:  # noqa: BLE001
                pass
        finally:
            out["footprint_peak"] = srv.footprint()
            srv.stop()
            os.makedirs(os.path.join(R.ROOT, "results"), exist_ok=True)
            path = os.path.join(R.ROOT, "results", f"probe-{args.label}.json")
            with open(path, "w") as f:
                json.dump(out, f, indent=1)
            print(json.dumps({k: v for k, v in out.items() if k != "log_lines"}, indent=1))
            print("LOG:", *out.get("log_lines", []), sep="\n  ")


if __name__ == "__main__":
    main()
