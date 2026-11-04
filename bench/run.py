"""Run one engine configuration through the evaluation protocol.

Usage:
  python3 bench/run.py --label kapitsa-27b --engine kapitsa --model 27b --phases core,quality
  python3 bench/run.py --list
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import platform
import random
import re
import statistics as st
import subprocess
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchlib import (Server, GpuLock, stream_chat, post_chat, http_json, http_raw,
                      cold_prompt, build_corpus_block, now)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(ROOT, "logs")

HF = os.path.expanduser("~/.cache/huggingface/hub")
MODELS = {
    "35b": f"{HF}/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec",
    "27b": None,   # resolved after download
    "4b": os.path.expanduser("~/.cache/kapitsa/models/qwen3.5-4b-4bit"),
}

TF12 = "tensorfold"
TF06 = "/opt/homebrew/opt/tensorfold@0.6/bin/tensorfold"


def resolve_models():
    if MODELS["27b"] is None:
        base = f"{HF}/models--TensorFold--Qwen3.8-27B-MLX-4bit"
        if os.path.isdir(base):
            ref = os.path.join(base, "refs", "main")
            sha = open(ref).read().strip() if os.path.exists(ref) else None
            if sha:
                MODELS["27b"] = os.path.join(base, "snapshots", sha)
        if not MODELS["27b"]:
            snaps = os.path.join(base, "snapshots")
            if os.path.isdir(snaps):
                MODELS["27b"] = os.path.join(snaps, sorted(os.listdir(snaps))[0])
    return MODELS


# ---------------------------------------------------------------- specs
SPECS = {
    "kapitsa-35b": dict(engine="kapitsa", model="35b", ctx=32768, slots=4, port=8101),
    "tf06-35b": dict(engine="tf06", model="35b", ctx=32768, port=8102),
    "kapitsa-27b": dict(engine="kapitsa", model="27b", ctx=32768, slots=4, port=8103),
    "tf12-27b": dict(engine="tf12", model="27b", ctx=32768, port=8104),
    "tf06-27b": dict(engine="tf06", model="27b", ctx=32768, port=8105),
}


def build_server(label: str, spec: dict) -> Server:
    engine, model = spec["engine"], MODELS[spec["model"]]
    ctx = spec.get("ctx", 32768)
    port = spec["port"]
    name = spec.get("name", "local-model")
    extra = spec.get("extra", [])
    if engine == "kapitsa":
        cmd = ["kapitsa", "serve", "--model", model, "--private", "--port", str(port),
               "--context", str(ctx), "--model-id", name]
        if spec.get("slots"):
            cmd += ["--slots", str(spec["slots"])]
        env = dict(spec.get("env") or {})
    elif engine == "tf12":
        cmd = [TF12, "serve", model, "--port", str(port), "--name", name, "--context", str(ctx)]
        env = dict(spec.get("env") or {})
        if spec.get("thinking_off", True):
            cmd.append("--no-thinking")
    elif engine == "tf06":
        cmd = [TF06, "serve", model, "--port", str(port), "--name", name, "--context", str(ctx)]
        env = dict(spec.get("env") or {})
        if spec.get("thinking_off", True):
            cmd.append("--no-thinking")
    else:
        raise ValueError(engine)
    cmd += [str(x) for x in extra]
    return Server(label=label, cmd=cmd, port=port, env=env, model_id=name)


# ---------------------------------------------------------------- helpers
THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}
THINK_ON = {"chat_template_kwargs": {"enable_thinking": True}}


def chat_body(messages, max_tokens=256, temperature=0.0, extra=None, stream=True):
    body = {"model": "local-model", "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": stream}
    body.update(THINK_OFF)
    if stream:
        body["stream_options"] = {"include_usage": True}
    if extra:
        body.update(extra)
    return body


def tps(res):
    """Decode tok/s from exact usage tokens over the first->last token interval.

    SSE events are not guaranteed to be one token each (Kapitsa groups some), so
    the event count is not a token count; the server-reported completion_tokens
    is.  Using the first and last event timestamps excludes prefill and queueing.
    """
    u = res.get("usage") or {}
    n = u.get("completion_tokens")
    ev = res.get("event_times") or []
    if not n or n < 2 or len(ev) < 2:
        return None
    dt = ev[-1] - ev[0]
    return (n - 1) / dt if dt > 0 else None


def summarize_runs(runs):
    ok = [r for r in runs if r.get("ok") and r.get("usage")]
    ttfts = [r["ttft"] for r in ok if r.get("ttft")]
    rates = [x for r in ok if (x := tps(r)) is not None]
    totals = [x for r in ok if (x := tps(r)) is not None]
    return {
        "n": len(runs), "ok": len(ok),
        "ttft_ms_median": round(st.median(ttfts) * 1000, 1) if ttfts else None,
        "ttft_ms_min": round(min(ttfts) * 1000, 1) if ttfts else None,
        "decode_tps_median": round(st.median(rates), 2) if rates else None,
        "decode_tps_min": round(min(rates), 2) if rates else None,
        "decode_tps_max": round(max(rates), 2) if rates else None,
        "total_tps_median": round(st.median(totals), 2) if totals else None,
        "completion_tokens_median": st.median([r["usage"]["completion_tokens"] for r in ok]) if ok else None,
        "prompt_tokens_median": st.median([r["usage"]["prompt_tokens"] for r in ok]) if ok else None,
        "server_tps_median": round(st.median([r["server"]["tokens_per_second"] for r in ok
                                              if r.get("server") and r["server"].get("tokens_per_second")]), 2)
        if any(r.get("server") and r["server"].get("tokens_per_second") for r in ok) else None,
    }


# ---------------------------------------------------------------- phases
def phase_startup(srv: Server, spec, out):
    fp = srv.footprint()
    st_models, models = http_json(srv.port, "/v1/models")
    st_health, health = http_json(srv.port, "/health")
    out["startup"] = {
        "load_s": round(srv.t_ready - srv.t_start, 2),
        "footprint_idle": fp,
        "models": models,
        "health": health,
        "cmd": srv.cmd,
    }


def phase_warmup(srv, spec, out):
    for _ in range(2):
        stream_chat(srv.port, chat_body([{"role": "user", "content": "Say ready."}], max_tokens=8))
    out["warmup"] = {"done": True, "footprint": srv.footprint()}


def phase_decode(srv, spec, out):
    prompt = ("Write a Python LRU cache class with a capacity and a get/put API, "
              "plus three unit tests. Be concise.")
    res = {"greedy": [], "sampled": []}
    for i in range(5):
        r = stream_chat(srv.port, chat_body([{"role": "user", "content": prompt}], 512, 0.0))
        res["greedy"].append(r)
    for i in range(5):
        r = stream_chat(srv.port, chat_body([{"role": "user", "content": prompt}], 512, 0.6,
                                            {"top_p": 0.95, "top_k": 20}))
        res["sampled"].append(r)
    out["decode"] = {
        "greedy": summarize_runs(res["greedy"]),
        "sampled": summarize_runs(res["sampled"]),
        "greedy_text_sha": hashlib.sha256(res["greedy"][0].get("content", "").encode()).hexdigest()[:16],
        "greedy_sample": (res["greedy"][0].get("content") or "")[:600],
    }


def phase_prefill(srv, spec, out):
    res = {}
    for size in (4096, 16384):
        nonce = uuid.uuid4().hex[:12]
        text = cold_prompt(size, nonce)
        r = stream_chat(srv.port, chat_body([{"role": "user", "content": text}], 8, 0.0))
        pt = (r.get("usage") or {}).get("prompt_tokens")
        res[f"{size}"] = {
            "prompt_tokens": pt,
            "ttft_s": round(r["ttft"], 4) if r.get("ttft") else None,
            "prefill_tps": round(pt / r["ttft"], 2) if pt and r.get("ttft") else None,
            "ok": r.get("ok"),
        }
        time.sleep(0.5)
    out["prefill"] = res


def phase_cached(srv, spec, out):
    nonce = uuid.uuid4().hex[:10]
    text = cold_prompt(4096, nonce)
    msgs = [{"role": "user", "content": text}]
    first = stream_chat(srv.port, chat_body(msgs, 32, 0.0))
    msgs.append({"role": "assistant", "content": first.get("content") or "OK"})
    turns = []
    for i in range(3):
        msgs.append({"role": "user", "content": f"Follow-up {i+1}: name one word from the passage."})
        r = stream_chat(srv.port, chat_body(msgs, 16, 0.0))
        u = r.get("usage") or {}
        turns.append({"ttft_ms": round(r["ttft"] * 1000, 1) if r.get("ttft") else None,
                      "prompt_tokens": u.get("prompt_tokens"),
                      "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                      "ok": r.get("ok")})
        msgs.append({"role": "assistant", "content": r.get("content") or "OK"})
        time.sleep(0.3)
    out["cached"] = {"first": {"ttft_ms": round(first["ttft"] * 1000, 1) if first.get("ttft") else None,
                               "prompt_tokens": (first.get("usage") or {}).get("prompt_tokens")},
                     "turns": turns}


def phase_concurrency(srv, spec, out):
    prompt = "Explain the difference between a mutex and a semaphore in two sentences."
    res = {}
    for n in (2, 4, 8):
        payloads = [chat_body([{"role": "user", "content": prompt}], 256, 0.0) for _ in range(n)]
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=n) as ex:
            runs = list(ex.map(lambda b: stream_chat(srv.port, b), payloads))
        wall = time.perf_counter() - t0
        ok = [r for r in runs if r.get("ok")]
        toks = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in ok)
        res[str(n)] = {
            "ok": len(ok), "wall_s": round(wall, 3),
            "aggregate_tps": round(toks / wall, 2) if wall else None,
            "ttft_ms_median": round(st.median([r["ttft"] for r in ok]) * 1000, 1) if ok else None,
            "per_stream_tps_median": round(st.median([tps(r) for r in ok if tps(r)]), 2) if ok else None,
            "completion_tokens": toks,
        }
        srv.wait_quiet(1.0)
    out["concurrency"] = res


def phase_longctx(srv, spec, out):
    ctx = spec.get("ctx", 32768)
    target = 24576 if ctx >= 32768 else max(4096, ctx // 2)
    block = build_corpus_block(int(target * 3.9))
    needle = "The access code is ZX-4471-QK."
    insert_at = int(len(block) * 0.6)
    doc = block[:insert_at] + "\n\n" + needle + "\n\n" + block[insert_at:]
    prompt = f"{doc}\n\nQuestion: what is the access code? Answer with the code only."
    r = stream_chat(srv.port, chat_body([{"role": "user", "content": prompt}], 32, 0.0), timeout=900)
    content = (r.get("content") or "").strip()
    out["longctx"] = {
        "target_tokens": target,
        "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
        "ttft_s": round(r["ttft"], 3) if r.get("ttft") else None,
        "ok": r.get("ok"),
        "answer": content[:120],
        "correct": "ZX-4471-QK" in content.upper(),
    }


TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "add_numbers", "description": "Add two integers",
     "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"]}}},
]


def phase_tools(srv, spec, out):
    res = {}
    msgs = [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]
    st_code, r = post_chat(srv.port, chat_body(msgs, 256, 0.0, {"tools": TOOLS, "tool_choice": "auto"}, stream=False))
    r = r if isinstance(r, dict) else {}
    tc = ((r.get("choices") or [{}])[0].get("message") or {}).get("tool_calls")
    res["nonstream"] = {"status": (r or {}).get("status"), "tool_calls": tc,
                        "pass": bool(tc and tc[0].get("function", {}).get("name") in ("get_weather", "add_numbers"))}
    # round trip
    if tc:
        msgs2 = msgs + [{"role": "assistant", "content": "", "tool_calls": tc},
                        {"role": "tool", "tool_call_id": tc[0].get("id", "c1"), "content": '{"temperature": 18, "unit": "C"}'}]
        st2, r2 = post_chat(srv.port, chat_body(msgs2, 128, 0.0, {"tools": TOOLS}, stream=False))
        r2 = r2 if isinstance(r2, dict) else {}
        res["roundtrip"] = {"status": st2,
                            "content": ((((r2 or {}).get("choices") or [{}])[0].get("message") or {}).get("content") or "")[:200],
                            "pass": bool(r2.get("choices"))}
    # streaming tools
    sr = stream_chat(srv.port, chat_body(msgs, 256, 0.0, {"tools": TOOLS, "tool_choice": "auto"}))
    res["stream"] = {"ok": sr.get("ok"), "tool_call_chunks": len(sr.get("tool_calls") or []),
                     "pass": bool(sr.get("tool_calls"))}
    out["tools"] = res


def phase_structured(srv, spec, out):
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                                               "city": {"type": "string"}},
              "required": ["name", "age", "city"], "additionalProperties": False}
    body = chat_body([{"role": "user", "content": "Invent one person: name, age, city."}], 200, 0.0,
                     {"response_format": {"type": "json_schema", "json_schema": {"name": "person", "schema": schema, "strict": True}}},
                     stream=False)
    st_code, r = post_chat(srv.port, body)
    r = r if isinstance(r, dict) else {}
    content = ""
    valid = False
    if r.get("choices"):
        content = ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        try:
            obj = json.loads(content)
            valid = isinstance(obj.get("name"), str) and isinstance(obj.get("age"), int) and isinstance(obj.get("city"), str)
        except Exception:  # noqa: BLE001
            valid = False
    # json_object mode
    st2, r2 = post_chat(srv.port, chat_body([{"role": "user", "content": "Return JSON: {\"ok\": true}"}], 100, 0.0,
                                        {"response_format": {"type": "json_object"}}, stream=False))
    r2 = r2 if isinstance(r2, dict) else {}
    valid2 = False
    try:
        json.loads(((r2.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        valid2 = bool(r2.get("choices"))
    except Exception:  # noqa: BLE001
        valid2 = False
    out["structured"] = {"json_schema": {"status": st_code, "valid": valid,
                                         "excerpt": content[:200]},
                         "json_object": {"status": st2, "valid": valid2}}


def phase_thinking(srv, spec, out):
    q = "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much is the ball? Answer with the amount only."
    off = stream_chat(srv.port, chat_body([{"role": "user", "content": q}], 256, 0.0))
    on_body = chat_body([{"role": "user", "content": q}], 512, 0.0)
    on_body.update(THINK_ON); on_body.pop("reasoning_effort", None)
    on = stream_chat(srv.port, on_body)
    def amt(s):
        m = re.search(r"0?\.0?5|\$?0\.05", s or "")
        return bool(m)
    out["thinking"] = {
        "off": {"tokens": (off.get("usage") or {}).get("completion_tokens"),
                "ttft_ms": round(off["ttft"] * 1000, 1) if off.get("ttft") else None,
                "total_s": round(off["t_end"] - off["t_send"], 3),
                "reasoning_chars": len(off.get("reasoning") or ""),
                "answer": (off.get("content") or "")[:80], "correct": amt(off.get("content"))},
        "on": {"tokens": (on.get("usage") or {}).get("completion_tokens"),
               "ttft_ms": round(on["ttft"] * 1000, 1) if on.get("ttft") else None,
               "total_s": round(on["t_end"] - on["t_send"], 3),
               "reasoning_chars": len(on.get("reasoning") or ""),
               "answer": (on.get("content") or "")[:80], "correct": amt(on.get("content"))},
    }


def phase_errors(srv, spec, out):
    cases = {
        "empty_messages": {"model": "local-model", "messages": [], "max_tokens": 8},
        "unknown_model": {"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
        "no_messages": {"model": "local-model", "max_tokens": 8},
        "huge_max_tokens": {"model": "local-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1000000},
        "bad_role": {"model": "local-model", "messages": [{"role": "wizard", "content": "hi"}], "max_tokens": 8},
    }
    res = {}
    for name, body in cases.items():
        st, resp = http_json(srv.port, "/v1/chat/completions", body, timeout=120)
        excerpt = ""
        if isinstance(resp, dict):
            excerpt = ((resp.get("error") or {}).get("message") or json.dumps(resp)[:160])
        else:
            excerpt = str(resp)[:160]
        res[name] = {"status": st, "error": excerpt[:200]}
    # malformed JSON
    import urllib.request, urllib.error
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{srv.port}/v1/chat/completions",
                                     data=b"{not json", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            res["malformed_json"] = {"status": r.status}
    except urllib.error.HTTPError as e:
        res["malformed_json"] = {"status": e.code, "error": e.read().decode(errors="replace")[:120]}
    except Exception as e:  # noqa: BLE001
        res["malformed_json"] = {"status": None, "error": str(e)[:120]}
    out["errors"] = res


def phase_cancel(srv, spec, out):
    import http.client
    body = json.dumps(chat_body([{"role": "user", "content": "Write 2000 words about databases."}], 512, 0.0)).encode()
    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=60)
    conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    n = 0
    for _ in resp:
        n += 1
        if n >= 5:
            break
    conn.close()
    time.sleep(0.5)
    st, health = http_json(srv.port, "/health", timeout=10)
    r = stream_chat(srv.port, chat_body([{"role": "user", "content": "Say OK."}], 8, 0.0))
    out["cancel"] = {"chunks_before_close": n, "health_after": st, "health": health,
                     "next_request_ok": r.get("ok"), "next_ttft_ms": round(r["ttft"] * 1000, 1) if r.get("ttft") else None}


def phase_sustain(srv, spec, out, seconds=75):
    prompt = "Write a short paragraph about memory-mapped files."
    rates, thermal0 = [], subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    end = time.time() + seconds
    while time.time() < end:
        r = stream_chat(srv.port, chat_body([{"role": "user", "content": prompt}], 256, 0.0))
        if r.get("ok"):
            v = tps(r)
            if v:
                rates.append(v)
    thermal1 = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    third = max(1, len(rates) // 3)
    out["sustain"] = {
        "runs": len(rates),
        "first_third_median": round(st.median(rates[:third]), 2) if rates else None,
        "last_third_median": round(st.median(rates[-third:]), 2) if rates else None,
        "falloff_pct": round((1 - st.median(rates[-third:]) / st.median(rates[:third])) * 100, 1)
        if rates and st.median(rates[:third]) else None,
        "thermal_before": "warn" if "warning" in thermal0.lower() and "no thermal warning" not in thermal0.lower() else "nominal",
        "thermal_after": "warn" if "warning" in thermal1.lower() and "no thermal warning" not in thermal1.lower() else "nominal",
    }


# ---------------------------------------------------------------- quality
QUALITY = [
    dict(id="arith", prompt="Compute 137 * 29 + 46. Reply with the number only.", check=r"4019"),
    dict(id="speed", prompt="A train covers 210 km in 1.75 hours. What is its average speed in km/h? Reply with the number only.", check=r"120"),
    dict(id="count_r", prompt="How many times does the letter r appear in the word 'strawberry'? Reply with the number only.", check=r"\b3\b"),
    dict(id="instr", prompt="Reply with exactly the word PONG and nothing else.", check=r"^\s*PONG\s*$"),
    dict(id="json", prompt='Return only JSON: {"name": "Ada", "age": 36, "lang": "Python"}', check=r'\{\s*"name"\s*:\s*"Ada"'),
    dict(id="french", prompt="Translate into French: 'The quick brown fox jumps over the lazy dog.' Reply with the translation only.", check=r"(renard|chien)"),
    dict(id="syllogism", prompt="All Bloops are Razzies. All Razzies are Lazzies. Are all Bloops Lazzies? Reply Yes or No only.", check=r"(?i)^\s*yes\s*$"),
    dict(id="units", prompt="Convert 3.5 miles to kilometres, using 1 mile = 1.60934 km. Reply with the number only, two decimals.", check=r"5\.6[23]"),
    dict(id="summarize", prompt="Summarise in at most 8 words: 'The committee met on Tuesday and approved the annual budget after a long debate.'", check=None),
    dict(id="code", prompt="Write a Python function `is_palindrome(s)` that returns True if s reads the same backwards, ignoring case and spaces. Output only the code, no explanation.", check=None),
    dict(id="date", prompt="What day of the week is 1 January 2030? Reply with the day only.", check=r"(?i)tuesday"),
    dict(id="multi", prompt="A shop sells pens at 3 for $2. How much do 12 pens cost in dollars? Reply with the number only.", check=r"\b8\b"),
]


def run_code(answer: str):
    code = answer
    m = re.search(r"```(?:python)?\n(.*?)```", answer, re.S)
    if m:
        code = m.group(1)
    ns = {}
    try:
        exec(code, ns)  # noqa: S102 - deliberately executing model-written test code in a sandbox-less local run
        fn = ns.get("is_palindrome")
        if not fn:
            return False, "no function"
        tests = [("A man a plan a canal Panama", True), ("hello", False), ("", True)]
        for s, want in tests:
            if bool(fn(s)) != want:
                return False, f"failed {s!r}"
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, f"error: {e}"


def phase_quality(srv, spec, out):
    rows = []
    for item in QUALITY:
        r = stream_chat(srv.port, chat_body([{"role": "user", "content": item["prompt"]}], 320, 0.0))
        content = (r.get("content") or "").strip()
        reasoning = r.get("reasoning") or ""
        if item["id"] == "code":
            ok, why = run_code(content)
        elif item["check"]:
            ok, why = bool(re.search(item["check"], content, re.S)), "regex"
        else:
            ok, why = (len(content.split()) <= 10 if item["id"] == "summarize" else None), "words"
        rows.append({"id": item["id"], "ok": ok, "why": why, "answer": content[:300],
                     "reasoning_chars": len(reasoning), "tokens": (r.get("usage") or {}).get("completion_tokens"),
                     "ttft_ms": round(r["ttft"] * 1000, 1) if r.get("ttft") else None})
    out["quality"] = {"rows": rows, "passed": sum(1 for r in rows if r["ok"]), "total": len(rows)}


def phase_endpoints(srv, spec, out):
    ep = {}
    for path in ("/v1/models", "/health", "/metrics"):
        st, body = http_raw(srv.port, path)
        ep[path] = {"status": st, "bytes": len(body) if body else 0}
    # completions
    st, body = http_json(srv.port, "/v1/completions", {"model": "local-model",
                                                       "prompt": "The capital of France is",
                                                       "max_tokens": 8, "temperature": 0})
    ep["/v1/completions"] = {"status": st, "ok": bool(isinstance(body, dict) and body.get("choices")),
                             "excerpt": json.dumps(body)[:160] if isinstance(body, dict) else str(body)[:160]}
    # responses
    st, body = http_json(srv.port, "/v1/responses", {"model": "local-model", "input": "Say OK.", "max_output_tokens": 16})
    ep["/v1/responses"] = {"status": st, "ok": bool(isinstance(body, dict) and (body.get("output") or body.get("output_text"))),
                           "excerpt": json.dumps(body)[:200] if isinstance(body, dict) else str(body)[:200]}
    # anthropic messages
    st, body = http_json(srv.port, "/v1/messages", {"model": "local-model", "max_tokens": 16,
                                                    "messages": [{"role": "user", "content": "Say OK."}]},
                         headers={"x-api-key": "none", "anthropic-version": "2023-06-01"})
    ep["/v1/messages"] = {"status": st, "ok": bool(isinstance(body, dict) and body.get("content")),
                          "excerpt": json.dumps(body)[:200] if isinstance(body, dict) else str(body)[:200]}
    # embeddings
    st, body = http_json(srv.port, "/v1/embeddings", {"model": "local-model", "input": "hello"})
    ep["/v1/embeddings"] = {"status": st}
    # image input (1x1 red png)
    png = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
    body = chat_body([{"role": "user", "content": [
        {"type": "text", "text": "What colour is this image? One word."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}}]}], 24, 0.0, stream=False)
    st, r = http_json(srv.port, "/v1/chat/completions", body, timeout=180)
    ep["image"] = {"status": st, "ok": bool(isinstance(r, dict) and r.get("choices")),
                   "excerpt": (json.dumps(r)[:200] if isinstance(r, dict) else str(r)[:200])}
    out["endpoints"] = ep



FIDELITY_PROMPTS = [
    "Name three prime numbers between 10 and 20. Answer in one line.",
    "What is the capital of Australia? One word.",
    "Write the first line of a haiku about winter.",
    "Explain what a hash table is in one sentence.",
    "List the numbers 1 to 10 separated by commas.",
    "Complete: 'To be or not to be, that is the'",
]


def phase_fidelity(srv, spec, out):
    rows = []
    for p in FIDELITY_PROMPTS:
        r = stream_chat(srv.port, chat_body([{"role": "user", "content": p}], 96, 0.0))
        c = r.get("content") or ""
        rows.append({"prompt": p, "sha": hashlib.sha256(c.encode()).hexdigest()[:20],
                     "chars": len(c), "answer": c[:200],
                     "tokens": (r.get("usage") or {}).get("completion_tokens")})
    out["fidelity"] = {"rows": rows,
                       "combined_sha": hashlib.sha256("".join(r["sha"] for r in rows).encode()).hexdigest()[:24]}

PHASES = {
    "startup": phase_startup, "warmup": phase_warmup, "decode": phase_decode,
    "prefill": phase_prefill, "cached": phase_cached, "concurrency": phase_concurrency,
    "longctx": phase_longctx, "tools": phase_tools, "structured": phase_structured,
    "thinking": phase_thinking, "errors": phase_errors, "cancel": phase_cancel,
    "sustain": phase_sustain, "quality": phase_quality, "endpoints": phase_endpoints, "fidelity": phase_fidelity,
}
FEATURES = ["startup", "warmup", "tools", "structured", "endpoints", "thinking", "errors", "cancel", "quality", "fidelity"]
CORE = ["startup", "warmup", "decode", "prefill", "cached", "concurrency", "longctx",
        "tools", "structured", "thinking", "errors", "cancel", "endpoints", "quality", "fidelity"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=False)
    ap.add_argument("--engine", choices=["kapitsa", "tf12", "tf06"])
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--slots", type=int)
    ap.add_argument("--phases", default="core")
    ap.add_argument("--extra", default="", help="extra server flags, space separated")
    ap.add_argument("--env", default="", help="extra environment KEY=VALUE, comma separated")
    ap.add_argument("--no-sustain", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    resolve_models()
    if args.list:
        for k, v in SPECS.items():
            print(k, v)
        return

    spec = dict(engine=args.engine, model=args.model, ctx=args.ctx or 32768,
                slots=args.slots, port=None, name="local-model",
                extra=args.extra.split() if args.extra else [],
                env=dict(kv.split("=", 1) for kv in args.env.split(",") if "=" in kv))
    label = args.label or f"{args.engine}-{args.model}"
    spec["port"] = 8200 + (int(hashlib.md5(label.encode()).hexdigest()[:6], 16) % 200)
    phases = CORE if args.phases == "core" else (FEATURES if args.phases == "features" else args.phases.split(","))

    srv = build_server(label, spec)
    result = {
        "meta": {
            "label": label, "spec": spec, "model_path": MODELS.get(args.model),
            "host": platform.node(), "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                                             capture_output=True, text=True).stdout.strip(),
            "macos": platform.mac_ver()[0], "started": now(), "phases": phases,
            "cmd": srv.cmd,
        },
        "phases": {},
    }
    os.makedirs(RESULTS, exist_ok=True)
    out_path = args.out or os.path.join(RESULTS, f"{time.strftime('%Y%m%d-%H%M%S')}-{label}.json")
    with GpuLock():
        try:
            srv.start()
            for name in phases:
                t0 = time.time()
                try:
                    PHASES[name](srv, spec, result["phases"])
                except Exception as e:  # noqa: BLE001
                    result["phases"][name] = {"error": f"{type(e).__name__}: {e}"}
                print(f"[{label}] {name} {time.time()-t0:.1f}s", flush=True)
        finally:
            result["meta"]["footprint_end"] = srv.footprint()
            srv.stop()
            result["meta"]["ended"] = now()
            with open(out_path, "w") as f:
                json.dump(result, f, indent=1, default=str)
            print("wrote", out_path)


if __name__ == "__main__":
    main()
