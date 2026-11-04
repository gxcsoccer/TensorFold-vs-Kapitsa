"""Shared measurement library for the Kapitsa vs TensorFold evaluation.

Deliberately dependency-free (stdlib only) so every number is reproducible on a
clean macOS install.  The protocol follows the fairness rules used by Kapitsa's
own cross-engine harness: one engine at a time under an exclusive GPU lock,
explicit sampling, thinking off, cold prompts with a nonce, and decode rates
taken over the middle of the stream (first 16 and last 4 chunks dropped).
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

GPU_LOCK = os.environ.get("BENCH_GPU_LOCK", "/tmp/tensorfold-vs-kapitsa-gpu.lock")


# ---------------------------------------------------------------- utilities
def now() -> float:
    return time.time()


def free_port(preferred: int) -> int:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", preferred))
        return preferred
    except OSError:
        s.close()
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def http_json(port: int, path: str, payload=None, timeout=30, headers=None):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return r.status, json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = body.decode(errors="replace")
        return e.code, parsed


def http_raw(port: int, path: str, timeout=10):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        return None, str(e)


class GpuLock:
    """Exclusive lock shared by every engine run on this Mac."""

    def __init__(self, path=GPU_LOCK, timeout=1800):
        self.path, self.timeout, self.fh = path, timeout, None

    def __enter__(self):
        self.fh = open(self.path, "w")
        deadline = time.time() + self.timeout
        while True:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.time() > deadline:
                    raise TimeoutError(f"GPU lock {self.path} busy for {self.timeout}s")
                time.sleep(1)

    def __exit__(self, *exc):
        if self.fh:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()


# ---------------------------------------------------------------- server
class Server:
    def __init__(self, label: str, cmd: list[str], port: int, health_path="/health",
                 ready_timeout=300, env=None, model_id=None, cwd=None, stop_signals=("INT",)):
        self.label, self.cmd, self.port = label, cmd, port
        self.health_path = health_path
        self.ready_timeout = ready_timeout
        self.env = {**os.environ, **(env or {})}
        self.model_id = model_id
        self.cwd = cwd
        self.proc = None
        self.log_path = None
        self.t_ready = None
        self.t_start = None

    def start(self):
        self.log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", f"{self.label}.log")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.log = open(self.log_path, "w")
        self.t_start = now()
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     env=self.env, cwd=self.cwd, start_new_session=True)
        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server {self.label} exited with {self.proc.returncode}; see {self.log_path}")
            try:
                status, body = http_json(self.port, self.health_path, timeout=2)
                if status == 200:
                    self.t_ready = now()
                    return self
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.25)
        raise TimeoutError(f"server {self.label} not ready in {self.ready_timeout}s")

    def wait_quiet(self, quiet_s=1.5, timeout=30):
        """Wait until /health reports no running requests (TensorFold) or just settle."""
        end = time.time() + timeout
        while time.time() < end:
            try:
                st, body = http_json(self.port, "/health", timeout=2)
                live = (body or {}).get("live", {}) if isinstance(body, dict) else {}
                if not live.get("connections"):
                    time.sleep(quiet_s)
                    return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
        time.sleep(quiet_s)

    def footprint(self):
        """phys_footprint (bytes) of the server process, from macOS `footprint`."""
        if not self.proc:
            return None
        try:
            out = subprocess.run(["footprint", str(self.proc.pid)], capture_output=True, text=True, timeout=30).stdout
            m = re.search(r"phys_footprint:\s*([\d.]+)\s*(KB|MB|GB)", out)
            peak = re.search(r"phys_footprint_peak:\s*([\d.]+)\s*(KB|MB|GB)", out)
            def conv(mm):
                if not mm:
                    return None
                val, unit = float(mm.group(1)), mm.group(2)
                return val * {"KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}[unit]
            return {"phys_footprint": conv(m), "phys_footprint_peak": conv(peak)}
        except Exception:  # noqa: BLE001
            return None

    def stop(self):
        if not self.proc:
            return
        if self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGINT)
                self.proc.wait(timeout=20)
            except Exception:  # noqa: BLE001
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=10)
                except Exception:  # noqa: BLE001
                    pass
        try:
            self.log.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- streaming request
def stream_chat(port: int, payload: dict, timeout=600):
    """POST /v1/chat/completions with stream=true and time every SSE event."""
    import http.client
    body = json.dumps(payload).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t_send = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=body,
                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    resp = conn.getresponse()
    t_headers = time.perf_counter()
    chunks, usage, text, reasoning, tools = [], None, [], [], []
    content_times = []
    err = None
    if resp.status != 200:
        raw = resp.read().decode(errors="replace")
        conn.close()
        return {"ok": False, "status": resp.status, "error": raw[:2000],
                "t_send": t_send, "t_headers": t_headers}
    while True:
        line = resp.readline()
        if not line:
            break
        t = time.perf_counter()
        if not line.startswith(b"data: "):
            continue
        data = line[6:].strip()
        if data == b"[DONE]":
            break
        try:
            obj = json.loads(data)
        except Exception:  # noqa: BLE001
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices", []):
            d = ch.get("delta") or {}
            if d.get("content"):
                text.append(d["content"])
                content_times.append(t)
            if d.get("reasoning_content") or d.get("reasoning"):
                reasoning.append(d.get("reasoning_content") or d.get("reasoning"))
                content_times.append(t)
            if d.get("tool_calls"):
                tools.extend(d["tool_calls"])
    t_end = time.perf_counter()
    conn.close()
    return {
        "ok": err is None,
        "status": 200,
        "t_send": t_send, "t_headers": t_headers, "t_end": t_end,
        "ttft": (content_times[0] - t_send) if content_times else None,
        "content": "".join(text),
        "reasoning": "".join(reasoning),
        "tool_calls": tools,
        "usage": usage,
        "n_events": len(content_times),
        "event_times": [round(t - t_send, 6) for t in content_times],
        "server": obj.get("tensorfold") if isinstance(obj, dict) else None,
        "speculative": obj.get("speculative") if isinstance(obj, dict) else None,
    }


def middle_decode_rate(ev_times, usage_tokens=None, drop_first=16, drop_last=4):
    """tok/s over the middle of the stream, matching the cross-engine harness."""
    if len(ev_times) < drop_first + drop_last + 2:
        # too short to trim: use first->last
        if len(ev_times) >= 2:
            dt = ev_times[-1] - ev_times[0]
            return (len(ev_times) - 1) / dt if dt > 0 else None
        return None
    mid = ev_times[drop_first:len(ev_times) - drop_last]
    dt = mid[-1] - mid[0]
    return (len(mid) - 1) / dt if dt > 0 else None


def post_chat(port: int, payload: dict, timeout=600):
    return http_json(port, "/v1/chat/completions", payload, timeout=timeout)


# ---------------------------------------------------------------- prompts
def cold_prompt(tokens: int, nonce: str) -> str:
    """Deterministic synthetic prose/code, roughly `tokens` tokens, with a nonce
    right after the system start so no engine can reuse a cached prefix."""
    base = ("The cache eviction policy tracks each entry's last use and folds the "
            "timestamps into a compact ledger. A reader must verify the invariant "
            "before it commits, and the writer keeps a journal of every change. ")
    per_token = 5.25
    reps = max(1, int(tokens * per_token / len(base)))
    body = base * reps
    return f"[session {nonce}]\n\n{body}\n\nSummarise the paragraph above in one sentence."


def build_corpus_block(n_chars: int) -> str:
    base = ("line {i}: the scheduler maps a request to a lane, then the lane folds "
            "its tiles into one pass. Every checkpoint keeps the prefix alive. ")
    out, i = [], 0
    while sum(len(x) for x in out) < n_chars:
        out.append(base.format(i=i))
        i += 1
    return "".join(out)[:n_chars]
