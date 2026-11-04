"""Extra feature probes that need their own server flags: vision and API auth."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchlib import Server, GpuLock, http_json, stream_chat
import run as R

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def probe(label, engine, model, extra, env=None, api_key=None):
    R.resolve_models()
    spec = dict(engine=engine, model=model, ctx=32768, port=8400 + (abs(hash(label)) % 60),
                name="local-model", extra=extra, env=env or {})
    srv = R.build_server(label, spec)
    if api_key:
        srv.cmd += ["--api-key", api_key]
    out = {"label": label, "cmd": srv.cmd}
    with GpuLock():
        try:
            srv.start()
            if api_key:
                st_no, body_no = http_json(srv.port, "/v1/chat/completions",
                                           {"model": "local-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
                                           timeout=60)
                st_ok, body_ok = http_json(srv.port, "/v1/chat/completions",
                                           {"model": "local-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
                                           timeout=60, headers={"Authorization": "Bearer " + api_key})
                out["auth"] = {"without_key": st_no, "with_key": st_ok,
                               "pass": st_no in (401, 403) and st_ok == 200}
            body = R.chat_body([{"role": "user", "content": [
                {"type": "text", "text": "What colour is this image? One word."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}]}], 24, 0.0, stream=False)
            st, r = http_json(srv.port, "/v1/chat/completions", body, timeout=180)
            content = ""
            if isinstance(r, dict):
                content = (((r.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
            out["vision"] = {"status": st, "content": content[:80],
                             "pass": st == 200 and bool(content),
                             "error": None if st == 200 else json.dumps(r)[:200]}
            st_m, models = http_json(srv.port, "/v1/models")
            out["models"] = models
        finally:
            srv.stop()
            path = os.path.join(R.ROOT, "results", f"extra-{label}.json")
            json.dump(out, open(path, "w"), indent=1)
            print(json.dumps(out, indent=1)[:1500])


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "tf12":
        probe("tf12-27b-vision", "tf12", "27b", ["--vision"])
    elif which == "tf06":
        probe("tf06-27b-vision", "tf06", "27b", ["--vision"])
    elif which == "kapitsa":
        probe("kapitsa-27b-vision-auth", "kapitsa", "27b", [], api_key="local-test-key")
