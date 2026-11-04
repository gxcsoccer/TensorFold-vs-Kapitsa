import json, time, subprocess, sys, urllib.request, os, signal

MODEL = "/Users/lang/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec"

def post(port, path, payload, stream=False, timeout=300):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type":"application/json"})
    t0=time.time()
    r = urllib.request.urlopen(req, timeout=timeout)
    if stream:
        chunks=[]
        for line in r:
            if line.startswith(b"data: "):
                chunks.append(line)
            if time.time()-t0 > timeout: break
        return chunks
    return json.loads(r.read())

def get(port, path):
    return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10).read())

def wait(port, n=120):
    for _ in range(n):
        try:
            get(port,"/health"); return True
        except Exception: time.sleep(1)
    return False

def stop(p):
    p.send_signal(signal.SIGINT)
    try: p.wait(10)
    except Exception: p.kill()

def probe_kapitsa():
    p = subprocess.Popen(["kapitsa","serve","--model",MODEL,"--private","--port","8011","--context","8192"],
                         stdout=open("/tmp/kap-probe.log","w"), stderr=subprocess.STDOUT, start_new_session=True)
    assert wait(8011), "kapitsa not ready"
    out={}
    out["health"]=get(8011,"/health")
    out["models"]=get(8011,"/v1/models")
    out["chat"]=post(8011,"/v1/chat/completions",{"model":"x","messages":[{"role":"user","content":"What is 17*23? Answer with just the number."}],"max_tokens":32,"temperature":0})
    out["stream"]=[l.decode() for l in post(8011,"/v1/chat/completions",{"model":"x","messages":[{"role":"user","content":"Count from 1 to 3."}],"max_tokens":16,"temperature":0,"stream":True},stream=True)][:14]
    try:
        out["err"]=post(8011,"/v1/chat/completions",{"model":"x","messages":[],"max_tokens":5})
    except Exception as e:
        out["err"]=f"HTTPERR {e}"
    stop(p)
    return out

def probe_tf():
    p = subprocess.Popen(["/opt/homebrew/opt/tensorfold@0.6/bin/tensorfold","serve",MODEL,"--port","8012","--name","local-model","--context","8192","--no-thinking","--metrics-open"],
                         stdout=open("/tmp/tf-probe.log","w"), stderr=subprocess.STDOUT, start_new_session=True)
    assert wait(8012), "tf not ready"
    out={}
    out["health"]=get(8012,"/health")
    out["models"]=get(8012,"/v1/models")
    out["chat"]=post(8012,"/v1/chat/completions",{"model":"local-model","messages":[{"role":"user","content":"What is 17*23? Answer with just the number."}],"max_tokens":32,"temperature":0})
    out["stream"]=[l.decode() for l in post(8012,"/v1/chat/completions",{"model":"local-model","messages":[{"role":"user","content":"Count from 1 to 3."}],"max_tokens":16,"temperature":0,"stream":True},stream=True)][:14]
    try:
        out["err"]=post(8012,"/v1/chat/completions",{"model":"local-model","messages":[],"max_tokens":5})
    except Exception as e:
        out["err"]=f"HTTPERR {e}"
    try:
        out["metrics"]=urllib.request.urlopen("http://127.0.0.1:8012/metrics",timeout=5).read().decode()[:1500]
    except Exception as e:
        out["metrics"]=f"ERR {e}"
    stop(p)
    return out

if __name__=="__main__":
    which=sys.argv[1]
    r = probe_kapitsa() if which=="k" else probe_tf()
    print(json.dumps(r, indent=1)[:8000])
