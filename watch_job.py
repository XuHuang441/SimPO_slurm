#!/usr/bin/env python3
import os, subprocess, time, sys, shlex, socket, urllib.parse, urllib.request

BOT_TOKEN = "8421438778:AAGtyzTt1MoM16Cp_3whzZe42-tlqe7E8Gg"
CHAT_ID = "719921093"
INTERVAL = int(os.environ.get("INTERVAL", "60"))

if not BOT_TOKEN or not CHAT_ID:
    print("Please set BOT_TOKEN and CHAT_ID env vars.")
    sys.exit(1)

def tg_send(text: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": CHAT_ID, "text": text}).encode()
    req = urllib.request.Request(url, data=data)
    urllib.request.urlopen(req, timeout=10).read()

def run(cmd: str) -> str:
    p = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return p.stdout.strip()

def get_state(jobid: str) -> str:
    state = run(f'squeue -j {jobid} -h -o %T')
    if not state:
        state = run(f'sacct -j {jobid} --format=State --noheader').splitlines()
        state = state[0].split()[0] if state else "NOT_IN_QUEUE"
    return state

host = socket.gethostname().split('.')[0]
jobids = sys.argv[1:]
if not jobids:
    print("Usage: slurm_tg_watch.py <JOBID1> [JOBID2 ...]")
    sys.exit(1)

last = {}
for jid in jobids:
    st = get_state(jid)
    last[jid] = st
    tg_send(f"[{host}] Job {jid} 初始状态：{st}")

while True:
    for jid in jobids:
        cur = get_state(jid)
        if cur != last[jid]:
            node = run(f'squeue -j {jid} -h -o %N')
            tuse = run(f'squeue -j {jid} -h -o %M')
            msg = f"[{host}] Job {jid} 状态变化：{last[jid]} → {cur}"
            if node: msg += f" | Node: {node}"
            if tuse: msg += f" | TIME: {tuse}"
            tg_send(msg)
            last[jid] = cur
    time.sleep(INTERVAL)
