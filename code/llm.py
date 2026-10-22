"""Batch Haiku calls through claude -p. 20 items/call, 4 concurrent, 120 s timeout, 1 retry, checkpointed per batch."""
import json, os, subprocess, sys, re, time, pandas as pd
from concurrent.futures import ThreadPoolExecutor
CWD='/tmp/llmtest'; os.makedirs(CWD,exist_ok=True)
def call(prompt,model='haiku'):
    for attempt in range(2):
        try:
            r=subprocess.run(['claude','-p',prompt,'--output-format','json','--model',model,'--tools','','--strict-mcp-config','--disable-slash-commands','--no-session-persistence','--system-prompt','You output only valid JSON, no prose.'],capture_output=True,text=True,timeout=120,cwd=CWD)
            j=json.loads(r.stdout); t=j['result']; m=re.search(r'\[.*\]',t,re.S)
            return json.loads(m.group(0))
        except Exception as e:
            err=repr(e)[:200]; time.sleep(3)
    return {'error':err}
def run(task,items,make_prompt,ck,bs=20):
    os.makedirs(ck,exist_ok=True)
    batches=[items[i:i+bs] for i in range(0,len(items),bs)]
    def one(ib):
        i,b=ib; p=f'{ck}/{i:04d}.json'
        if os.path.exists(p): return
        res=call(make_prompt(b))
        if isinstance(res,dict): print(task,'batch',i,'FAILED',res,flush=True); return
        json.dump(res,open(p,'w'))
    with ThreadPoolExecutor(4) as ex: list(ex.map(one,list(enumerate(batches))))
    out=[]
    for f in sorted(os.listdir(ck)): out+=json.load(open(f'{ck}/{f}'))
    return out
if __name__=='__main__':
    task=sys.argv[1]
    pg=pd.read_parquet('state/pages.parquet').set_index('number')
    L=pd.read_parquet('state/labels_all.parquet'); nums=sorted(L.number.unique())
    items=[(int(n),pg.statement[n][:700]) for n in nums]
    if task=='probe':
        mk=lambda b:"From your own memory only, for each Erdős problem below say whether it has been fully SOLVED (proved or disproved) as of your latest knowledge. If unsure, guess. Reply with a JSON list of objects {\"id\":<int>,\"solved\":true|false}.\n\n"+"\n".join(f"[{i}] {s}" for i,s in b)
        out=run('probe',items,mk,'state/llm_probe')
    else:
        mk=lambda b:"Rate each math problem statement on how STANDARD the techniques likely needed to resolve it are, 1 (needs entirely new ideas) to 7 (routine application of well-known techniques). Judge from the statement alone. Reply with a JSON list of objects {\"id\":<int>,\"standard\":<1-7>}.\n\n"+"\n".join(f"[{i}] {s}" for i,s in b)
        out=run('rate',items,mk,'state/llm_rate')
    pd.DataFrame(out).to_parquet(f'state/llm_{task}.parquet'); print(task,len(out),'of',len(items))
