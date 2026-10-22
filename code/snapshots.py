"""Reconstruct per-problem status at dates from git history of teorth/erdosproblems problems.yaml."""
import subprocess, yaml, json, datetime as dt, pandas as pd
R='data/erdosproblems'
def git(*a): return subprocess.run(['git','-C',R,*a],capture_output=True,text=True,check=True).stdout
def origins():
    out=[]
    y,m=2025,9
    while (y,m)<=(2026,10):
        out.append(dt.date(y,m,1)); m+=1
        if m>12: y,m=y+1,1
    return out
rows=[]
dates=origins()+[dt.date(2026,10,22)]
for d in dates:
    c=git('rev-list','-1',f'--before={d.isoformat()} 00:00','HEAD').strip() if d<dt.date(2026,10,22) else git('rev-parse','HEAD').strip()
    if not c: print('no commit before',d); continue
    y=yaml.load(git('show',f'{c}:data/problems.yaml'),Loader=yaml.CSafeLoader)
    for p in y:
        rows.append(dict(date=d.isoformat(),number=int(p['number']),informal=(p.get('informal_status') or p['status'])['state'],informal_upd=(p.get('informal_status') or p['status']).get('last_update'),
            formal=p.get('formal_status',{}).get('state'),prize=p.get('prize'),tags='|'.join(p.get('tags',[]) or []),oeis=len(p.get('oeis',[]) or []),
            formalized=(p.get('formalized') or {}).get('state')))
    print(d,c[:8],len(y))
pd.DataFrame(rows).to_parquet('state/snapshots.parquet')
