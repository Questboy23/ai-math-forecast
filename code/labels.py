"""Stage 1: parse AI wiki, build (problem, origin) rows with 3-month event labels."""
import re, glob, datetime as dt, pandas as pd
W=glob.glob('data/erdosproblems.wiki/AI-contributions*')[0]
MON={m:i+1 for i,m in enumerate('Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split())}
sect=None; ai=[]
for line in open(W,encoding='utf-8'):
    m=re.match(r'<a id="sect-(\w+)">',line)
    if m: sect=m.group(1); continue
    if not line.startswith('| [['): continue
    cells=[c.strip() for c in line.strip().strip('|').split('|')]
    nums=[int(x) for x in re.findall(r'\[\[(\d+)\]\]',cells[0])]
    dm=re.search(r'(\d{1,2}) (\w{3}), (\d{4})',line)
    date=dt.date(int(dm.group(3)),MON[dm.group(2)],int(dm.group(1))) if dm else None
    em=re.search(r'[🟢🟡🔴⚪]',line); color=em.group(0) if em else None
    for n in nums: ai.append(dict(number=n,section=sect,date=date,color=color,text=line.strip()[:300]))
ai=pd.DataFrame(ai); ai.to_parquet('state/ai_entries.parquet')
print('AI rows',len(ai),'no date',ai.date.isna().sum()); print(ai.groupby(['section','color'],dropna=False).size().unstack(fill_value=0))
# qualifying: primary sections (1a-1d) green/yellow = new solution; 2a green/yellow = literature identification
q=ai[(ai.color.isin(['🟢','🟡']))&(ai.section.isin(['1a','1b','1c','1d','2a']))&ai.date.notna()].copy()
q['kind']=q.section.map(lambda s:'lit_id' if s=='2a' else 'new_sol')

snap=pd.read_parquet('state/snapshots.parquet'); snap['date']=pd.to_datetime(snap.date)
OPEN='open'
def add_months(d,k):
    y=d.year+(d.month-1+k)//12; m=(d.month-1+k)%12+1; return pd.Timestamp(y,m,1)
origins=[pd.Timestamp(2025,9,1)]+[add_months(pd.Timestamp(2025,9,1),k) for k in range(1,11)]  # 2025-09-01..2026-07-01
sd={d:g.set_index('number') for d,g in snap.groupby('date')}
last_upd=sd[pd.Timestamp(2026,10,22)].informal_upd
rows=[]
for o in origins:
    end=add_months(o,3); s0=sd[o]; s1=sd[end]
    for n,r in s0.iterrows():
        if r.informal!=OPEN: continue
        st1=s1.informal.get(n,OPEN)
        left = st1!=OPEN and not st1 in ('falsifiable','verifiable','decidable')
        ev=comp=0; kind=None; chg=None
        if left:
            # status change date: last_update in the end snapshot, clipped to window
            chg=pd.to_datetime(s1.informal_upd.get(n),errors='coerce')
            if pd.isna(chg) or chg<o or chg>end: chg=end
            qq=q[(q.number==n)&(pd.to_datetime(q.date)>=chg-pd.Timedelta(days=120))&(pd.to_datetime(q.date)<=chg+pd.Timedelta(days=30))]
            if len(qq): ev=1; kind='lit_id' if (qq.kind=='lit_id').all() else 'new_sol'
            else: comp=1
        rows.append(dict(number=n,origin=o,event=ev,competing=comp,kind=kind,state_end=st1,chg=chg))
L=pd.DataFrame(rows)
# competing events removed from later risk sets automatically (status no longer open); report
L.to_parquet('state/labels_all.parquet')
g=L.groupby('origin').agg(n=('event','size'),events=('event','sum'),competing=('competing','sum')); print(g)
print(L[L.event==1].kind.value_counts())
