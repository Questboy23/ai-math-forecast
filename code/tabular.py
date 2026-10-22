"""Obscurity/attention features as of each origin, merged with labels + bridge features -> state/dataset.parquet"""
import numpy as np, pandas as pd, re
snap=pd.read_parquet('state/snapshots.parquet'); snap['date']=pd.to_datetime(snap.date)
pg=pd.read_parquet('state/pages.parquet'); fd=pd.read_parquet('state/formal_dates.parquet'); fd['formal_date']=pd.to_datetime(fd.formal_date)
ai=pd.read_parquet('state/ai_entries.parquet'); ai['date']=pd.to_datetime(ai.date)
# er source year: earliest year among Erdős-authored keys, else earliest overall
import glob,html
def src(n):
    h=open(f'data/pages/{n}.html').read(); keys=list(dict.fromkeys(re.findall(r"addNewBox\('([^']+)', '%d'"%n,h)))
    def yr(k):
        mm=re.search(r'(\d{4})|(\d{2})[a-z]?(?:,|$|\b)',k)
        return (int(mm.group(1)) if mm.group(1) else 1900+int(mm.group(2))) if mm else None
    ek=[yr(k) for k in keys if k.startswith('Er') or 'Er' in k[:6]]; ek=[y for y in ek if y and 1920<=y<=2026]
    ak=[yr(k) for k in keys]; ak=[y for y in ak if y and 1900<=y<=2026]
    return (min(ek) if ek else (min(ak) if ak else np.nan))
pg['src_year']=[src(n) for n in pg.number]
pg['stmt_len']=np.log1p(pg.statement.str.len())
dates=sorted(snap.date.unique())
rows=[]
snap=snap.sort_values(['number','date'])
for n,g in snap.groupby('number'):
    g=g.reset_index(drop=True)
    chg=((g.informal!=g.informal.shift())|(g.informal_upd!=g.informal_upd.shift())|(g.formal!=g.formal.shift())).astype(int); chg.iloc[0]=0
    g['edits']=chg.cumsum()
    rows.append(g)
S=pd.concat(rows)
S['prize_usd']=pd.to_numeric(S.prize.astype(str).str.replace(r'[^\d.]','',regex=True),errors='coerce').fillna(0)
S['log_prize']=np.log1p(S.prize_usd)
S['upd_age']=(S.date-pd.to_datetime(S.informal_upd,errors='coerce')).dt.days.clip(lower=0).fillna(0)
S['formalized_yes']=(S.formalized=='yes').astype(int)
TAGS=['number theory','additive combinatorics','graph theory','combinatorics','geometry','analysis','discrete geometry','ramsey theory','sequences','primes','arithmetic progressions','set theory','probability','algebra','irrationality','divisors','hypergraphs','chromatic number','distances','sidon sets']
for t in TAGS: S['tag_'+t.replace(' ','_')]=S.tags.str.contains(t,regex=False).astype(int)
S['n_tags']=S.tags.str.count(r'\|')+(S.tags!='').astype(int)
S=S.merge(pg[['number','src_year','stmt_len']],on='number').merge(fd,on='number',how='left')
S['formal_before']=((S.formal_date<S.date)).astype(int); S['formal_age']=((S.date-S.formal_date).dt.days.clip(lower=0)).fillna(0)
S['src_age']=S.date.dt.year-S.src_year
# prior AI wiki entries (any outcome, any section) dated strictly before origin
def prior(n,d,secs=None):
    a=ai[(ai.number==n)&(ai.date<d)]
    return len(a)
cnt=ai.groupby('number')
S['prior_ai']=[len(ai[(ai.number==n)&(ai.date<d)]) for n,d in zip(S.number,S.date)]
S=S.rename(columns={'date':'origin'})
keep=['number','origin','log_prize','prize_usd','src_age','formal_before','formal_age','stmt_len','edits','upd_age','formalized_yes','oeis','n_tags','prior_ai']+[c for c in S if c.startswith('tag_')]
S=S[keep]
S.to_parquet('state/tabular.parquet'); print(S.describe().T[['mean','min','max']].head(30)); print(S.isna().sum()[lambda x:x>0])
