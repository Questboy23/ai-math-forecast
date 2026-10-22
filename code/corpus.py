import pyarrow.parquet as pq, glob, pandas as pd, re
out=[]
for f in sorted(glob.glob('data/arxiv/data/*.parquet')):
    t=pq.read_table(f,columns=['id','title','categories','abstract','versions']).to_pandas()
    cat=t.categories.str.split().str[0]
    keep=cat.str.startswith('math.')|cat.isin(['cs.DM','cs.IT','cs.CC','cs.CG','cs.LO','cs.DS','math-ph','stat.TH'])|t.categories.str.contains('math.CO|math.NT')
    t=t[keep].copy()
    t['date']=pd.to_datetime(t.versions.map(lambda v:v[0]['created']),errors='coerce',utc=True).dt.tz_localize(None)
    t['primary']=t.categories.str.split().str[0]
    out.append(t[['id','title','abstract','categories','primary','date']]); print(f,len(t),flush=True)
c=pd.concat(out); c=c[c.date.notna()&(c.date>='1990-01-01')]
c.to_parquet('state/corpus_raw.parquet'); print(len(c)); print(c.date.min(),c.date.max()); print(c.primary.value_counts().head(8))
print(c.groupby(c.date.dt.year).size().tail(8))
