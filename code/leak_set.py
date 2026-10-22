"""Docs to drop everywhere (leakage filter): (a) mention Erdős with an Erdős-problem number, (b) arXiv ids cited on the problem pages or in the AI wiki."""
import re, glob, pandas as pd
c=pd.read_parquet('state/corpus_raw.parquet').sort_values('date').reset_index(drop=True)  # same order as embeddings
txt=(c.title+' '+c.abstract).str.replace('\n',' ')
has=txt.str.contains(r'Erd[oőö]s|Erdos',regex=True)
pat=re.compile(r'(?:problems?|conjectures?|questions?)\s*(?:no\.?|number|#|nos\.?)?\s*#?\s*(\d{1,4})\b|#\s*(\d{1,4})\b|erdosproblems\.com/(?:forum/)?(?:thread/)?(\d{1,4})',re.I)
leak={}; 
for i in c.index[has]:
    for m in pat.finditer(txt[i]):
        n=int([g for g in m.groups() if g][0])
        if 1<=n<=1221: leak.setdefault(i,set()).add(n)
ids=set()
rx=re.compile(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})',re.I)
for f in glob.glob('data/pages/*.html')+glob.glob('data/erdosproblems.wiki/*.md'):
    ids|={m.lower() for m in rx.findall(open(f,errors='ignore').read())}
idx=set(c.index[c.id.str.lower().isin(ids)])
drop=sorted(set(leak)|idx)
print('erdos+number docs',len(leak),'cited arxiv ids',len(ids),'matched',len(idx),'total drop',len(drop),'erdos mention docs',int(has.sum()))
pd.Series(drop).to_frame('idx').to_parquet('state/leak_drop.parquet')
