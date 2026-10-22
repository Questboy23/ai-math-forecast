import re, glob, html, pandas as pd
rows=[]
for f in glob.glob('data/pages/*.html'):
    n=int(re.search(r'(\d+)\.html',f).group(1)); h=open(f).read()
    m=re.search(r'<div id="content">(.*?)</div>',h,re.S)
    st=html.unescape(re.sub(r'<[^>]+>',' ',m.group(1))).strip() if m else ''
    st=re.sub(r'\s+',' ',st)
    keys=re.findall(r"addNewBox\('([^']+)', '%d'"%n,h); keys=list(dict.fromkeys(keys))
    yrs=[]
    for k in keys:
        mm=re.search(r'(\d{4})|(\d{2})[a-z]?$',k)
        if mm: yrs.append(int(mm.group(1)) if mm.group(1) else 1900+int(mm.group(2)))
    yrs=[y for y in yrs if 1900<=y<=2026]
    rows.append(dict(number=n,statement=st,nrefs=len(keys),ref_years=yrs,src_year=min(yrs) if yrs else None,ok=bool(m)))
d=pd.DataFrame(rows).sort_values('number'); d.to_parquet('state/pages.parquet')
print(len(d),d.ok.mean(),(d.statement.str.len()>20).mean(),d.src_year.notna().mean()); print(d.statement.str.len().describe()); print(d.head(3).T)
