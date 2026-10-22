import os, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
c=pd.read_parquet('state/corpus_raw.parquet').sort_values('date').reset_index(drop=True)
c[['id','primary','categories','date','title']].to_parquet('state/corpus_meta.parquet')
txt=(c.title.str.replace('\n',' ')+'. '+c.abstract.str.replace('\n',' ').str.slice(0,600)).tolist()
m=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='mps'); m.max_seq_length=128
CH=40000
for i in range(0,len(txt),CH):
    p=f'state/emb/{i//CH:03d}.npy'
    if os.path.exists(p): continue
    e=m.encode(txt[i:i+CH],batch_size=128,normalize_embeddings=True).astype(np.float16)
    np.save(p+'.tmp.npy',e); os.rename(p+'.tmp.npy',p); print(i,flush=True)
print('DONE')
