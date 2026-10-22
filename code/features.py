"""Stage 2: features as of each origin. Bridge features use only corpus docs dated before the origin, minus leakage set."""
import glob, numpy as np, pandas as pd, torch, re, datetime as dt
from sentence_transformers import SentenceTransformer
dev='mps'
meta=pd.read_parquet('state/corpus_meta.parquet')
dates=meta.date.values.astype('datetime64[ns]')
files=sorted(glob.glob('state/emb/*.npy')); E=np.concatenate([np.load(f) for f in files]); assert len(E)==len(meta),(len(E),len(meta))
leak=pd.read_parquet('state/leak_drop.parquet').idx.values
E[leak]=0
Et=torch.from_numpy(E).to(dev)  # fp16
pg=pd.read_parquet('state/pages.parquet').sort_values('number').reset_index(drop=True)
m=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu'); m.max_seq_length=256
P=m.encode(pg.statement.tolist(),batch_size=64,normalize_embeddings=True); np.save('state/problem_emb.npy',P)
Pt=torch.from_numpy(P).to(dev).half(); num2i={n:i for i,n in enumerate(pg.number)}
cats=meta.primary.astype('category'); catcode=torch.from_numpy(cats.cat.codes.values.astype(np.int64))
snap=pd.read_parquet('state/snapshots.parquet'); snap['date']=pd.to_datetime(snap.date)
origins=sorted(d for d in snap.date.unique() if d<=pd.Timestamp('2026-07-01'))+[pd.Timestamp('2026-10-22')]

def topk(Q,n,k,chunk=60000):
    vals=None
    for s in range(0,n,chunk):
        e=min(n,s+chunk); sim=Q@Et[s:e].T
        v,i=sim.topk(min(k,e-s),dim=1); i=i+s
        if vals is None: vals,idx=v,i
        else:
            vals=torch.cat([vals,v],1); idx=torch.cat([idx,i],1); vals,o=vals.topk(k,dim=1); idx=idx.gather(1,o)
    return vals.float(),idx

out=[]
for o in origins:
    n=int(np.searchsorted(dates,np.datetime64(o))); print('origin',o.date(),'corpus',n,flush=True)
    v50,i50=topk(Pt,n,50)
    cc=np.array([len(set(catcode[i50[r,:20].cpu()].tolist())) for r in range(len(pg))])
    # 2-hop: neighbors' own top-10 (excluding self) similarity to problem
    q=Et[i50[:,:10].reshape(-1)]; v11,i11=topk(q,n,11)
    nn=i11.reshape(len(pg),10,11)  # first col is self (or near-duplicate); drop first
    nn2=nn[:,:,1:].reshape(len(pg),-1)
    two=(Pt.float().unsqueeze(1)*Et[nn2].float()).sum(-1).mean(1)
    d=pd.DataFrame(dict(number=pg.number,origin=o,d_near=(1-v50[:,0]).cpu().numpy(),d_mass=v50.mean(1).cpu().numpy(),
                        d_2hop=two.cpu().numpy(),concept_cost=cc,corpus_n=n))
    # hub score among problems open at origin
    op=snap[(snap.date==o)&(snap.informal=='open')].number.values
    if o==pd.Timestamp('2026-10-22'): op=snap[(snap.date==o)&(snap.informal=='open')].number.values
    ii=[num2i[x] for x in op]; S=P[ii]@P[ii].T; iu=np.triu_indices(len(ii),1); thr=np.percentile(S[iu],95)
    hub=dict(zip(op,(S>thr).sum(1)-1)); d['hub']=d.number.map(hub); d['hub_thr']=thr
    out.append(d)
F=pd.concat(out); F.to_parquet('state/bridge_features.parquet'); print(F.describe().T)
