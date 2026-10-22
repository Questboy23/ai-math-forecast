"""POST HOC, EXPLORATORY (run after the preregistered backtest, not part of H1/H2): problem-grouped 5-fold CV over all 11 origins, i.e. more training data but not out-of-time."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'code'); from modeling import *
from sklearn.model_selection import GroupKFold
F=pd.read_parquet('state/features_all.parquet'); L=pd.concat([pd.read_parquet('state/labels_train.parquet'),pd.read_parquet('state/labels_test_SEALED.parquet')]).drop_duplicates(['number','origin'])
D=F.merge(L[['number','origin','event']],on=['number','origin']).reset_index(drop=True); D['y']=D.event
obs=feats(D,'obs'); ob=feats(D,'obs+bridge'); out={}
rng=np.random.default_rng(0); res={}
for nm,cols,k in [('lr_obs',obs,'lr'),('lr_obs_bridge',ob,'lr'),('hgb_all',ob,'gb')]:
    p=np.zeros(len(D))
    for tr,te in GroupKFold(5).split(D,groups=D.number):
        m=make_model(k,0.03 if k=='lr' else None).fit(D.loc[tr,cols].values.astype(float),D.y[tr]); p[te]=m.predict_proba(D.loc[te,cols].values.astype(float))[:,1]
    res[nm]=p; out[nm]=float(roc_auc_score(D.y,p)); print(nm,out[nm],flush=True)
P=D[['number','origin','y']].assign(p=res['lr_obs_bridge'],p0=res['lr_obs'])
a1,a2,z,pv=delong(D.y,res['lr_obs_bridge'],res['lr_obs']); lo,hi=cluster_boot_auc(P,500,second='p0')
out['delta_bridge']=a1-a2; out['delong_p']=pv; out['delta_cluster_ci']=[lo,hi]; out['rows']=len(D); out['events']=int(D.y.sum())
print(out); json.dump(out,open('out/exploratory_results.json','w'),indent=1)
