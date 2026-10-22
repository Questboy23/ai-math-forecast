"""Stage 4. Run ONLY after PREREGISTRATION.md is hashed (state/prereg_hash.txt must exist)."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0,'code'); from modeling import *
assert os.path.exists('state/prereg_hash.txt'), 'preregister first'
F=pd.read_parquet('state/features_all.parquet')
L=pd.concat([pd.read_parquet('state/labels_train.parquet'),pd.read_parquet('state/labels_test_SEALED.parquet')])
D=F.merge(L[['number','origin','event','competing','kind']],on=['number','origin'],how='inner'); D['y']=D.event
TEST=[pd.Timestamp(x) for x in ['2026-01-01','2026-02-01','2026-03-01','2026-04-01','2026-05-01','2026-06-01','2026-07-01']]
obs=feats(D,'obs'); ob=feats(D,'obs+bridge'); allf=feats(D,'all')
res={}; rng=np.random.default_rng(0)
def run(name,Dx,cols,kind,fixedC=None):
    P,Cs=backtest(Dx,TEST,cols,kind,fixedC); return P,Cs
P_obs,C_obs=run('lr_obs',D,obs,'lr'); P_ob,C_ob=run('lr_ob',D,ob,'lr'); P_gb,_=run('hgb',D,allf,'gb')
P_rand=P_obs.copy(); P_rand['p']=rng.uniform(size=len(P_rand))
preds=P_obs.rename(columns={'p':'p_lr_obs'}).merge(P_ob[['number','origin','p']].rename(columns={'p':'p_lr_obs_bridge'}),on=['number','origin']).merge(P_gb[['number','origin','p']].rename(columns={'p':'p_hgb_all'}),on=['number','origin']).merge(P_rand[['number','origin','p']].rename(columns={'p':'p_random'}),on=['number','origin'])
preds.to_parquet('out/backtest_predictions.parquet')
out={'models':{}}
for nm,col in [('random','p_random'),('lr_obs','p_lr_obs'),('lr_obs_bridge','p_lr_obs_bridge'),('hgb_all','p_hgb_all')]:
    P=preds[['number','origin','y']].assign(p=preds[col]); m=metrics(P); lo,hi=cluster_boot_auc(P,2000); m['auc_ci']=[lo,hi]; m['calib']=calib(P).to_dict('list'); out['models'][nm]=m
    print(nm,{k:(round(v,4) if isinstance(v,float) else v) for k,v in m.items() if k not in('calib',)},flush=True)
# H1
out['H1']={'auc':out['models']['lr_obs']['auc'],'ci':out['models']['lr_obs']['auc_ci'],'supported':bool(out['models']['lr_obs']['auc']>0.6)}
# H2: DeLong + cluster bootstrap of difference + permutation
a1,a2,z,pv=delong(preds.y,preds.p_lr_obs_bridge,preds.p_lr_obs)
d=a1-a2; dlo,dhi=cluster_boot_auc(preds.assign(p=preds.p_lr_obs_bridge),1000,second='p_lr_obs')
perm=[]; bcols=BRIDGE+HUB; Dp=D.copy()
for b in range(300):
    Dp=D.copy()
    for o,idx in D.groupby('origin').groups.items():
        sh=D.loc[idx,bcols].sample(frac=1,random_state=int(rng.integers(1e9))).values; Dp.loc[idx,bcols]=sh
    Pp,_=backtest(Dp,TEST,ob,'lr',fixedC=C_ob); perm.append(roc_auc_score(Pp.y,Pp.p))
pperm=(1+sum(x>=a1 for x in perm))/(1+len(perm))
out['H2']={'auc_obs':a2,'auc_obs_bridge':a1,'delta':d,'delong_z':z,'delong_p':pv,'delta_cluster_boot_ci':[dlo,dhi],'perm_p':pperm,'perm_aucs':[float(x) for x in perm],'perm_mean_auc':float(np.mean(perm)),'perm_sd':float(np.std(perm)),'supported':bool(d>0 and pv<0.05)}
print('H2',out['H2'],flush=True)
# Secondary: event types
sec={}
for kind in ['lit_id','new_sol']:
    Dk=D[~((D.event==1)&(D.kind!=kind))].copy(); Dk['y']=((Dk.event==1)&(Dk.kind==kind)).astype(int)
    r={}
    for nm,cols,k in [('lr_obs',obs,'lr'),('lr_obs_bridge',ob,'lr'),('hgb_all',allf,'gb')]:
        P,_=backtest(Dk,TEST,cols,k); m=metrics(P); r[nm]={'auc':m['auc'],'prauc':m['prauc'],'base':m['base'],'events':m['events']}
    sec[kind]=r; print('secondary',kind,r,flush=True)
out['secondary']=sec
# Robustness: drop each group from lr_obs_bridge and hgb_all
G=groups(D); rob={}
for g,cols in G.items():
    keep=[c for c in ob if c not in cols]
    P1,_=backtest(D,TEST,keep,'lr'); P2,_=backtest(D,TEST,[c for c in allf if c not in cols],'gb')
    rob[g]={'lr_obs_bridge_auc':roc_auc_score(P1.y,P1.p),'hgb_all_auc':roc_auc_score(P2.y,P2.p)}
out['robustness']={'full_lr_obs_bridge':a1,'full_hgb_all':out['models']['hgb_all']['auc'],'drop':rob}
# Side test D (technique rating)
if D.tech_standard.notna().mean()>0.9:
    Dd=D.copy(); Dd['tech_standard']=Dd.tech_standard.fillna(Dd.tech_standard.median())
    Pd_,_=backtest(Dd,TEST,obs+['tech_standard'],'lr'); out['side_D']={'auc_obs_plus_D':roc_auc_score(Pd_.y,Pd_.p),'auc_obs':out['models']['lr_obs']['auc'],'contaminated':False,'probe':open('state/probe_result.txt').read()}
json.dump(out,open('out/backtest_results.json','w'),indent=1,default=float)
print('DONE')
