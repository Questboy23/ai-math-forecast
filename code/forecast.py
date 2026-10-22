"""Stage 5: refit on all rows, forecast every problem open on 2026-10-22."""
import sys, json, hashlib, numpy as np, pandas as pd
sys.path.insert(0,'code'); from modeling import *
F=pd.read_parquet('state/features_all.parquet'); L=pd.concat([pd.read_parquet('state/labels_train.parquet'),pd.read_parquet('state/labels_test_SEALED.parquet')])
D=F.merge(L[['number','origin','event']],on=['number','origin']); D['y']=D.event
R=json.load(open('out/backtest_results.json'))
cand={k:R['models'][k]['brier'] for k in ['lr_obs','lr_obs_bridge','hgb_all']}; pick=min(cand,key=cand.get)  # preregistered rule: lowest pooled test Brier
print('forecast model',pick,cand)
now=pd.Timestamp('2026-10-22'); X=F[F.origin==now].copy()
snap=pd.read_parquet('state/snapshots.parquet'); snap['date']=pd.to_datetime(snap.date); s=snap[snap.date==now]
open_nums=set(s[s.informal=='open'].number); X=X[X.number.isin(open_nums)]
cols=feats(D,'obs') if pick=='lr_obs' else feats(D,'obs+bridge')
kind='gb' if pick=='hgb_all' else 'lr'
C=None
if kind=='lr':
    C=pick_C(D[cols].values.astype(float),D.y.values,D.origin)
m=make_model(kind,C).fit(D[cols].values.astype(float),D.y.values)
p3=m.predict_proba(X[cols].values.astype(float))[:,1]
pg=pd.read_parquet('state/pages.parquet').set_index('number')
out=pd.DataFrame({'number':X.number.values,'p_ai_resolved_3mo':p3})
out['p_ai_resolved_12mo_LOW_constant_hazard']=1-(1-out.p_ai_resolved_3mo)**4
out['statement']=out.number.map(lambda n:pg.statement[n][:160])
out['url']=out.number.map(lambda n:f'https://www.erdosproblems.com/{n}')
out=out.sort_values('p_ai_resolved_3mo',ascending=False).reset_index(drop=True); out.insert(0,'rank',out.index+1)
out['forecast_date']='2026-10-22'; out['model']=pick
out.to_csv('out/forecast.csv',index=False,float_format='%.6f')
h=hashlib.sha256(open('out/forecast.csv','rb').read()).hexdigest(); open('out/forecast.sha256','w').write(h+'  forecast.csv\n')
json.dump({'model':pick,'C':C,'n_open':len(out),'sum_p3':float(out.p_ai_resolved_3mo.sum()),'sha256':h,'cols':cols},open('out/forecast_meta.json','w'),indent=1)
print(len(out),'open problems; expected AI resolutions in 3 months',round(out.p_ai_resolved_3mo.sum(),1)); print(out.head(10)[['rank','number','p_ai_resolved_3mo']])
