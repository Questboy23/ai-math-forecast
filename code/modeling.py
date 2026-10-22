"""Shared modeling utilities: feature groups, models, expanding-window backtest, metrics, DeLong."""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from scipy import stats
warnings.filterwarnings('ignore')
BRIDGE=['d_near','d_mass','d_2hop','concept_cost']
HUB=['hub']
OBS_GROUPS={
 'prize':['log_prize'],'source_age':['src_age'],'formalization':['formal_before','formal_age','formalized_yes'],
 'statement_len':['stmt_len'],'status_edits':['edits','upd_age'],'oeis':['oeis'],'tags':['n_tags'],'prior_ai_attempts':['prior_ai'],
}
TAGCOLS=None
def groups(df):
    g=dict(OBS_GROUPS); g['tags']=g['tags']+[c for c in df if c.startswith('tag_') and df[c].std()>0]
    g['bridge_d_near']=['d_near'];g['bridge_d_mass']=['d_mass'];g['bridge_d_2hop']=['d_2hop'];g['bridge_concept_cost']=['concept_cost'];g['hub']=['hub']
    return g
def feats(df,which):
    g=groups(df); obs=sum([g[k] for k in OBS_GROUPS],[]); br=BRIDGE+HUB
    return {'obs':obs,'obs+bridge':obs+br,'all':obs+br}[which]
CS=[0.003,0.01,0.03,0.1,0.3,1,3]
def add_months(o,k):
    y=o.year+(o.month-1+k)//12; m=(o.month-1+k)%12+1; return pd.Timestamp(y,m,1)
def pick_C(Xtr,ytr,orig_tr,fixed=None):
    if fixed: return fixed
    os_=sorted(orig_tr.unique())
    if len(os_)<4: return 0.1
    best=None
    for C in CS:
        ll=[]
        for k in range(2,len(os_)):  # time-split: validate on origin k using origins <k whose window closed (k-3 months lag ignored for simplicity: use <=k-3)
            tr=orig_tr<=os_[max(k-3,0)] if k>=3 else orig_tr<os_[0]
            va=orig_tr==os_[k]
            if tr.sum()<50 or ytr[tr].sum()<3 or ytr[va].sum()<1: continue
            m=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=500)).fit(Xtr[tr],ytr[tr])
            ll.append(log_loss(ytr[va],m.predict_proba(Xtr[va])[:,1],labels=[0,1]))
        if ll:
            s=np.mean(ll)
            if best is None or s<best[0]: best=(s,C)
    return best[1] if best else 0.1
def make_model(kind,C):
    if kind=='lr': return make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=1000))
    return HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=150,l2_regularization=1.0,min_samples_leaf=20,random_state=0)
def backtest(D,test_origins,cols,kind='lr',fixedC=None,seed=0):
    """D has label column y, plus number, origin. Train rows: window (origin+3m) <= test origin."""
    preds=[];Cs={}
    for T in test_origins:
        tr=D[D.origin.map(lambda o:add_months(o,3))<=T]; te=D[D.origin==T]
        if tr.y.sum()<3: continue
        Xtr=tr[cols].values.astype(float); Xte=te[cols].values.astype(float)
        C=pick_C(Xtr,tr.y.values,tr.origin.reset_index(drop=True) if False else tr.origin,fixedC.get(T) if fixedC else None) if kind=='lr' else None
        Cs[T]=C
        m=make_model(kind,C).fit(Xtr,tr.y.values); p=m.predict_proba(Xte)[:,1]
        preds.append(te[['number','origin','y']].assign(p=p))
    return pd.concat(preds),Cs
def cluster_boot_auc(P,B=2000,seed=0,second=None):
    rng=np.random.default_rng(seed); nums=P.number.unique(); grp={n:g for n,g in P.groupby('number')}
    out=[]
    for _ in range(B):
        s=rng.choice(nums,len(nums)); d=pd.concat([grp[n] for n in s])
        if d.y.nunique()<2: continue
        a=roc_auc_score(d.y,d.p)
        out.append(a if second is None else a-roc_auc_score(d.y,d[second]))
    return np.percentile(out,[2.5,97.5])
def delong(y,p1,p2):
    """DeLong test for correlated AUCs. Returns auc1, auc2, z, p (two-sided)."""
    y=np.asarray(y); pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
    def comps(p):
        p=np.asarray(p); pp=p[pos]; pn=p[neg]
        psi=(pp[:,None]>pn[None,:]).astype(float)+0.5*(pp[:,None]==pn[None,:])
        return psi.mean(), psi.mean(1), psi.mean(0)
    a1,v10_1,v01_1=comps(p1); a2,v10_2,v01_2=comps(p2)
    m,n=len(pos),len(neg)
    S10=np.cov(np.vstack([v10_1,v10_2])); S01=np.cov(np.vstack([v01_1,v01_2]))
    S=S10/m+S01/n; var=S[0,0]+S[1,1]-2*S[0,1]
    z=(a1-a2)/np.sqrt(var) if var>0 else 0.0
    return a1,a2,z,2*(1-stats.norm.cdf(abs(z)))
def metrics(P):
    y=P.y.values;p=P.p.values
    prec=[]
    for o,g in P.groupby('origin'):
        prec.append(g.sort_values('p',ascending=False).head(20).y.mean())
    return dict(auc=roc_auc_score(y,p),prauc=average_precision_score(y,p),base=y.mean(),brier=brier_score_loss(y,p),p_at_20=float(np.mean(prec)),n=len(P),events=int(y.sum()))
def calib(P,bins=10):
    q=pd.qcut(P.p,bins,duplicates='drop'); g=P.groupby(q,observed=True).agg(pred=('p','mean'),obs=('y','mean'),n=('y','size')); return g.reset_index(drop=True)
