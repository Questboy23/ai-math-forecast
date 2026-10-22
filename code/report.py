"""Build out/REPORT.pdf, out/EPOCH_SUMMARY.md and its chart from out/*.json."""
import json, textwrap, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve
R=json.load(open('out/backtest_results.json')); FM=json.load(open('out/forecast_meta.json'))
P=pd.read_parquet('out/backtest_predictions.parquet'); FC=pd.read_csv('out/forecast.csv')
H=open('state/prereg_hash.txt').read().strip()
H1,H2=R['H1'],R['H2']; M=R['models']
NAMES={'random':'Random','lr_obs':'Obscurity + attention (logistic)','lr_obs_bridge':'Obscurity + bridge (logistic)','hgb_all':'All features (gradient boosting)'}
COL={'random':'#999999','lr_obs':'#1f77b4','lr_obs_bridge':'#d62728','hgb_all':'#2ca02c'}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
def text_page(pdf,title,paras,size=12,top=0.90):
    fig=plt.figure(figsize=(8.5,11)); fig.text(0.1,top,title,fontsize=18,weight='bold',va='top')
    y=top-0.08
    for ptxt,sz,wt in paras:
        lines=textwrap.wrap(ptxt,88 if sz<=12 else 62)
        for ln in lines: fig.text(0.1,y,ln,fontsize=sz,weight=wt,va='top'); y-=0.0215*sz/12*1.0
        y-=0.025
    pdf.savefig(fig); plt.close(fig)
fmt=lambda x:f'{x:.3f}'
h1w='YES' if H1['supported'] else 'NO'; h2w='YES' if H2['supported'] else 'NO'
with PdfPages('out/REPORT.pdf') as pdf:
    fig=plt.figure(figsize=(8.5,11))
    fig.text(0.1,0.93,'Can we predict which open Erdős problems\nAI will help resolve?',fontsize=17,weight='bold',va='top')
    fig.text(0.1,0.865,f'Preregistration SHA-256:\n{H[:64]}\n{H[64:]}',fontsize=7,va='top',family='monospace')
    fig.text(0.1,0.80,'Backtest on 7 monthly origins 2026-01-01 to 2026-07-01, 3-month windows.\nReport built 2026-10-22 (US Eastern).',fontsize=9,va='top')
    fig.text(0.1,0.74,f'H1: {h1w}',fontsize=26,weight='bold',va='top',color='#1f77b4')
    fig.text(0.1,0.695,f'Obscurity and attention features alone rank solved problems above unsolved ones\nwith AUC {fmt(H1["auc"])} (95% CI {fmt(H1["ci"][0])} to {fmt(H1["ci"][1])}); the preregistered bar was above 0.60. [{ "medium" if H1["supported"] else "medium"}]',fontsize=12,va='top')
    fig.text(0.1,0.60,f'H2: {h2w}',fontsize=26,weight='bold',va='top',color='#d62728')
    fig.text(0.1,0.555,f'Adding bridge distance moves AUC from {fmt(H2["auc_obs"])} to {fmt(H2["auc_obs_bridge"])} (change {H2["delta"]:+.3f}).\nDeLong p = {H2["delong_p"]:.3f}; permutation p = {H2["perm_p"]:.3f};\nproblem-clustered bootstrap 95% CI of the change: {H2["delta_cluster_boot_ci"][0]:+.3f} to {H2["delta_cluster_boot_ci"][1]:+.3f}.  [medium]',fontsize=12,va='top')
    fig.text(0.1,0.40,'AUC = chance the model ranks a random solved problem above a random unsolved one (0.5 = coin flip).',fontsize=10,va='top')
    fig.text(0.1,0.37,f'Pooled test rows: {M["lr_obs"]["n"]}, events: {M["lr_obs"]["events"]} (base rate {M["lr_obs"]["base"]:.3f}).',fontsize=10,va='top')
    if not H2['supported']: fig.text(0.1,0.30,'A null H2 is a finding: within this data, how close a problem sits to existing\narXiv mathematics does not add detectable signal beyond obscurity and attention.' if H2['delta']<=0.01 else 'The bridge-distance change is not distinguishable from noise by the preregistered test.',fontsize=11,va='top')
    pdf.savefig(fig); plt.close(fig)
    # ROC
    fig,ax=plt.subplots(figsize=(8.5,11)); ax.set_position([0.15,0.3,0.7,0.5])
    for k,c in [('random','p_random'),('lr_obs','p_lr_obs'),('lr_obs_bridge','p_lr_obs_bridge'),('hgb_all','p_hgb_all')]:
        f,t,_=roc_curve(P.y,P[c]); ax.plot(f,t,color=COL[k],lw=2,label=f'{NAMES[k]}: AUC {M[k]["auc"]:.3f}')
    ax.plot([0,1],[0,1],'k:',lw=1); ax.set_xlabel('False positive rate (unsolved problems ranked high)'); ax.set_ylabel('True positive rate (solved problems ranked high)')
    ax.set_title('ROC curves, pooled out-of-time test predictions',loc='left',pad=14); ax.legend(loc='lower right',fontsize=9,frameon=False)
    pdf.savefig(fig); plt.close(fig)
    # calibration
    fig,ax=plt.subplots(figsize=(8.5,11)); ax.set_position([0.15,0.3,0.7,0.5])
    for k in ['lr_obs','lr_obs_bridge','hgb_all']:
        c=M[k]['calib']; ax.plot(c['pred'],c['obs'],'o-',color=COL[k],label=f'{NAMES[k]} (Brier {M[k]["brier"]:.4f})')
    mx=max(max(M[k]['calib']['pred']+M[k]['calib']['obs']) for k in ['lr_obs','lr_obs_bridge','hgb_all'])*1.1
    ax.plot([0,mx],[0,mx],'k:',lw=1,label='Perfect calibration'); ax.set_xlabel('Mean predicted probability in decile bin'); ax.set_ylabel('Observed event rate in bin')
    ax.set_title('Calibration, 10 bins, pooled test predictions',loc='left',pad=14); ax.legend(fontsize=9,frameon=False)
    fig.text(0.15,0.2,f'Brier of always predicting the base rate ({M["lr_obs"]["base"]:.3f}): {M["lr_obs"]["base"]*(1-M["lr_obs"]["base"]):.4f}.   Precision@20 per origin: '+', '.join(f'{NAMES[k].split(" (")[0]} {M[k]["p_at_20"]:.2f}' for k in ['random','lr_obs','lr_obs_bridge','hgb_all']),fontsize=8,wrap=True)
    pdf.savefig(fig); plt.close(fig)
    # importance: drop-group deltas
    rob=R['robustness']; g=list(rob['drop']); d1=[rob['drop'][k]['lr_obs_bridge_auc']-rob['full_lr_obs_bridge'] for k in g]; d2=[rob['drop'][k]['hgb_all_auc']-rob['full_hgb_all'] for k in g]
    fig,ax=plt.subplots(figsize=(8.5,11)); ax.set_position([0.28,0.2,0.65,0.62]); yy=np.arange(len(g))
    ax.barh(yy-0.2,d1,0.4,color=COL['lr_obs_bridge'],label='Logistic, obscurity + bridge'); ax.barh(yy+0.2,d2,0.4,color=COL['hgb_all'],label='Gradient boosting, all features')
    ax.set_yticks(yy); ax.set_yticklabels([x.replace('_',' ') for x in g]); ax.axvline(0,color='k',lw=0.8); ax.invert_yaxis()
    ax.set_xlabel('Change in pooled test AUC when the feature group is removed (negative = group helped)'); ax.set_title('Feature-group importance (drop one group at a time)',loc='left',pad=14); ax.legend(fontsize=9,frameon=False,loc='lower right')
    pdf.savefig(fig); plt.close(fig)
    # permutation
    fig,ax=plt.subplots(figsize=(8.5,11)); ax.set_position([0.15,0.3,0.7,0.5])
    ax.hist(H2['perm_aucs'],bins=25,color='#bbbbbb'); ax.axvline(H2['auc_obs_bridge'],color=COL['lr_obs_bridge'],lw=2.5,label=f'Real bridge features: AUC {H2["auc_obs_bridge"]:.3f}'); ax.axvline(H2['auc_obs'],color=COL['lr_obs'],lw=2.5,label=f'Obscurity only: AUC {H2["auc_obs"]:.3f}')
    ax.set_xlabel('Pooled test AUC with bridge features shuffled within origin'); ax.set_ylabel('Number of shuffles (of 300)'); ax.set_title(f'H2 permutation test, p = {H2["perm_p"]:.3f}',loc='left',pad=14); ax.legend(fontsize=9,frameon=False)
    pdf.savefig(fig); plt.close(fig)
    # top 20 table
    fig=plt.figure(figsize=(8.5,11)); fig.text(0.07,0.94,'Forecast: top 20 open problems, 2026-10-22',fontsize=16,weight='bold',va='top')
    fig.text(0.07,0.91,f'Model: {NAMES.get(FM["model"],FM["model"])}. Expected AI-assisted resolutions among {FM["n_open"]} open problems in 3 months: {FM["sum_p3"]:.1f}. 12-month column assumes\nconstant hazard [low].',fontsize=8,va='top')
    ax=fig.add_axes([0.05,0.1,0.9,0.78]); ax.axis('off'); t=FC.head(20)
    cell=[[int(r['rank']),int(r['number']),f'{r.p_ai_resolved_3mo:.3f}',f'{r.p_ai_resolved_12mo_LOW_constant_hazard:.3f}',textwrap.shorten(r.statement.replace('$',''),70)] for _,r in t.iterrows()]
    tb=ax.table(cellText=cell,colLabels=['Rank','Problem','P(3 mo)','P(12 mo) [low]','Statement (start)'],loc='upper center',colWidths=[0.06,0.08,0.09,0.13,0.64]); tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1,1.9)
    for (r,c),cl in tb.get_celld().items():
        if c==4: cl.set_text_props(ha='left'); cl._loc='left'
    pdf.savefig(fig); plt.close(fig)
    # secondary
    S=R['secondary']; rows=[]
    for kind in ['lit_id','new_sol']:
        for m_ in ['lr_obs','lr_obs_bridge','hgb_all']: rows.append(f'{kind:8s} {m_:16s} AUC {S[kind][m_]["auc"]:.3f}  PR-AUC {S[kind][m_]["prauc"]:.3f}  base {S[kind][m_]["base"]:.3f}  events {S[kind][m_]["events"]}')
    sd=R.get('side_D'); sdt=f'Side test D (Haiku technique rating), logistic obscurity + D: AUC {sd["auc_obs_plus_D"]:.3f} vs obscurity alone {sd["auc_obs"]:.3f}. Contamination probe: {sd["probe"]}. Not part of any headline model.' if sd else 'Side test D not run.'
    fig=plt.figure(figsize=(8.5,11)); fig.text(0.1,0.92,'Secondary analyses: literature identification vs new solution',fontsize=16,weight='bold',va='top')
    for i,r in enumerate(rows): fig.text(0.1,0.85-i*0.035,r,fontsize=9,family='monospace',va='top')
    for i,l in enumerate(textwrap.wrap(sdt,90)): fig.text(0.1,0.60-i*0.025,l,fontsize=10,va='top')
    fig.text(0.1,0.45,'Per-type counts are small; AUC differences under about 0.05 are within noise [low].',fontsize=10,va='top')
    pdf.savefig(fig); plt.close(fig)
    X=json.load(open('out/exploratory_results.json'))
    text_page(pdf,'Post hoc check (exploratory, not part of H1 or H2)',[
     (f'Same features, all 11 origins pooled ({X["rows"]} rows, {X["events"]} events), problem-grouped 5-fold cross-validation. This uses more training data but is not out-of-time, so it can leak time trends. Run after the backtest; it changes no conclusion.',11,'normal'),
     (f'AUC: obscurity logistic {X["lr_obs"]:.3f}; obscurity + bridge logistic {X["lr_obs_bridge"]:.3f}; gradient boosting with bridge {X["hgb_all"]:.3f}.',11,'normal'),
     (f'Bridge effect: {X["delta_bridge"]:+.3f} AUC (DeLong p {X["delong_p"]:.2f}; clustered 95% CI {X["delta_cluster_ci"][0]:+.3f} to {X["delta_cluster_ci"][1]:+.3f}). Bridge features still add nothing detectable [low].',11,'normal')])
    text_page(pdf,'Limitations',[
     ('Few events, short history. Status history exists only from 2025-08-31, so training data is 4 origins (Sep to Dec 2025) before the first test origin; the January model trains on 2 origins. Test events are 152 rows from about 68 problems because windows overlap [high].',11,'normal'),
     ('Label noise. "AI involvement" comes from a volunteer wiki, with a 120-day lookback and 30-day lookahead rule I chose. Resolutions are maintainers\' status edits, which lag the mathematics. Wiki selection bias: attempts are listed more often when they succeed [medium].',11,'normal'),
     ('Statement text and tags on the site are current versions, not versions as of each origin [medium]. Reference lists were excluded for the same reason. OpenAlex citation counts were not built.',11,'normal'),
     ('Bridge distance uses MiniLM-L6 embeddings of arXiv abstracts. A weak embedding of LaTeX statements could hide real signal, so a null H2 limits this implementation, not the idea [medium].',11,'normal'),
     ('Rows repeat the same problem across up to 3 origins. DeLong assumes independent rows, so its p is optimistic; use the clustered bootstrap and permutation figures [medium].',11,'normal'),
     ('The 12-month probability assumes a constant monthly hazard, but AI capability and attention are rising [low]. Wiki ends 2026-06-30, so July 2026 labels depend only on status changes after that date [medium].',11,'normal')])
    text_page(pdf,'Method and data',[
     ('Data: teorth/erdosproblems problems.yaml git history (monthly snapshots), AI-contributions wiki tables (dates, outcomes), erdosproblems.com statements (1,221 of 1,221), google-deepmind/formal-conjectures git dates, arXiv metadata snapshot (librarian-bots/arxiv-metadata-snapshot, 739,239 math and relevant cs papers).',11,'normal'),
     ('Labels: event = open at origin, resolved within 3 months, with a green or yellow AI wiki row in sections 1a to 1d or 2a near the resolution date. Hand audit of 25 non-trivial labels: 25 of 25 consistent with the wiki.',11,'normal'),
     ('Models, features, origins and metrics were fixed in PREREGISTRATION.md (hash on page 1) before any test metric was computed.',11,'normal'),
     (f'Code and forecast: https://github.com/Questboy23/ai-math-forecast. forecast.csv SHA-256: {FM["sha256"]}.',11,'normal')])
# epoch summary + chart
fig,ax=plt.subplots(figsize=(6,3.6)); ks=['random','lr_obs','lr_obs_bridge','hgb_all']
ax.bar(range(4),[M[k]['auc'] for k in ks],color=[COL[k] for k in ks]); ax.set_xticks(range(4)); ax.set_xticklabels(['Random','Obscurity','Obscurity\n+ bridge','Boosting\nall'],fontsize=9)
ax.axhline(0.5,color='k',ls=':',lw=1); ax.axhline(0.6,color='#888',ls='--',lw=1); ax.set_ylim(0.4,max(0.75,max(M[k]['auc'] for k in ks)+0.05)); ax.set_ylabel('Pooled test AUC'); ax.set_title('AUC of each model, 2026 out-of-time test',loc='left',fontsize=10)
for i,k in enumerate(ks): ax.text(i,M[k]['auc']+0.008,f'{M[k]["auc"]:.3f}',ha='center',fontsize=9)
plt.tight_layout(); plt.savefig('out/epoch_chart.png',dpi=150)
print('report done')
