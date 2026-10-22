"""Merge features (no labels) and seal labels. Test labels (origins >= 2026-01-01) are written to a separate file opened only by run_backtest.py after PREREGISTRATION.md is hashed."""
import pandas as pd
T=pd.read_parquet('state/tabular.parquet'); B=pd.read_parquet('state/bridge_features.parquet'); R=pd.read_parquet('state/llm_rate.parquet').drop_duplicates('id').rename(columns={'id':'number','standard':'tech_standard'})
F=T.merge(B,on=['number','origin'],how='inner').merge(R[['number','tech_standard']],on='number',how='left')
F=F.drop_duplicates(['number','origin']).reset_index(drop=True); F.to_parquet('state/features_all.parquet')
L=pd.read_parquet('state/labels_all.parquet').drop_duplicates(['number','origin']).reset_index(drop=True)  # problem 1035 appears twice in problems.yaml
cut=pd.Timestamp('2026-01-01')
L[L.origin<cut].to_parquet('state/labels_train.parquet'); L[L.origin>=cut].to_parquet('state/labels_test_SEALED.parquet')
print(F.shape, F.isna().sum()[lambda x:x>0].to_dict(), 'train rows',(L.origin<cut).sum(),'test rows',(L.origin>=cut).sum())
