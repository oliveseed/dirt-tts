import numpy as np
import pandas as pd

df = pd.read_parquet('/home/appleseed/Desktop/frampton/data/emilia-yodas_300.parquet')
scores = pd.read_parquet('/home/appleseed/Desktop/frampton/data/emilia_spkr_synth_scores_300.parquet')

df['phone_density'] = df['phone_count'] / df['duration']
df_merged = pd.merge(df, scores, left_on='speaker', right_on='speaker_id')
df_merged = df_merged[df_merged['prob_mean'] < 0.1]
df_merged = df_merged[(df_merged['phone_density'] >= 5) & (df_merged['phone_density'] <= 25)]

df_clean_np = df_merged[['text', 'tokens']].values
np.save('emilia_300_clean.npy', df_clean_np)
# np.save('emilia_debug.npy', df_clean_np[:100, :])