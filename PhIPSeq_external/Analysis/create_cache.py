import os
import numpy
import pandas

import PhIPSeq_external.config as config
base_path = config.ANALYSIS_PATH
input_path = os.path.join(base_path, "agilent_PNP_for_paper")
cache_path = os.path.join(base_path, "Cache")

if not os.path.exists(cache_path):
    os.makedirs(cache_path)

df = pandas.read_csv(os.path.join(input_path, 'cohort_info.csv'), index_col=0)
df.old_RegistrationCode = df.old_RegistrationCode.astype(str)

meta_cols = ['RegistrationCode', 'old_RegistrationCode', 'num_passed', 'Date']
sbj_cols = list(df.columns.difference(meta_cols)) + ['age']

base_df = df[df['RegistrationCode'] == df['old_RegistrationCode']].copy()
print("base_df with repeating individuals %d" % len(base_df))
base_df = base_df.reset_index().sort_values('Date').groupby('RegistrationCode').first()
print("base_df with without repeating individuals %d" % len(base_df))
base_df = base_df[base_df.num_passed > 200]
base_df['age'] = [int(base_df.loc[x].Date[:4])-base_df.loc[x].yob for x in base_df.index]

print("Base cohort %d subjects, age range %d-%d" % (len(base_df), base_df['age'].min(), base_df['age'].max()))

MB = pandas.read_csv(os.path.join(input_path, "MB.csv"), index_col=0)

dfs = {}
for i in base_df['index'].values:
    dfs[i] = pandas.read_csv(os.path.join(input_path, 'PhIPSeq_data', 'passing_%s.csv' % i), index_col=0)

p_df = {i: dfs[i].p_value for i in dfs.keys()}
p_df = pandas.DataFrame(p_df).T

p_df.T.to_pickle(os.path.join(cache_path, 'pval_agilent_above200.pkl'))

e_df = 1 * (p_df > 0)
e_df.T.to_pickle(os.path.join(cache_path, 'exist_agilent_above200.pkl'))

f_df = {i: dfs[i].fold_change for i in dfs.keys()}
f_df = pandas.DataFrame(f_df).T
f_df.fillna(0, inplace=True)
f_df[f_df < 0] = 0
f_df.T.to_pickle(os.path.join(cache_path, 'fold_agilent_above200.pkl'))

f_df[f_df == 0] = 1
f_df = numpy.log10(f_df)
f_df.T.to_pickle(os.path.join(cache_path, 'log_fold_agilent_above200.pkl'))

MB.loc[MB.index.intersection(base_df['index'].values)].T.to_pickle(os.path.join(cache_path, 'MB_agilent_above200.pkl'))
base_df.reset_index().set_index('index')[meta_cols].T.to_pickle(os.path.join(cache_path, 'meta_agilent_above200.pkl'))
base_df.reset_index().set_index('index')[sbj_cols].T.to_pickle(os.path.join(cache_path, 'sbj_info_above200.pkl'))

new_matched = df[df.RegistrationCode != df.old_RegistrationCode].index
MB_new_matched = MB.index.intersection(new_matched)
old_matched = base_df[numpy.isin(base_df.index, df.loc[new_matched].old_RegistrationCode.values)]['index'].values
MB_old_matched = base_df[numpy.isin(base_df.index, df.loc[MB_new_matched].old_RegistrationCode.values)]['index'].values

matched_df = df.loc[list(new_matched) + list(old_matched)]
matched_df.sort_values(['old_RegistrationCode', 'Date'], inplace=True)
matched_df['age'] = [int(matched_df.loc[x].Date[:4])-matched_df.loc[x].yob for x in matched_df.index]

MBmatched = df.loc[list(MB_new_matched) + list(MB_old_matched)].sort_values(['old_RegistrationCode', 'Date']).index

print("matched cohort age range: at start %d-%d, at end %d-%d" %
      (matched_df.iloc[::2]['age'].min(), matched_df.iloc[::2]['age'].max(),
       matched_df.iloc[1::2]['age'].min(), matched_df.iloc[1::2]['age'].max()))

dfs = {}
for i in matched_df.index:
    dfs[i] = pandas.read_csv(os.path.join(input_path, 'PhIPSeq_data', 'passing_%s.csv' % i), index_col=0)

p_df = {i: dfs[i].p_value for i in dfs.keys()}
p_df = pandas.DataFrame(p_df).T

p_df.T.to_pickle(os.path.join(cache_path, 'pval_matched_agilent_above200.pkl'))

e_df = 1 * (p_df > 0)
e_df.T.to_pickle(os.path.join(cache_path, 'exist_matched_agilent_above200.pkl'))

f_df = {i: dfs[i].fold_change for i in dfs.keys()}
f_df = pandas.DataFrame(f_df).T
f_df.fillna(0, inplace=True)
f_df[f_df < 0] = 0
f_df.T.to_pickle(os.path.join(cache_path, 'fold_matched_agilent_above200.pkl'))

f_df[f_df == 0] = 1
f_df = numpy.log10(f_df)
f_df.T.to_pickle(os.path.join(cache_path, 'log_fold_matched_agilent_above200.pkl'))

MB.loc[MBmatched].T.to_pickle(os.path.join(cache_path, 'MB_matched_agilent_above200.pkl'))
matched_df[meta_cols].T.to_pickle(os.path.join(cache_path, 'meta_matched_agilent_above200.pkl'))
matched_df[meta_cols].T.to_pickle(os.path.join(cache_path, 'sbj_info_matched_agilent_above200.pkl'))

df_info = pandas.read_csv(os.path.join(input_path, "library_content_info.csv"), index_col=0)
df_info.to_pickle(os.path.join(cache_path, "df_info_agilent.pkl"))
