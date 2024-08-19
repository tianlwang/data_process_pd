import os
import pickle as pkl
import pandas as pd
import numpy as np
pd.set_option('mode.chained_assignment',None)
os.makedirs('out/mortality',exist_ok=True)
df_all = pd.read_csv('out/checkpoints/df_all.csv')
df_all['Date'] = pd.to_datetime(df_all['Date'])
reversion = pkl.load(open('out/checkpoints/reversion.pkl', 'rb'))
reversion_time = pkl.load(open('out/checkpoints/reversion_time.pkl', 'rb'))
static_data = pkl.load(open('out/checkpoints/static_data.pkl', 'rb'))
# Construct mortality risk prediction dataset

# pipeline

threshold = 365
os.makedirs(f'out/mortality/threshold_{threshold}',exist_ok=True)
# step1. Generate dataframe subset of df_all according to the threshold, remove results with uncertain labels
cur_point = 0
df_all_threshold = pd.DataFrame()
for PDID, df_tmp in df_all.groupby('PDID'):
    df_tmp.sort_values(by='Date', inplace=True)
    tmp_reversion_time = reversion_time[PDID]
    tmp_reversion = reversion[PDID]
    if tmp_reversion == 0:
        delta_t = tmp_reversion_time - df_tmp['Date']
        delta_t = np.array([item.days for item in delta_t])
        df_tmp = df_tmp[delta_t > threshold]
    if len(df_tmp) != 0:
        df_all_threshold = pd.concat([df_all_threshold, df_tmp], axis=0)
df_all_threshold.to_csv(f'out/mortality/threshold_{threshold}/df_all_{threshold}.csv', encoding='utf_8_sig', index=False)


# step2. Calculate data statistics
feature_list = [col for col in df_all_threshold.columns if col not in ['PDID', 'Date']]
statistics_info = df_all_threshold[feature_list].describe()
statistics_info.to_csv(f'out/mortality/threshold_{threshold}/statistics_info.csv', encoding='utf_8_sig')

# step3. Z-score normalization
# step3. Z-score normalization
def zscore_normalization(df, columns, eps=1e-12):
    mean = df[column].mean(skipna=True)  
    std = df[column].std(skipna=True)
    return (df[column] - mean) / ( std + eps) 
feature_list = [col for col in df_all_threshold.columns if col not in ['PDID', 'Date', 'Respiratory System', 'Peritoneal Dialysis-Related Complications', 'Cardiovascular System', 'Digestive System', 'Acute Upper Respiratory Tract Infection', 'Peritoneal Dialysis-Related Peritonitis']]
for column in feature_list:
    df_all_threshold[column] = zscore_normalization(df_all_threshold, column)
df_all_threshold.to_csv(f'out/mortality/threshold_{threshold}/df_all_{threshold}_std.csv', encoding='utf_8_sig', index=False)
# step4. Perform missing value filling at the patient level and generate the dataset
feature_list = [col for col in df_all_threshold.columns if col not in ['PDID', 'Date']]
df_median = df_all_threshold[feature_list].median()
basetime = pd.to_datetime(pd.Series('1900-01-01'))[0]
PDID2idx = {}
x_list = []
x_static_list = []
y_list = []
missing_mask_list = []
timestamp_list = []

for idx, (PDID, df_tmp) in enumerate(df_all_threshold.groupby('PDID')):
    df_tmp.sort_values(by='Date', inplace=True)
    tmp_reversion = reversion[PDID]
    tmp_reversion_time = reversion_time[PDID]
    tmp_y = []
    tmp_timestamp = []

    for i in range(len(df_tmp)):
        cur_time = df_tmp.iloc[i]['Date']
        delta_t = (tmp_reversion_time - cur_time).days
        cur_timestamp = (df_tmp.iloc[i]['Date'] - basetime).days
        # Patient ultimately dies: y is 1 within the death date threshold, 0 outside the threshold
        # Patient ultimately survives: y is 0 (records with uncertain labels have been removed)
        if tmp_reversion == 1:
            if delta_t <= threshold:
                tmp_y.append(1)
            else:
                tmp_y.append(0)
        else:
            tmp_y.append(0)
        tmp_timestamp.append(cur_timestamp)

    y_list.append(tmp_y)
    timestamp_list.append([item - tmp_timestamp[0] for item in tmp_timestamp])  # Privacy protection

    tmp_missing = df_tmp[feature_list].isnull()
    tmp_missing = (tmp_missing == False).astype('uint8')
    missing_mask_list.append(tmp_missing.values.tolist())

    df_tmp = df_tmp.ffill()  # Forward fill
    for col in feature_list:  # Then fill with median of the entire dataset
        df_tmp[col] = df_tmp[col].fillna(df_median[col])
    x_list.append(df_tmp[feature_list].values.tolist())

    tmp_static = static_data[PDID]
    x_static_list.append(tmp_static)
    PDID2idx[PDID] = idx
idx2PDID = {v: k for k, v in PDID2idx.items()}
pkl.dump(x_list, open(os.path.join('out/mortality', f'threshold_{threshold}', 'x.pkl'), 'wb'))
pkl.dump(x_static_list, open(os.path.join('out/mortality', f'threshold_{threshold}', 'x_static.pkl'), 'wb'))
pkl.dump(y_list, open(os.path.join('out/mortality', f'threshold_{threshold}', 'y.pkl'), 'wb'))
pkl.dump(missing_mask_list, open(os.path.join('out/mortality', f'threshold_{threshold}', 'missing_mask.pkl'), 'wb'))
pkl.dump(timestamp_list, open(os.path.join('out/mortality', f'threshold_{threshold}', 'timestamp.pkl'), 'wb'))
pkl.dump(PDID2idx, open(os.path.join('out/mortality', f'threshold_{threshold}', 'PDID2idx.pkl'), 'wb'))
pkl.dump(idx2PDID, open(os.path.join('out/mortality', f'threshold_{threshold}', 'idx2PDID.pkl'), 'wb'))

print('Mortality dataset construction completed!')