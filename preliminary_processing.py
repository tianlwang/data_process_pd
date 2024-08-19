import os
import pandas as pd
import numpy as np
import pickle
pd.set_option('mode.chained_assignment',None)
os.makedirs('out/checkpoints/',exist_ok=True)

# file path 
base_info_path = 'data/PatientBasicInformation-NoName.xlsx'
labtest_path = 'data/LaboratoryTestData-NoName.xlsx'
dialysis_adequacy_path = 'data/DialysisAdequacyData-NoName.xlsx'
dialysis_protocol_path = 'data/DialysisProtocolData-NoName.xlsx'
dialysis_evaluation_path = 'data/DialysisEvaluationData-NoName.xlsx'
complication_path = 'data/ComplicationData-NoName.xlsx'
## 1.1 Basic Information Table
df_base_info = pd.read_excel(base_info_path)
print(df_base_info.columns.tolist())
print(f'Number of patients {len(df_base_info)}')
print(f'Number of features {len(df_base_info.columns.tolist())}')

# step1. Select feature subset
base_info_feature = ['PDID', 'Date of Birth', 'Catheter Implantation Time', 'Gender', 'Diabetes', 'Primary Disease']
df_base_info = df_base_info[base_info_feature]

# step2. Ensure PDID, Date of Birth, Catheter Implantation Time, Gender, Diabetes have no missing values
df_base_info.dropna(subset=['PDID', 'Date of Birth', 'Catheter Implantation Time', 'Gender', 'Diabetes'], inplace=True)
print(f'Number of patients {len(df_base_info)}')

# step 3. Get static features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_static_feature = df_base_info[['PDID', 'Gender', 'Diabetes', 'Primary Disease']]
df_static_feature['Primary Disease'] = df_static_feature['Primary Disease'].fillna('Missing')
df_static_feature['Gender'] = le.fit_transform(df_static_feature['Gender'])
df_static_feature['Diabetes'] = le.fit_transform(df_static_feature['Diabetes'])
others_list = []
static_data = {}
for i in range(len(df_static_feature)):
    tmp_data = df_static_feature.iloc[i]
    static_data[tmp_data['PDID']] = [tmp_data['Gender'], tmp_data['Diabetes']]
    key_list = ['Glomerulonephritis', 'Hypertension', 'Chronic Interstitial Nephritis', 'Polycystic', 'IGA']
    flag = 0
    if 'Diabetes' in tmp_data['Primary Disease'] or 'DN' in tmp_data['Primary Disease']:
        static_data[tmp_data['PDID']].append(1)
        flag = 1
    else:
        static_data[tmp_data['PDID']].append(0)
    for key_word in key_list:
        if key_word in tmp_data['Primary Disease']:
            static_data[tmp_data['PDID']].append(1)
            flag = 1
        else:
            static_data[tmp_data['PDID']].append(0)
    if flag == 0:
        others_list.append(tmp_data['Primary Disease'])

df_base_info.to_csv('out/checkpoints/df_base_info.csv', encoding='utf_8_sig', index=False)
pickle.dump(static_data, open('out/checkpoints/static_data.pkl', 'wb'))

print(f'Proportion of primary diseases not covered: {len(others_list) / len(df_static_feature):.3f}')

# step 4. Get date of birth, implantation time as baseline time
df_basetime = df_base_info[['PDID', 'Date of Birth', 'Catheter Implantation Time']]

basetime1 = {}
basetime2 = {}
for i in range(len(df_basetime)):
    tmp = df_basetime.iloc[i]
    basetime1[tmp['PDID']] = tmp['Date of Birth']
    basetime2[tmp['PDID']] = tmp['Catheter Implantation Time']
df_base_info.to_csv('out/checkpoints/df_base_info.csv', encoding='utf_8_sig', index=False)
df_base_info.to_csv('out/checkpoints/df_base_info.csv', encoding='utf_8_sig', index=False)
## 1.2 Labtest
df_labtest = pd.read_excel(labtest_path)
print(df_labtest.columns.tolist())
print(f'Number of labtests: {len(df_labtest)}, Number of PDID: {len(set(df_labtest.PDID))}, Number of features: {len(df_labtest.columns)}')

# step1. Ensure PDID, Date have no missing values
df_labtest.dropna(subset=['PDID', 'Date'], inplace=True)
print(f'after step 1: Number of labtests: {len(df_labtest)}, Number of PDID: {len(set(df_labtest.PDID))}, Number of features: {len(df_labtest.columns)}')

# step2. Remove records with PDID not in the basic information table
df_labtest = df_labtest.loc[[PDID in df_base_info['PDID'].tolist() for PDID in df_labtest['PDID']]]
df_labtest = df_labtest[[column for column in df_labtest.columns if column != 'ID']]
print(f'after step 2: Number of labtests: {len(df_labtest)}, Number of PDID: {len(set(df_labtest.PDID))}, Number of features: {len(df_labtest.columns)}')

# step3. Merge records of the same day
df_labtest = df_labtest.groupby(['PDID', 'Date'], as_index=False, dropna=True).mean()
print(f'after step 3: Number of labtests: {len(df_labtest)}, Number of PDID: {len(set(df_labtest.PDID))}, Number of features: {len(df_labtest.columns)}')
df_labtest.to_csv('out/checkpoints/df_labtest.csv',encoding='utf_8_sig',index=False)
## 1.3 Dialysis Adequacy
df_dialysis_adequacy = pd.read_excel(dialysis_adequacy_path)
print(df_dialysis_adequacy.columns.tolist())
print(f'Number of dialysis_adequacy: {len(df_dialysis_adequacy)}, Number of PDID: {len(set(df_dialysis_adequacy.PDID))}, Number of features: {len(df_dialysis_adequacy.columns)}')

# step1. Ensure PDID, Date have no missing values
df_dialysis_adequacy.dropna(subset=['PDID', 'Date'], inplace=True)
print(f'after step 1: Number of dialysis_adequacy: {len(df_dialysis_adequacy)}, Number of PDID: {len(set(df_dialysis_adequacy.PDID))}, Number of features: {len(df_dialysis_adequacy.columns)}')

# step2. Remove records with PDID not in the basic information table
df_dialysis_adequacy = df_dialysis_adequacy.loc[[PDID in df_base_info['PDID'].tolist() for PDID in df_dialysis_adequacy['PDID']]]
print(f'after step 2: Number of dialysis_adequacy: {len(df_dialysis_adequacy)}, Number of PDID: {len(set(df_dialysis_adequacy.PDID))}, Number of features: {len(df_dialysis_adequacy.columns)}')

# step3. Merge records of the same day
df_dialysis_adequacy = df_dialysis_adequacy.groupby(['PDID', 'Date'], as_index=False, dropna=True).mean()
print(f'after step 3: Number of dialysis_adequacy: {len(df_dialysis_adequacy)}, Number of PDID: {len(set(df_dialysis_adequacy.PDID))}, Number of features: {len(df_dialysis_adequacy.columns)}')

# step4. Calculate relevant indicators based on dialysis adequacy formula
sex = {}
for i in range(len(df_base_info)):
    tmp = df_base_info.iloc[i]
    if tmp['Gender'] == 'Male':
        sex[tmp['PDID']] = 1
    elif tmp['Gender'] == 'Female':
        sex[tmp['PDID']] = 2
    else:
        print("missing")
age_list = []
for i in range(len(df_dialysis_adequacy)):
    tmp = df_dialysis_adequacy.iloc[i]
    age_list.append((tmp['Date'] - basetime1[tmp['PDID']]).days / 365)
df_dialysis_adequacy['Age'] = age_list
fc = lambda x: sex[x]
sex_list = [fc(x) for x in df_dialysis_adequacy.PDID.tolist()]
df_dialysis_adequacy['Gender'] = sex_list

formula_use_feature = ['Age', 'Gender', 'Height', 'Actual Weight', 'Blood Urea', 'Blood Creatinine', '24-hour Urine Volume', '24-hour Urine Urea', '24-hour Urine Creatinine', '24-hour Dialysate Volume', '24-hour Dialysate Urea', '24-hour Dialysate Creatinine']
df_calculate_adequacy = df_dialysis_adequacy[formula_use_feature]
# # Set urine urea and urine creatinine to 0 for records with urine volume of 0
# df_calculate_adequacy['24-hour Urine Urea'] = df_calculate_adequacy['24-hour Urine Urea'].fillna(0)
# df_calculate_adequacy['24-hour Urine Creatinine'] = df_calculate_adequacy['24-hour Urine Creatinine'].fillna(0)

weight = [[], []]
BSA = [[], []]
V = [[], []]
GFR = [[], []]
Krt = [[], []]
Kpt = [[], []]
Kt = [[], []]
CrCr = [[], []]
CpCr = [[], []]
CCr = [[], []]
nPNA = [[], []]
for i in range(len(df_calculate_adequacy)):
    tmp = df_calculate_adequacy.iloc[i]
    if tmp['Gender'] == 1:
        weight[0].append(tmp['Height'] - 105)
    elif tmp['Gender'] == 2:
        weight[0].append(tmp['Height'] - 110)
    weight[1].append(tmp['Actual Weight'])
    BSA[0].append(0.007184 * tmp['Height']**0.725 * tmp['Actual Weight']**0.425)
    BSA[1].append(0.007184 * tmp['Height']**0.725 * tmp['Actual Weight']**0.425)
    if tmp['Gender'] == 1:
        V[0].append(2.477 + (0.3362 * weight[0][-1]) + (0.1074 * tmp['Height']) - (0.09516 * tmp['Age']))
        V[1].append(2.477 + (0.3362 * tmp['Actual Weight']) + (0.1074 * tmp['Height']) - (0.09516 * tmp['Age']))
    elif tmp['Gender'] == 2:
        V[0].append(-2.097 + (0.2466 * weight[0][-1]) + (0.1069 * tmp['Height']))
        V[1].append(-2.097 + (0.2466 * tmp['Actual Weight']) + (0.1069 * tmp['Height']))
    GFR[0].append((tmp['24-hour Urine Urea'] * tmp['24-hour Urine Volume'] / tmp['Blood Urea'] + tmp['24-hour Urine Creatinine'] * tmp['24-hour Urine Volume'] * 1.73 / tmp['Blood Creatinine'] / BSA[0][-1]) / 2 / 1440)
    GFR[1].append((tmp['24-hour Urine Urea'] * tmp['24-hour Urine Volume'] / tmp['Blood Urea'] + tmp['24-hour Urine Creatinine'] * tmp['24-hour Urine Volume'] * 1.73 / tmp['Blood Creatinine'] / BSA[1][-1]) / 2 / 1440)
    Krt[0].append((tmp['24-hour Urine Urea'] * tmp['24-hour Urine Volume'] / tmp['Blood Urea'] / V[0][-1] / 1000) * 7)
    Krt[1].append((tmp['24-hour Urine Urea'] * tmp['24-hour Urine Volume'] / tmp['Blood Urea'] / V[1][-1] / 1000) * 7)
    Kpt[0].append((tmp['24-hour Dialysate Urea'] * tmp['24-hour Dialysate Volume'] / tmp['Blood Urea'] / V[0][-1] / 1000) * 7)
    Kpt[1].append((tmp['24-hour Dialysate Urea'] * tmp['24-hour Dialysate Volume'] / tmp['Blood Urea'] / V[1][-1] / 1000) * 7)
    Kt[0].append(Krt[0][-1] + Kpt[0][-1])
    Kt[1].append(Krt[1][-1] + Kpt[1][-1])
    CrCr[0].append(((tmp['24-hour Urine Creatinine'] * tmp['24-hour Urine Volume'] / tmp['Blood Creatinine'] / 1000 + tmp['24-hour Urine Urea'] * tmp['24-hour Urine Volume'] / tmp['Blood Urea'] / 1000) / 2) * 1.73 / BSA[0][-1] * 7)
    CrCr[1].append(((tmp['24-hour Urine Creatinine'] * tmp['24-hour Urine Volume'] / tmp['Blood Creatinine'] / 1000 + tmp['24-hour Urine Urea'] * tmp['24-hour Urine Volume'] / tmp['Blood Urea'] / 1000) / 2) * 1.73 / BSA[1][-1] * 7)
    CpCr[0].append((tmp['24-hour Dialysate Creatinine'] * tmp['24-hour Dialysate Volume'] / tmp['Blood Creatinine'] / 1000) * 1.73 / BSA[0][-1] * 7)
    CpCr[1].append((tmp['24-hour Dialysate Creatinine'] * tmp['24-hour Dialysate Volume'] / tmp['Blood Creatinine'] / 1000) * 1.73 / BSA[1][-1] * 7)
    CCr[0].append(CrCr[0][-1] + CpCr[0][-1])
    CCr[1].append(CrCr[1][-1] + CpCr[1][-1])

df_adequacy = df_dialysis_adequacy[['PDID', 'Date']]
df_adequacy['weight_0'] = weight[0]
df_adequacy['BSA_0'] = BSA[0]
df_adequacy['V_0'] = V[0]
df_adequacy['GFR_0'] = GFR[0]
df_adequacy['Krt_0'] = Krt[0]
df_adequacy['Kpt_0'] = Kpt[0]
df_adequacy['Kt_0'] = Kt[0]
df_adequacy['CrCr_0'] = CrCr[0]
df_adequacy['CpCr_0'] = CpCr[0]
df_adequacy['CCr_0'] = CCr[0]
df_adequacy['weight_1'] = weight[1]
df_adequacy['BSA_1'] = BSA[1]
df_adequacy['V_1'] = V[1]
df_adequacy['GFR_1'] = GFR[1]
df_adequacy['Krt_1'] = Krt[1]
df_adequacy['Kpt_1'] = Kpt[1]
df_adequacy['Kt_1'] = Kt[1]
df_adequacy['CrCr_1'] = CrCr[1]
df_adequacy['CpCr_1'] = CpCr[1]
df_adequacy['CCr_1'] = CCr[1]

print(f'after step 4: Number of dialysis_adequacy: {len(df_adequacy)}, Number of PDID: {len(set(df_adequacy.PDID))}, Number of features: {len(df_dialysis_adequacy.columns)}')
df_adequacy.to_csv('out/checkpoints/df_adequacy.csv', encoding='utf_8_sig', index=False)
## 1.4 Dialysis Protocol
df_dialysis_protocol = pd.read_excel(dialysis_protocol_path)
print(df_dialysis_protocol.columns.tolist())
print(f'Number of dialysis_protocol: {len(df_dialysis_protocol)}, Number of PDID: {len(set(df_dialysis_protocol.PDID))}, Number of features: {len(df_dialysis_protocol.columns)}')

# step1. Ensure PDID, Date have no missing values
df_dialysis_protocol.dropna(subset=['PDID', 'Date'], inplace=True)
print(f'after step 1: Number of dialysis_protocol: {len(df_dialysis_protocol)}, Number of PDID: {len(set(df_dialysis_protocol.PDID))}, Number of features: {len(df_dialysis_protocol.columns)}')

# step2. Remove records with PDID not in the basic information table
df_dialysis_protocol = df_dialysis_protocol.loc[[PDID in df_base_info['PDID'].tolist() for PDID in df_dialysis_protocol['PDID']]]
print(f'after step 2: Number of dialysis_protocol: {len(df_dialysis_protocol)}, Number of PDID: {len(set(df_dialysis_protocol.PDID))}, Number of features: {len(df_dialysis_protocol.columns)}')

# step3. Merge records of the same day dialysis, calculate dialysis dose and glucose exposure (multiply horizontally, add vertically)
PDID_list = []
date_list = []
dosage_list = []  # Dialysate dose
for _, tmp in df_dialysis_protocol.groupby(['PDID', 'Date']):
    x1 = 0
    for j in range(len(tmp)):
        x1 += tmp.iloc[j]['Dialysate Dose'] * tmp.iloc[j]['Dialysis Frequency']
    PDID_list.append(tmp.iloc[-1]['PDID'])
    date_list.append(tmp.iloc[-1]['Date'])
    dosage_list.append(x1)

df_protocol = pd.DataFrame({'PDID': PDID_list, 'Date': date_list, 'Dialysate Dose': dosage_list})
print(f'after step 3: Number of dialysis_protocol: {len(df_protocol)}, Number of PDID: {len(set(df_protocol.PDID))}, Number of features: {len(df_protocol.columns)}')

df_protocol.to_csv('out/checkpoints/df_protocol.csv',encoding='utf_8_sig',index=False)
## 1.5 Dialysis Evaluation
df_dialysis_evaluation = pd.read_excel(dialysis_evaluation_path)
print(df_dialysis_evaluation.columns.tolist())
print(f'Number of dialysis_evaluation: {len(df_dialysis_evaluation)}, Number of PDID: {len(set(df_dialysis_evaluation.PDID))}, Number of features: {len(df_dialysis_evaluation.columns)}')

# step1. Ensure PDID, Date have no missing values
df_dialysis_evaluation.dropna(subset=['PDID', 'Date'], inplace=True)
print(f'after step 1: Number of dialysis_evaluation: {len(df_dialysis_evaluation)}, Number of PDID: {len(set(df_dialysis_evaluation.PDID))}, Number of features: {len(df_dialysis_evaluation.columns)}')

# step2. Remove records with PDID not in the basic information table
df_dialysis_evaluation = df_dialysis_evaluation.loc[[PDID in df_base_info['PDID'].tolist() for PDID in df_dialysis_evaluation['PDID']]]
print(f'after step 2: Number of dialysis_evaluation: {len(df_dialysis_evaluation)}, Number of PDID: {len(set(df_dialysis_evaluation.PDID))}, Number of features: {len(df_dialysis_evaluation.columns)}')

# step3. Keep part of the features
evaluation_use_feature = ['PDID', 'Date', 'Home Systolic Blood Pressure', 'Home Diastolic Blood Pressure', 'Heart Rate', 'Actual Weight', 'Edema', 'Urine Volume', 'Ultrafiltration Volume']
df_dialysis_evaluation = df_dialysis_evaluation[evaluation_use_feature]
print(f'after step 3: Number of dialysis_evaluation: {len(df_dialysis_evaluation)}, Number of PDID: {len(set(df_dialysis_evaluation.PDID))}, Number of features: {len(df_dialysis_evaluation.columns)}')

# step4. Merge records of the same day dialysis
df_evaluation = df_dialysis_evaluation.groupby(['PDID', 'Date'], as_index=False, dropna=True).mean()
print(f'after step 4: Number of dialysis_evaluation: {len(df_dialysis_evaluation)}, Number of PDID: {len(set(df_dialysis_evaluation.PDID))}, Number of features: {len(df_dialysis_evaluation.columns)}')
df_evaluation.to_csv('out/checkpoints/df_evaluation.csv',encoding='utf_8_sig',index=False)
## 1.6 Complications
df_complication = pd.read_excel(complication_path)
print(df_complication.columns.tolist())
print(f'Number of complications: {len(df_complication)}, Number of PDID: {len(set(df_complication.PDID))}, Number of features: {len(df_complication.columns)}')

# step1. Ensure PDID, Date have no missing values
df_complication.dropna(subset=['PDID', 'Date of Onset'], inplace=True)
print(f'after step 1: Number of complications: {len(df_complication)}, Number of PDID: {len(set(df_complication.PDID))}, Number of features: {len(df_complication.columns)}')

# step2. Remove records with PDID not in the basic information table
df_complication = df_complication.loc[[PDID in df_base_info['PDID'].tolist() for PDID in df_complication['PDID']]]
print(f'after step 2: Number of complications: {len(df_complication)}, Number of PDID: {len(set(df_complication.PDID))}, Number of features: {len(df_complication.columns)}')

# step3. Keep part of the features
complication_date = df_complication['Date of Onset']
df_tmp = df_complication[['PDID', 'Disease Category', 'Disease Name']]
df_tmp['Date'] = complication_date
category_list = df_tmp['Disease Category'].tolist()
category1_list = [1 if item == 'Respiratory System' else 0 for item in category_list]
category2_list = [1 if item == 'Peritoneal Dialysis-Related Complications' else 0 for item in category_list]
category3_list = [1 if item == 'Cardiovascular System' else 0 for item in category_list]
category4_list = [1 if item == 'Digestive System' else 0 for item in category_list]
name_list = df_tmp['Disease Name'].tolist()
name1_list = [1 if item == 'Acute Upper Respiratory Tract Infection' else 0 for item in name_list]
name2_list = [1 if item == 'Acute Upper Respiratory Tract Infection' else 0 for item in name_list]
df_tmp = df_tmp[['PDID', 'Date']]
df_tmp['Respiratory System'] = category1_list
df_tmp['Peritoneal Dialysis-Related Complications'] = category2_list
df_tmp['Cardiovascular System'] = category3_list
df_tmp['Digestive System'] = category4_list
df_tmp['Acute Upper Respiratory Tract Infection'] = name1_list
df_tmp['Peritoneal Dialysis-Related Peritonitis'] = name2_list

# There may be multiple records in one day, complications and other information are handled separately
df_tmp1 = df_tmp[['PDID', 'Date']].drop_duplicates(subset=['PDID', 'Date'])
df_tmp2 = df_tmp[['PDID', 'Date', 'Respiratory System', 'Peritoneal Dialysis-Related Complications', 'Cardiovascular System', 'Digestive System', 'Acute Upper Respiratory Tract Infection', 'Peritoneal Dialysis-Related Peritonitis']].groupby(['PDID', 'Date'], as_index=False, dropna=False).sum()
df_tmp2 = df_tmp2[['Respiratory System', 'Peritoneal Dialysis-Related Complications', 'Cardiovascular System', 'Digestive System', 'Acute Upper Respiratory Tract Infection', 'Peritoneal Dialysis-Related Peritonitis']]
df_tmp1.reset_index(drop=True, inplace=True)
df_tmp2.reset_index(drop=True, inplace=True)
df_tmp = pd.concat([df_tmp1, df_tmp2], axis=1)
df_complication = df_tmp

print(f'after step 3: Number of complications: {len(df_complication)}, Number of PDID: {len(set(df_complication.PDID))}, Number of features: {len(df_complication.columns)}')
df_complication.to_csv('out/checkpoints/df_complication.csv',encoding='utf_8_sig',index=False)
# 2. Merge labtest-related tables, dialysis-related tables, and complication table
# step 1. Merge each table
df_tmp = pd.merge(df_labtest, df_adequacy, how='outer', on=['PDID', 'Date'])
df_tmp = pd.merge(df_tmp, df_protocol, how='outer', on=['PDID', 'Date'])
df_tmp = pd.merge(df_tmp, df_evaluation, how='outer', on=['PDID', 'Date'])
df_tmp = pd.merge(df_tmp, df_complication, how='outer', on=['PDID', 'Date'])
df_tmp.sort_values(by=['PDID', 'Date'], inplace=True)
df_all = df_tmp
print(f'after step 1: Number of all: {len(df_all)}, Number of PDID: {len(set(df_all.PDID))}, Number of features: {len(df_all.columns)}')

# step 2. Remove rows without records
df_all.dropna(how='all', subset=[feature for feature in df_all.columns if feature not in ['PDID', 'Date']], inplace=True)
print(f'after step 2: Number of all: {len(df_all)}, Number of PDID: {len(set(df_all.PDID))}, Number of features: {len(df_all.columns)}')

# step 3. Add 2 features
to_basetime1 = []
to_basetime2 = []
for i in range(len(df_tmp)):
    tmp = df_tmp.iloc[i]
    to_basetime1.append((tmp['Date'] - basetime1[tmp['PDID']]).days)
    to_basetime2.append((tmp['Date'] - basetime2[tmp['PDID']]).days)

df_all['to_basetime1'] = to_basetime1
df_all['to_basetime2'] = to_basetime2
df_all['to_basetime1'] = df_all['to_basetime1'] / 365
df_all['to_basetime2'] = df_all['to_basetime2'] / 365
df_all.rename(columns={'to_basetime1': 'Age'}, inplace=True)
df_all.rename(columns={'to_basetime2': 'Dialysis Time'}, inplace=True)

print(f'after step 3: Number of all: {len(df_all)}, Number of PDID: {len(set(df_all.PDID))}, Number of features: {len(df_all.columns)}')

# 3. Get patient outcomes
# step1 Complete outcome information
df_reversion = pd.read_excel(base_info_path)[['PDID', 'Exit Peritoneal Dialysis Time', 'Reason for Exit', 'Outcome']]
na_reason = df_reversion['Reason for Exit'].isnull()
na_outcome = df_reversion['Outcome'].isnull()
indices = (~na_reason & na_outcome)  # Has exit reason but no outcome
df_reversion.loc[indices, 'Outcome'] = df_reversion.loc[indices, 'Reason for Exit']

# step2 Select required patient information
indices = df_reversion['PDID'].isin(df_all['PDID'])
df_reversion = df_reversion[indices]

na_exit_time = df_reversion['Exit Peritoneal Dialysis Time'].isnull()
na_outcome = df_reversion['Outcome'].isnull()
print(f'All patients who exited peritoneal dialysis have (1) exit time (2) exit reason: {(na_exit_time != na_outcome).sum() == 0}')

# step3 Use the last time point in the dataset to complete the outcome time (patients who did not exit peritoneal dialysis are assumed to be still receiving treatment at the end of the window)
df_reversion['Exit Peritoneal Dialysis Time'] = df_reversion['Exit Peritoneal Dialysis Time'].fillna(df_all['Date'].max())
df_reversion['Outcome'] = df_reversion['Outcome'].fillna('Still receiving treatment')

# step4 Outcome result: death as one category, others as another, record the outcome time of each patient
death_filter = [("death" in item or 'Multiple Organ Failure' in item or "General Failure" in item or "Systemic Organ Failure" in item or "Shock" in item) and "Lost Contact" not in item for item in df_reversion.Outcome]
df_death = df_reversion[death_filter]
df_others = df_reversion[~np.array(death_filter)]
reversion = {}
reversion_time = {}
for i in range(len(df_death)):
    tmp = df_death.iloc[i]
    reversion[tmp['PDID']] = 1
    reversion_time[tmp['PDID']] = tmp['Exit Peritoneal Dialysis Time']
for i in range(len(df_others)):
    tmp = df_others.iloc[i]
    reversion[tmp['PDID']] = 0
    reversion_time[tmp['PDID']] = tmp['Exit Peritoneal Dialysis Time']
pickle.dump(reversion, open('out/checkpoints/reversion.pkl','wb'))
pickle.dump(reversion_time, open('out/checkpoints/reversion_time.pkl','wb'))
# 4. Handle conflicting data
# step1 Handle records after the outcome time point
contradictory_PDID_1_list = []
contradictory_PDID_2_list = []
df_all_new = pd.DataFrame()
for PDID, tmp_df in df_all.groupby('PDID'):
    tmp_df.sort_values(by='Date', inplace=True)
    if tmp_df.iloc[-1]['Date'] > reversion_time[PDID]:
        if reversion[PDID] == 1:
            contradictory_PDID_1_list.append(PDID)
            contradictory_PDID_2_list.append(PDID)
            df_all_new = pd.concat([df_all_new, tmp_df[tmp_df['Date'] < reversion_time[PDID]]], axis=0)
    else:
        df_all_new = pd.concat([df_all_new, tmp_df], axis=0)
pd.DataFrame({'PDID': contradictory_PDID_1_list}).to_csv('out/checkpoints/contradictory_PDID_1.csv', index=False)
pd.DataFrame({'PDID': contradictory_PDID_2_list}).to_csv('out/checkpoints/contradictory_PDID_2.csv', index=False)
df_all = df_all_new

# step2 Delete records before 2011-01-01
start_time = pd.to_datetime(pd.Series('2011-01-01'))[0]
df_all = df_all.loc[[date > start_time for date in df_all['Date']]]
df_all.to_csv('out/checkpoints/df_all.csv',encoding='utf_8_sig',index=False)