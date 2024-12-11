import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Constants
FILE_PATH = '../data/sepsis_01NVa_Sep2024_CHENYANG.xlsx'
HRV_FEATURES = [
    'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN',
    'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_SDRMSSD', 'HRV_Prc20NN', 'HRV_Prc80NN',
    'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN', 'HRV_HF',
    'HRV_VHF', 'HRV_TP', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_S',
    'HRV_CSI', 'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP', 'HRV_IALS', 'HRV_PSS', 'HRV_PAS',
    'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_SD1d', 'HRV_SD1a',
    'HRV_C2d', 'HRV_C2a', 'HRV_SD2d', 'HRV_SD2a', 'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd',
    'HRV_SDNNa', 'HRV_DFA_alpha1', 'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
    'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max', 'HRV_MFDFA_alpha1_Delta',
    'HRV_MFDFA_alpha1_Asymmetry', 'HRV_MFDFA_alpha1_Fluctuation',
    'HRV_MFDFA_alpha1_Increment', 'HRV_ApEn', 'HRV_SampEn', 'HRV_ShanEn', 'HRV_FuzzyEn',
    'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC'
]
PPG_ADDITIONAL_FEATURES = ['PAV']
ECG_ADDITIONAL_FEATURES = [
    'ECG_a_R', 'ECG_RR0', 'ECG_RR1', 'ECG_RR2', 'ECG_RRm', 'ECG_RR_0_1', 'ECG_RR_2_1', 'ECG_RR_m_1', 'ECG_t_PR', 
    'ECG_t_QR', 'ECG_t_RS',	'ECG_t_RT', 'ECG_t_PQ', 'ECG_t_PS', 'ECG_t_PT', 'ECG_t_QS', 'ECG_t_QT', 'ECG_t_ST', 
    'ECG_a_PQ', 'ECG_a_QR', 'ECG_a_RS', 'ECG_a_ST', 'ECG_a_PS', 'ECG_a_PT', 'ECG_a_QS', 'ECG_a_QT', 'ECG_a_ST_QS', 
    'ECG_a_RS_QR'
]

def load_signal_features(usubjid, time_window=20, signal_type='PPG', lead=1):
    """Load signal features for a given patient."""
    # Determine file path based on signal type
    hrv_file_map = {
        'PPG': f'../data/PPG/{usubjid}_hrv_{time_window}s.csv',
        'ECG': f'../data/ECG/{usubjid}_ecg_lead{lead}_{time_window}s.csv',
    }
    hrv_file = hrv_file_map.get(signal_type)
    if not hrv_file or not os.path.exists(hrv_file):
        return None

    # Define feature names
    features_name = HRV_FEATURES.copy()
    if signal_type == 'PPG':
        features_name += PPG_ADDITIONAL_FEATURES
    elif signal_type == 'ECG':
        features_name = ECG_ADDITIONAL_FEATURES + features_name

    # Load data
    data = pd.read_csv(hrv_file)
    feature_dict = {}
    time_values = data['Time_Elapsed_Seconds'].values

    for feature in features_name:
        y = data.get(feature, pd.Series(np.nan, index=data.index)).values
        valid_mask = ~np.isnan(y) & np.isfinite(y)
        y_valid = y[valid_mask]
        feature_prefix = f'{signal_type}_{feature}' if not feature.startswith(signal_type) else feature

        feature_dict[f'{feature_prefix}_mean'] = y_valid.mean() if y_valid.size > 0 else np.nan
        feature_dict[f'{feature_prefix}_var'] = y_valid.var() if y_valid.size > 0 else np.nan

    return feature_dict

def process_sheet(excel_data, sheet_name, drop_cols=None, binary_cols=None, custom_processing=None):
    """Process a sheet from the Excel file."""
    df = pd.read_excel(excel_data, sheet_name)
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
    if binary_cols:
        df[binary_cols] = df[binary_cols].replace({1: 'Y', 0: 'N'})
    if custom_processing:
        df = custom_processing(df)
    df = df.groupby('USUBJID').first().reset_index()
    return df

def load_excel(time_window=20):
    """Load and merge data from Excel file."""
    excel_data = pd.ExcelFile(FILE_PATH)
    sheets_to_process = {
        'sepsis_DM': {'drop_cols': ['STUDY', 'SUBJID', 'ADM_DATE', 'ADM_TIME']},
        'HIST': {
            'drop_cols': [
                'STUDY', 'SITEID', 'SUBJID', 'MONITORING_START_DATE', 'MONITORING_START_TIME',
                'MONITOTING_END_DATE', 'MONITORING_END_TIME', 'PPGTYPE', 'PPGPOSI'
            ]
        },
        'ES': {
            'drop_cols': ['STUDY', 'SITEID', 'SUBJID'],
            'binary_cols': ['PNEUMONIA', 'SEPTIC SHOCK', 'UNKOWN_SOURCE']
        },
        'CLI': {'custom_processing': lambda df: df[['USUBJID', 'OXY_START_DAY']].assign(
            OXY_START_DAY=lambda x: x['OXY_START_DAY'].apply(lambda y: 'Y' if y == 0 else 'N' if pd.notna(y) else np.nan)
        )}
    }

    features = pd.DataFrame()
    for sheet_name, params in sheets_to_process.items():
        df = process_sheet(excel_data, sheet_name, **params)
        if features.empty:
            features = df
        else:
            features = features.merge(df, on='USUBJID', how='outer')

    # Process DAILY_VITAL sheet
    daily_vital = pd.read_excel(excel_data, 'DAILY_VITAL').drop(columns=['EVENT', 'DATE', 'TIME'])
    daily_vital['MONITORING_MAP'] = (daily_vital['MONITORING_SYSBP'] + 2 * daily_vital['MONITORING_DIABP']) / 3
    vital_stats = daily_vital.groupby('USUBJID').agg(['mean', 'var']).reset_index()
    vital_stats.columns = ['_'.join(filter(None, col)).strip() for col in vital_stats.columns.values]
    features = features.merge(vital_stats, on='USUBJID', how='outer')

    # Load signal features
    signal_features = []
    for usubjid in features['USUBJID']:
        combined_features = {}
        for signal_type in ['PPG', 'ECG']:
            sf = load_signal_features(usubjid, time_window, signal_type)
            if sf:
                combined_features.update(sf)
        if combined_features:
            combined_features['USUBJID'] = usubjid
            signal_features.append(combined_features)
    signal_features_df = pd.DataFrame(signal_features)
    features = features.merge(signal_features_df, on='USUBJID', how='left')

    # Load labels
    label_cli = pd.read_excel(excel_data, 'CLI')[['USUBJID', 'DIED', 'HFNOV', 'MEVEN']].astype(int)
    label_dis = pd.read_excel(excel_data, 'DIS')[['USUBJID', 'OUTCOME', 'CULTURE_RESULT']]
    labels = label_cli.merge(label_dis, on='USUBJID', how='outer').set_index('USUBJID')

    # Align features and labels
    features.set_index('USUBJID', inplace=True)
    common_indices = features.index.intersection(labels.index)
    features, labels = features.loc[common_indices], labels.loc[common_indices]

    assert len(features) == len(labels), "Mismatch between features and labels."
    return features, labels

class DataPreprocessor:
    """Preprocess the dataset for model training and evaluation."""
    def __init__(self, label_name='DIED', missing_method='median', normalize=True, feature_list=None):
        assert label_name in ['DIED', 'HFNOV', 'MEVEN', 'OUTCOME', 'CULTURE_RESULT'], "Invalid label."
        self.features, self.labels = load_excel()
        if feature_list:
            self.features = self.features[feature_list]
        valid_indices = self.labels[self.labels[label_name].notna()].index
        self.features = self.features.loc[valid_indices]
        self.labels = self.labels.loc[valid_indices, label_name]

        # One-hot encode categorical variables
        self.features = pd.get_dummies(self.features, drop_first=True)

        self.missing_method = missing_method
        self.normalize = normalize
        self.n_samples = len(self.labels)
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def split_data(self, method='kfold', n_splits=10, split_index=0):
        """Split data into training and testing sets."""
        split_methods = {
            'kfold': StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_index),
            'leaveoneout': LeaveOneOut(),
            'full': None
        }
        cv = split_methods.get(method)
        if cv:
            splits = list(cv.split(self.features, self.labels))
            train_idx, test_idx = splits[split_index % len(splits)]
            self.X_train, self.X_test = self.features.iloc[train_idx], self.features.iloc[test_idx]
            self.y_train, self.y_test = self.labels.iloc[train_idx], self.labels.iloc[test_idx]
        else:
            self.X_train, self.y_train = self.features.copy(), self.labels.copy()
            self.X_test, self.y_test = pd.DataFrame(), pd.Series()

        self.handle_missing_values()
        self.normalize_features()

    def handle_missing_values(self):
        """Handle missing values in the dataset."""
        if self.missing_method == 'median':
            imputer = SimpleImputer(strategy='median')
            self.X_train = pd.DataFrame(imputer.fit_transform(self.X_train), columns=self.X_train.columns)
            if not self.X_test.empty:
                self.X_test = pd.DataFrame(imputer.transform(self.X_test), columns=self.X_test.columns)
        elif self.missing_method:
            raise ValueError(f"Unknown missing_method: {self.missing_method}")

    def normalize_features(self):
        """Normalize the features."""
        if self.normalize:
            scaler = StandardScaler()
            self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns=self.X_train.columns)
            if not self.X_test.empty:
                self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_test.columns)

    def get_dataset(self):
        """Get the processed datasets."""
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_number_of_samples(self):
        """Get the total number of samples."""
        return self.n_samples
