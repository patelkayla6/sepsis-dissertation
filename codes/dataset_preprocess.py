import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Path to the Excel file
file_path = '../data/sepsis_01NVa_Sep2024_CHENYANG.xlsx'

def load_signal_features(usubjid, time_window=20, signal_type='PPG', lead=1):
    if signal_type == 'PPG':
        hrv_file = f'../data/PPG/{usubjid}_hrv_{time_window}s.csv'
    elif signal_type == 'ECG':
        hrv_file = f'../data/ECG/{usubjid}_ecg_lead{lead}_{time_window}s.csv'
    if not os.path.exists(hrv_file):
        return None
    features_name = [
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
    if signal_type == 'PPG':
        features_name = features_name + ['PAV']
    elif signal_type == 'ECG':
        features_name = ['ECG_a_R', 'ECG_RR0', 'ECG_RR1', 'ECG_RR2', 'ECG_RRm', 'ECG_RR_0_1', 'ECG_RR_2_1', 'ECG_RR_m_1', 'ECG_t_PR', 
                        'ECG_t_QR', 'ECG_t_RS',	'ECG_t_RT', 'ECG_t_PQ', 'ECG_t_PS', 'ECG_t_PT', 'ECG_t_QS', 'ECG_t_QT', 'ECG_t_ST', 
                        'ECG_a_PQ', 'ECG_a_QR', 'ECG_a_RS', 'ECG_a_ST', 'ECG_a_PS', 'ECG_a_PT', 'ECG_a_QS', 'ECG_a_QT', 'ECG_a_ST_QS', 
                        'ECG_a_RS_QR'] + features_name
    data = pd.read_csv(hrv_file)
    feature_dict = {}
    for feature in features_name:
        X = data['Time_Elapsed_Seconds'].values.reshape(-1, 1)
        if feature in data.columns:
            y = data[feature].values
        else:
            y = np.full_like(X, np.nan)
        # remove NaN values
        valid_idx = ~np.isnan(y)
        X, y = X[valid_idx], y[valid_idx]
        # remove infinite values
        valid_idx = np.isfinite(y)
        X, y = X[valid_idx], y[valid_idx]
        if signal_type == 'PPG':
            feature = f'PPG_{feature}'
        elif signal_type == 'ECG' and feature[:3] != 'ECG':
            feature = f'ECG_{feature}'
        feature_dict[f'{feature}_mean'] = y.mean()
        feature_dict[f'{feature}_var'] = y.var()
    return feature_dict

def load_excel(time_window=20):
    # Load the Excel file
    excel_data = pd.ExcelFile(file_path)
    
    # Function to process each sheet
    def process_sheet(sheet_name, drop_cols=None, binary_cols=None, custom_processing=None, aggregate_func=None):
        df = pd.read_excel(excel_data, sheet_name)
        if drop_cols:
            df = df.drop(columns=drop_cols)
        if binary_cols:
            df[binary_cols] = df[binary_cols].applymap(lambda x: 'Y' if x == 1 else 'N')
        if custom_processing:
            df = custom_processing(df)
        if aggregate_func:
            df = df.groupby('USUBJID').aggregate(aggregate_func).reset_index()
        return df

    # Process sheets
    feature_sepsis_DM = process_sheet(
        'sepsis_DM', drop_cols=['STUDY', 'SUBJID', 'ADM_DATE', 'ADM_TIME'],
        aggregate_func='first'
    )
    hist_drop_cols = [
        'STUDY', 'SITEID', 'SUBJID', 'MONITORING_START_DATE', 'MONITORING_START_TIME',
        'MONITOTING_END_DATE', 'MONITORING_END_TIME', 'PPGTYPE', 'PPGPOSI'
    ]
    feature_HIST = process_sheet(
        'HIST', drop_cols=hist_drop_cols,
        aggregate_func='first'
    )
    feature_ES = process_sheet(
        'ES', drop_cols=['STUDY', 'SITEID', 'SUBJID'],
        binary_cols=['PNEUMONIA', 'SEPTIC SHOCK', 'UNKOWN_SOURCE'],
        aggregate_func='first'
    )

    # Custom processing for 'CLI' sheet
    def process_CLI(df):
        df['OXY_START_DAY'] = df['OXY_START_DAY'].apply(
            lambda x: 'Y' if x == 0 else ('N' if pd.notna(x) else np.nan)
        )
        return df[['USUBJID', 'OXY_START_DAY']]
    feature_CLI = process_sheet('CLI', custom_processing=process_CLI, aggregate_func='first')

    # Process 'DAILY_VITAL' sheet
    DAILY_VITAL = pd.read_excel(excel_data, 'DAILY_VITAL').drop(columns=['EVENT', 'DATE', 'TIME'])
    DAILY_VITAL['MONITORING_MAP'] = (
        DAILY_VITAL['MONITORING_SYSBP'] + 2 * DAILY_VITAL['MONITORING_DIABP']
    ) / 3
    variables = [
        'MONITORING_TEMP', 'MONITORING_PULSE', 'MONITORING_SYSBP',
        'MONITORING_DIABP', 'MONITORING_RESP', 'MONITORING_SPO2', 'MONITORING_MAP'
    ]

    # Calculate statistics for 'DAILY_VITAL'
    feature_DAILY_VITAL = []
    for usubjid, group in DAILY_VITAL.groupby('USUBJID'):
        result = {'USUBJID': usubjid}
        for var in variables:
            valid_data = group[['VITAL_VITAL_SEQ', var]].dropna()
            if valid_data.empty:
                for stat in ['mean', 'var']:
                    result[f'{var}_{stat}'] = np.nan
                continue
            X = valid_data['VITAL_VITAL_SEQ'].values.reshape(-1, 1)
            y = valid_data[var].values
            result[f'{var}_mean'] = y.mean()
            result[f'{var}_var'] = y.var()
        feature_DAILY_VITAL.append(result)
    feature_DAILY_VITAL = pd.DataFrame(feature_DAILY_VITAL)

    # Merge all features
    features = feature_sepsis_DM.merge(feature_HIST, on='USUBJID', how='outer')
    features = features.merge(feature_ES, on='USUBJID', how='outer')
    features = features.merge(feature_CLI, on='USUBJID', how='outer')
    features = features.merge(feature_DAILY_VITAL, on='USUBJID', how='outer')

    # Add PPG features for each patient
    signal_features = []
    for usubjid in features['USUBJID']:
        hrv_features = load_signal_features(usubjid, time_window=time_window, signal_type='PPG')
        if hrv_features is not None:
            hrv_features['USUBJID'] = usubjid
            signal_features.append(hrv_features)
    signal_features_df = pd.DataFrame(signal_features)
    features = features.merge(signal_features_df, on='USUBJID', how='left')

    # Add ECG features for each patient
    signal_features = []
    for usubjid in features['USUBJID']:
        hrv_features = load_signal_features(usubjid, time_window=time_window, signal_type='ECG', lead=1)
        if hrv_features is not None:
            hrv_features['USUBJID'] = usubjid
            signal_features.append(hrv_features)    
    signal_features_df = pd.DataFrame(signal_features)
    features = features.merge(signal_features_df, on='USUBJID', how='left')

    # Load and process labels
    label_CLI = pd.read_excel(excel_data, 'CLI')
    label_columns = ['DIED', 'HFNOV', 'MEVEN']
    label_CLI = label_CLI[['USUBJID'] + label_columns]
    for col in label_columns:
        label_CLI[col] = label_CLI[col].map({True: 1, False: 0})
    label_CLI = label_CLI.groupby('USUBJID').aggregate('first').reset_index()

    label_DIS = pd.read_excel(excel_data, 'DIS')[['USUBJID', 'OUTCOME', 'CULTURE_RESULT']]
    label_DIS = label_DIS.groupby('USUBJID').aggregate('first').reset_index()
    labels = label_CLI.merge(label_DIS, on='USUBJID', how='outer')

    # Set index after ensuring uniqueness
    features = features.set_index('USUBJID')
    labels = labels.set_index('USUBJID')

    # Align features and labels
    common_usubjid = features.index.intersection(labels.index)
    features = features.loc[common_usubjid]
    labels = labels.loc[common_usubjid]

    assert len(features) == len(labels), "Features and labels DataFrames have different lengths."

    return features, labels


class DataPreprocessor:
    def __init__(self, label_name='DIED', missing_method='median', normalize=True, feature_list=None):
        assert label_name in ['DIED', 'HFNOV', 'MEVEN', 'OUTCOME', 'CULTURE_RESULT']
        assert missing_method in [None, 'median']

        features, labels = load_excel(time_window=20)

        if feature_list is not None:
            # only keep the features in the feature_list
            features = features[feature_list]

        # Select IDs with non-missing labels and available features
        IDs_with_labels = labels[labels[label_name].notna()].index
        ID_list = IDs_with_labels.intersection(features.index)

        self.features = features.loc[ID_list]
        self.labels = labels.loc[ID_list, label_name]

        # One-hot encode categorical variables
        categorical_cols = self.features.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            self.features = pd.get_dummies(
                self.features, columns=categorical_cols, dummy_na=False, drop_first=True
            )

        # Initialize data splitting variables
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

        # Store missing_method and normalize for later use
        self.missing_method, self.normalize = missing_method, normalize

        # Store the number of samples
        self.n_samples = len(self.labels)

    def split_data(self, method='kfold', n_splits=10, split_index=0):
        assert method in ['kfold', 'leaveoneout', 'full']
        if method == 'kfold':
            assert n_splits >= 2
            cv = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=split_index // n_splits
            )
            splits = list(cv.split(self.features, self.labels))
            train_index, test_index = splits[split_index % n_splits]
            self.X_train, self.X_test = self.features.iloc[train_index], self.features.iloc[test_index]
            self.y_train, self.y_test = self.labels.iloc[train_index], self.labels.iloc[test_index]
        elif method == 'leaveoneout':
            cv = LeaveOneOut()
            splits = list(cv.split(self.features, self.labels))
            n_splits = len(splits)
            assert split_index < n_splits, "split_index out of range for leave-one-out cross-validation"
            train_index, test_index = splits[split_index]
            self.X_train, self.X_test = self.features.iloc[train_index], self.features.iloc[test_index]
            self.y_train, self.y_test = self.labels.iloc[train_index], self.labels.iloc[test_index]
        elif method == 'full':
            # Use the full dataset as the training set and create empty test sets
            self.X_train = self.features.copy()
            self.y_train = self.labels.copy()
            self.X_test = pd.DataFrame(columns=self.features.columns)
            self.y_test = pd.Series(dtype=self.labels.dtype)
        else:
            raise ValueError('Invalid method for data splitting')

        if self.missing_method:
            self.deal_with_missing(method=self.missing_method)
        if self.normalize:
            self.normalize_features()

    def deal_with_missing(self, method='median'):
        if method == 'median':
            # Fill numerical columns with the median of the training set
            numerical_cols = self.X_train.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                num_imputer = SimpleImputer(strategy='median')
                self.X_train[numerical_cols] = num_imputer.fit_transform(self.X_train[numerical_cols])
                if not self.X_test.empty:
                    self.X_test[numerical_cols] = num_imputer.transform(self.X_test[numerical_cols])

            # Fill categorical columns with the mode of the training set
            categorical_cols = self.X_train.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                self.X_train[categorical_cols] = cat_imputer.fit_transform(self.X_train[categorical_cols])
                if not self.X_test.empty:
                    self.X_test[categorical_cols] = cat_imputer.transform(self.X_test[categorical_cols])
        else:
            raise ValueError('Invalid method to deal with missing values')

    def normalize_features(self):
        numerical_cols = self.X_train.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            scaler = StandardScaler()
            self.X_train[numerical_cols] = scaler.fit_transform(self.X_train[numerical_cols])
            if not self.X_test.empty:
                self.X_test[numerical_cols] = scaler.transform(self.X_test[numerical_cols])

    def get_dataset(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_number_of_samples(self):
        """Returns the total number of samples available."""
        return self.n_samples
