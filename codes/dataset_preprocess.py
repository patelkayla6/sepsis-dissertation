import os
import functools
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Path to the Excel file
file_path = '../data/sepsis_01NVa_Sep2024_CHENYANG.xlsx'

def _aggregate_stats(X, y):
    """Collapse a per-patient time series (X=elapsed time, y=values) into summary
    stats. Used to turn windowed HRV/vital-sign readings into per-patient features."""
    if len(y) == 0:
        return {'mean': np.nan, 'var': np.nan, 'std': np.nan,
                'median': np.nan, 'mad': np.nan, 'slope': np.nan}
    median = np.median(y)
    stats = {
        'mean': y.mean(),
        'var': y.var(),
        'std': y.std(),
        'median': median,
        'mad': np.median(np.abs(y - median)),
        'slope': np.polyfit(X.flatten(), y, 1)[0] if len(y) > 1 else np.nan,
    }
    return stats

def load_signal_features(usubjid, time_window=20, signal_type='PPG', lead=1, time_cutoff_hours=None):
    if signal_type == 'PPG':
        hrv_file = f'../data/PPG/01NVa-{usubjid}_{time_window}s.csv'
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
    if signal_type == 'ECG':
        features_name = ['ECG_a_R', 'ECG_RR0', 'ECG_RR1', 'ECG_RR2', 'ECG_RRm', 'ECG_RR_0_1', 'ECG_RR_2_1', 'ECG_RR_m_1', 'ECG_t_PR', 
                        'ECG_t_QR', 'ECG_t_RS',	'ECG_t_RT', 'ECG_t_PQ', 'ECG_t_PS', 'ECG_t_PT', 'ECG_t_QS', 'ECG_t_QT', 'ECG_t_ST', 
                        'ECG_a_PQ', 'ECG_a_QR', 'ECG_a_RS', 'ECG_a_ST', 'ECG_a_PS', 'ECG_a_PT', 'ECG_a_QS', 'ECG_a_QT', 'ECG_a_ST_QS', 
                        'ECG_a_RS_QR'] + features_name
    data = pd.read_csv(hrv_file)
    if time_cutoff_hours is not None:
        data = data[data['Time_Elapsed_Seconds'] <= time_cutoff_hours * 3600]
    feature_dict = {}
    for feature in features_name:
        X = data['Time_Elapsed_Seconds'].values.reshape(-1, 1)
        if feature in data.columns:
            # coerce first: some HRV columns contain Excel formula-error strings
            # (e.g. '#NAME?') instead of numbers, which np.isnan() can't handle
            y = pd.to_numeric(data[feature], errors='coerce').values
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
        for stat_name, value in _aggregate_stats(X, y).items():
            feature_dict[f'{feature}_{stat_name}'] = value
    return feature_dict

@functools.lru_cache(maxsize=2)
# Bounded to 2 (not unbounded, not 1): within a single cutoff's classifier/
# feature-set grid, caching still avoids redundant re-reads; maxsize=2 also
# covers DataPreprocessor's train+test dual-key pattern (a constant train-side
# cutoff plus a varying test-side cutoff) without those two evicting each
# other on every call. Unbounded caching previously retained every distinct
# cutoff's full feature table for the rest of the process, causing memory to
# grow across a multi-cutoff sweep until a large CSV read failed with an
# out-of-memory error.
def load_excel(time_window=20, time_cutoff_hours=None):
    # Load the Excel file
    excel_data = pd.ExcelFile(file_path)
    
    # Function to process each sheet
    def process_sheet(sheet_name, drop_cols=None, binary_cols=None, custom_processing=None, aggregate_func=None):
        df = pd.read_excel(excel_data, sheet_name)
        if drop_cols:
            df = df.drop(columns=drop_cols)
        if binary_cols:
            df[binary_cols] = df[binary_cols].map(lambda x: 'Y' if x == 1 else 'N')
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
    # FIX (previously a future-data leak): DATE/TIME used to be dropped immediately,
    # so MONITORING_* vitals were aggregated over the ENTIRE monitoring session
    # regardless of `time_cutoff_hours` -- meaning the Clinical feature set silently
    # used vitals recorded after the cutoff applied to PPG/ECG. We now reconstruct a
    # real per-reading timestamp and truncate it the same way PPG/ECG already is.
    DAILY_VITAL = pd.read_excel(excel_data, 'DAILY_VITAL')
    DAILY_VITAL['Vital_Timestamp'] = pd.to_datetime(
        DAILY_VITAL['DATE'].dt.strftime('%Y-%m-%d') + ' ' + DAILY_VITAL['TIME'].dt.strftime('%H:%M:%S')
    )
    DAILY_VITAL = DAILY_VITAL.drop(columns=['EVENT', 'DATE', 'TIME'])

    # Per-patient monitoring start timestamp, used only to compute elapsed time for
    # truncation below -- not merged into `features` as a model input.
    monitoring_start = pd.read_excel(excel_data, 'HIST')[
        ['USUBJID', 'MONITORING_START_DATE', 'MONITORING_START_TIME']
    ].groupby('USUBJID').first().reset_index()
    monitoring_start['Monitoring_Start_Timestamp'] = pd.to_datetime(
        monitoring_start['MONITORING_START_DATE'].dt.strftime('%Y-%m-%d') + ' ' +
        monitoring_start['MONITORING_START_TIME'].dt.strftime('%H:%M:%S')
    )
    DAILY_VITAL = DAILY_VITAL.merge(
        monitoring_start[['USUBJID', 'Monitoring_Start_Timestamp']], on='USUBJID', how='left'
    )
    DAILY_VITAL['Elapsed_Hours'] = (
        DAILY_VITAL['Vital_Timestamp'] - DAILY_VITAL['Monitoring_Start_Timestamp']
    ).dt.total_seconds() / 3600
    if time_cutoff_hours is not None:
        # FIX: truncate vitals to the same observation window as PPG/ECG, so
        # `Clinical`-containing feature sets stay consistent with the cutoff
        # applied elsewhere instead of leaking later vitals from the same session.
        DAILY_VITAL = DAILY_VITAL[DAILY_VITAL['Elapsed_Hours'] <= time_cutoff_hours]

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
            X = valid_data['VITAL_VITAL_SEQ'].values.reshape(-1, 1)
            y = valid_data[var].values
            for stat_name, value in _aggregate_stats(X, y).items():
                result[f'{var}_{stat_name}'] = value
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
        hrv_features = load_signal_features(usubjid, time_window=time_window, signal_type='PPG',
                                             time_cutoff_hours=time_cutoff_hours)
        if hrv_features is not None:
            hrv_features['USUBJID'] = usubjid
            signal_features.append(hrv_features)
    signal_features_df = pd.DataFrame(signal_features)
    if not signal_features_df.empty:
        features = features.merge(signal_features_df, on='USUBJID', how='left')

    # Add ECG features for each patient
    signal_features = []
    for usubjid in features['USUBJID']:
        hrv_features = load_signal_features(usubjid, time_window=time_window, signal_type='ECG', lead=1,
                                             time_cutoff_hours=time_cutoff_hours)
        if hrv_features is not None:
            hrv_features['USUBJID'] = usubjid
            signal_features.append(hrv_features)
    signal_features_df = pd.DataFrame(signal_features)
    if not signal_features_df.empty:
        features = features.merge(signal_features_df, on='USUBJID', how='left')

    # Add NEWS score and SOFA score for each patient
    news_csv = pd.read_csv('../data/SOFANEWS SCORES_CHENYANG_27.csv')
    news_csv = news_csv[['USUBJID', 'NEWS_ADM', 'NEWS_ENR', 'SOFA_ADM', 'SOFA_enr']]
    features = features.merge(news_csv, on='USUBJID', how='left')

    # Save the names of the features as a file
    feature_names = features.columns
    if not os.path.exists('../data/feature_names.txt'):
        with open('../data/feature_names.txt', 'w') as f:
            for name in feature_names:
                f.write(f"'{name}', ")

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
    def __init__(self, label_name='DIED', missing_method='median', normalize=True, feature_list=None,
                 time_cutoff_hours=None, test_time_cutoff_hours=None):
        assert label_name in ['DIED', 'HFNOV', 'MEVEN', 'OUTCOME', 'CULTURE_RESULT']
        assert missing_method in [None, 'median']

        features, labels = load_excel(time_window=20, time_cutoff_hours=time_cutoff_hours)

        if feature_list is not None:
            # only keep the features in the feature_list
            features = features[feature_list]

        # Select IDs with non-missing labels and available features
        IDs_with_labels = labels[labels[label_name].notna()].index
        ID_list = IDs_with_labels.intersection(features.index)

        # Remove patients died within 24 hours if label_name is 'DIED'
        if label_name == 'DIED':
            remove_ids = ['003-1011', '003-1016', '003-1017', '003-1214', '003-1219', '003-1239']
            ID_list = ID_list.difference(remove_ids)

        self.features = features.loc[ID_list]
        self.labels = labels.loc[ID_list, label_name]

        # One-hot encode categorical variables
        categorical_cols = self.features.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            self.features = pd.get_dummies(
                self.features, columns=categorical_cols, dummy_na=False, drop_first=True
            )

        # Optionally source test-fold rows from a separately-truncated feature table,
        # so the model can be trained on one time window (e.g. full recordings) and
        # evaluated on a different, shorter window (e.g. the first 2 hours) for the
        # same patients.
        self.test_source_features = None
        if test_time_cutoff_hours is not None:
            test_features, _ = load_excel(time_window=20, time_cutoff_hours=test_time_cutoff_hours)
            if feature_list is not None:
                test_features = test_features[feature_list]
            test_features = test_features.loc[ID_list]
            test_categorical_cols = test_features.select_dtypes(include=['object']).columns
            if len(test_categorical_cols) > 0:
                test_features = pd.get_dummies(
                    test_features, columns=test_categorical_cols, dummy_na=False, drop_first=True
                )
            # Align columns with the train-side feature table in case one-hot encoding
            # produced a different set of dummy columns between the two cutoffs.
            self.test_source_features = test_features.reindex(columns=self.features.columns, fill_value=0)

        # Initialize data splitting variables
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

        # Store missing_method and normalize for later use
        self.missing_method, self.normalize = missing_method, normalize

        # Store the number of samples
        self.n_samples = len(self.labels)

    def split_data(self, method='kfold', n_splits=10, split_index=0):
        assert method in ['kfold', 'leaveoneout', 'full']
        test_source = self.test_source_features if self.test_source_features is not None else self.features
        if method == 'kfold':
            assert n_splits >= 2
            cv = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=split_index // n_splits
            )
            splits = list(cv.split(self.features, self.labels))
            train_index, test_index = splits[split_index % n_splits]
            self.X_train, self.X_test = self.features.iloc[train_index], test_source.iloc[test_index]
            self.y_train, self.y_test = self.labels.iloc[train_index], self.labels.iloc[test_index]
        elif method == 'leaveoneout':
            cv = LeaveOneOut()
            splits = list(cv.split(self.features, self.labels))
            n_splits = len(splits)
            assert split_index < n_splits, "split_index out of range for leave-one-out cross-validation"
            train_index, test_index = splits[split_index]
            self.X_train, self.X_test = self.features.iloc[train_index], test_source.iloc[test_index]
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
            self._deal_with_missing(method=self.missing_method)
        if self.normalize:
            self._normalize_features()

    def _deal_with_missing(self, method='median'):
        if method == 'median':
            # Fill numerical columns with the median of the training set
            numerical_cols = self.X_train.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                num_imputer = SimpleImputer(strategy='median', keep_empty_features=True)
                self.X_train.loc[:, numerical_cols] = num_imputer.fit_transform(self.X_train[numerical_cols])
                if not self.X_test.empty:
                    self.X_test.loc[:, numerical_cols] = num_imputer.transform(self.X_test[numerical_cols])

            # Fill categorical columns with the mode of the training set
            categorical_cols = self.X_train.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent', keep_empty_features=True)
                self.X_train.loc[:, categorical_cols] = cat_imputer.fit_transform(self.X_train[categorical_cols])
                if not self.X_test.empty:
                    self.X_test.loc[:, categorical_cols] = cat_imputer.transform(self.X_test[categorical_cols])
        else:
            raise ValueError('Invalid method to deal with missing values')

    def _normalize_features(self):
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
