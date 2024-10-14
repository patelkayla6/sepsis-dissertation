import os
import pandas as pd
import numpy as np 
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from ppg_preprocess import save_ppg_features

# Path to the Excel file
file_path = '../data/sepsis_01NVa_Sep2024_CHENYANG.xlsx'

def load_excel():
    # Load the Excel file
    excel_data = pd.ExcelFile(file_path)

    # Load and process 'sepsis_DM' sheet
    sepsis_DM = pd.read_excel(excel_data, 'sepsis_DM')
    feature_sepsis_DM = sepsis_DM.drop(columns=['STUDY', 'SUBJID', 'ADM_DATE', 'ADM_TIME'])

    # Load and process 'HIST' sheet
    HIST = pd.read_excel(excel_data, 'HIST')
    hist_drop_cols = [
        'STUDY', 'SITEID', 'SUBJID', 'MONITORING_START_DATE', 'MONITORING_START_TIME',
        'MONITOTING_END_DATE', 'MONITORING_END_TIME', 'PPGTYPE', 'PPGPOSI'
    ]
    feature_HIST = HIST.drop(columns=hist_drop_cols)

    # Load and process 'ES' sheet
    ES = pd.read_excel(excel_data, 'ES')
    feature_ES = ES.drop(columns=['STUDY', 'SITEID', 'SUBJID'])
    binary_columns = ['PNEUMONIA', 'SEPTIC SHOCK', 'UNKOWN_SOURCE']
    feature_ES[binary_columns] = feature_ES[binary_columns].applymap(lambda x: 'Y' if x == 1 else 'N')

    # Load and process 'CLI' sheet for features
    CLI = pd.read_excel(excel_data, 'CLI')
    feature_CLI = CLI[['USUBJID', 'OXY_START_DAY']]
    feature_CLI['OXY_START_DAY'] = feature_CLI['OXY_START_DAY'].apply(
        lambda x: 'Y' if x == 0 else ('N' if pd.notna(x) else np.nan)
    )

    # Load and process 'DAILY_VITAL' sheet
    DAILY_VITAL = pd.read_excel(excel_data, 'DAILY_VITAL')
    df_DAILY_VITAL = DAILY_VITAL.drop(columns=['EVENT', 'DATE', 'TIME'])
    # Add 'MONITORING_MAP' column, which is calculated by (SYSBP + 2 * DIABP) / 3
    df_DAILY_VITAL['MONITORING_MAP'] = (df_DAILY_VITAL['MONITORING_SYSBP'] + 2 * df_DAILY_VITAL['MONITORING_DIABP']) / 3
    variables = [
        'MONITORING_TEMP', 'MONITORING_PULSE', 'MONITORING_SYSBP',
        'MONITORING_DIABP', 'MONITORING_RESP', 'MONITORING_SPO2', 'MONITORING_MAP'
    ]
    

    # Calculate statistics for 'DAILY_VITAL'
    feature_DAILY_VITAL = []
    for usubjid, group in df_DAILY_VITAL.groupby('USUBJID'):
        result = {'USUBJID': usubjid}
        for var in variables:
            valid_data = group[['VITAL_VITAL_SEQ', var]].dropna()
            X = valid_data['VITAL_VITAL_SEQ'].values.reshape(-1, 1)
            y = valid_data[var].values
            if len(y) > 1:
                result[f'{var}_mean'] = y.mean()
                result[f'{var}_var'] = y.var()
                # Fit linear regression
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                result[f'{var}_slope'] = model.coef_[0]
                result[f'{var}_intercept'] = model.intercept_
                result[f'{var}_rmse'] = np.sqrt(mean_squared_error(y, y_pred))
                result[f'{var}_r2'] = r2_score(y, y_pred)
            else:
                result[f'{var}_mean'] = y.mean() if len(y) > 0 else np.nan
                result[f'{var}_var'] = y.var() if len(y) > 0 else np.nan
                result[f'{var}_slope'] = np.nan
                result[f'{var}_intercept'] = np.nan
                result[f'{var}_rmse'] = np.nan
                result[f'{var}_r2'] = np.nan
        feature_DAILY_VITAL.append(result)
    feature_DAILY_VITAL = pd.DataFrame(feature_DAILY_VITAL)

    # Merge all features
    features = feature_sepsis_DM.merge(feature_HIST, on='USUBJID', how='outer')
    features = features.merge(feature_ES, on='USUBJID', how='outer')
    features = features.merge(feature_CLI, on='USUBJID', how='outer')
    features = features.merge(feature_DAILY_VITAL, on='USUBJID', how='outer')

    # Process ppg data for each patient
    raw_path = f'../data/01NVa_Sepsis/RAW_DATA/'
    patient_names = [f for f in os.listdir(raw_path)]
    for usubjid in features['USUBJID']:
        for patient in patient_names:
            if usubjid in patient:
                save_ppg_features(patient)

    # Load and process labels
    label_CLI = CLI[['USUBJID', 'DIED', 'HFNOV', 'MEVEN']]
    label_columns = ['DIED', 'HFNOV', 'MEVEN']
    for col in label_columns:
        label_CLI[col] = label_CLI[col].apply(lambda x: 1 if x is True else (0 if x is False else np.nan))

    DIS = pd.read_excel(excel_data, 'DIS')
    # The values for 'CULTURE_RESULT' are 'POS', 'NEG', or 'ND'
    label_DIS = DIS[['USUBJID', 'OUTCOME', 'CULTURE_RESULT']]
    labels = label_CLI.merge(label_DIS, on='USUBJID', how='outer')

    return features.set_index('USUBJID'), labels.set_index('USUBJID')


class DataPreprocessor:
    def __init__(self, label_name='DIED', missing_method='median', normalize=True):
        assert label_name in ['DIED', 'HFNOV', 'MEVEN', 'OUTCOME', 'CULTURE_RESULT']
        assert missing_method in [None, 'median']

        features, labels = load_excel()

        feature_list = features.columns.tolist()
        # Select IDs with non-missing labels and available features
        IDs_with_labels = labels[labels[label_name].notna()].index
        ID_list = IDs_with_labels.intersection(features.index)

        self.features = features.loc[ID_list, feature_list]
        self.labels = labels.loc[ID_list, label_name]

        # One-hot encode categorical variables
        categorical_cols = self.features.select_dtypes(include=['object']).columns
        self.features = pd.get_dummies(self.features, columns=categorical_cols, dummy_na=False)

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
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_index//n_splits)
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
                self.num_imputer = SimpleImputer(strategy='median')
                self.X_train[numerical_cols] = self.num_imputer.fit_transform(self.X_train[numerical_cols])
                if not self.X_test.empty:
                    self.X_test[numerical_cols] = self.num_imputer.transform(self.X_test[numerical_cols])

            # Fill categorical columns with the mode of the training set
            categorical_cols = self.X_train.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                self.cat_imputer = SimpleImputer(strategy='most_frequent')
                self.X_train[categorical_cols] = self.cat_imputer.fit_transform(self.X_train[categorical_cols])
                if not self.X_test.empty:
                    self.X_test[categorical_cols] = self.cat_imputer.transform(self.X_test[categorical_cols])
        else:
            raise ValueError('Invalid method to deal with missing values')


    def normalize_features(self):
        numerical_cols = self.X_train.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            self.scaler = StandardScaler()
            self.scaler.fit(self.X_train[numerical_cols])

            self.X_train[numerical_cols] = self.scaler.transform(self.X_train[numerical_cols])
            if not self.X_test.empty:
                self.X_test[numerical_cols] = self.scaler.transform(self.X_test[numerical_cols])


    def get_dataset(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_number_of_samples(self):
        """Returns the total number of samples available."""
        return self.n_samples
