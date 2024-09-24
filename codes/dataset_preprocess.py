# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# # Path to the Excel file
# file_path = '../data/sepsis_01NVa_Sep2024_CHENYANG.xlsx'


# # Load the Excel file
# def load_excel():
#     excel_data = pd.ExcelFile(file_path)

#     sepsis_DM = pd.read_excel(excel_data, 'sepsis_DM')
#     # remove the collumns: STUDY, SUBJID, ADM_DATE, ADM_TIME
#     feature_sepsis_DM = sepsis_DM.drop(columns=['STUDY', 'SUBJID', 'ADM_DATE', 'ADM_TIME'])
#     feature_sepsis_DM.head()

#     HIST = pd.read_excel(excel_data, 'HIST')
#     # remove the collumns: STUDY, SITEID, SUBJID, MONITORING_START_DATE, MONITORING_START_TIME, MONITORING_END_DATE, MONITORING_END_TIME
#     feature_HIST = HIST.drop(columns=['STUDY', 'SITEID', 'SUBJID', 'MONITORING_START_DATE', 'MONITORING_START_TIME', 'MONITOTING_END_DATE', 'MONITORING_END_TIME'])
#     # currently also remove PPGTYPE, PPGPOS because they may not be useful
#     feature_HIST = feature_HIST.drop(columns=['PPGTYPE', 'PPGPOSI'])

#     ES = pd.read_excel(excel_data, 'ES')
#     # remove the collumns: STUDY, SITEID, SUBJID
#     feature_ES = ES.drop(columns=['STUDY', 'SITEID', 'SUBJID'])
#     # For PNEUMONIA, SEPTIC SHOCK and UNKOWN_SOURCE, replace 1 with 'Y' and nan with 'N'
#     feature_ES['PNEUMONIA'] = feature_ES['PNEUMONIA'].apply(lambda x: 'Y' if x == 1 else 'N')
#     feature_ES['SEPTIC SHOCK'] = feature_ES['SEPTIC SHOCK'].apply(lambda x: 'Y' if x == 1 else 'N')
#     feature_ES['UNKOWN_SOURCE'] = feature_ES['UNKOWN_SOURCE'].apply(lambda x: 'Y' if x == 1 else 'N')

#     CLI = pd.read_excel(excel_data, 'CLI')
#     # Only remain the following columns: USUBJID, OXY_START_DAY
#     feature_CLI = CLI[['USUBJID', 'OXY_START_DAY']]
#     # For OXY_START_DAY, replace 0 with 'Y' and other values with 'N' (excluding nan)
#     feature_CLI['OXY_START_DAY'] = feature_CLI['OXY_START_DAY'].apply(lambda x: 'Y' if x == 0 else ('N' if pd.notna(x) else np.nan))
    
#     # for DAILY_VITA, calculate the mean, variance, slope, intercept, rmse, r2_score
#     DAILY_VITAL = pd.read_excel(excel_data, 'DAILY_VITAL')
#     # remove the collumns: EVENT, DATE, TIME
#     df_DAILY_VITAL = DAILY_VITAL.drop(columns=['EVENT', 'DATE', 'TIME'])

#     variables = [
#         'MONITORING_TEMP', 'MONITORING_PULSE', 'MONITORING_SYSBP',
#         'MONITORING_DIABP', 'MONITORING_RESP', 'MONITORING_SPO2'
#     ]

#     feature_DAILY_VITAL = []
#     # Group by 'USUBJID'
#     for usubjid, group in df_DAILY_VITAL.groupby('USUBJID'):
#         result = {'USUBJID': usubjid}
#         for var in variables:
#             # Drop missing values
#             valid_data = group[['VITAL_VITAL_SEQ', var]].dropna()
#             X = valid_data['VITAL_VITAL_SEQ'].values.reshape(-1, 1)
#             y = valid_data[var].values
#             if len(y) > 1:  # Need at least two data points
#                 # Compute mean and variance
#                 mean = y.mean()
#                 variance = y.var()
#                 result[f'{var}_mean'] = mean
#                 result[f'{var}_var'] = variance

#                 # Fit linear regression
#                 model = LinearRegression()
#                 model.fit(X, y)
#                 slope = model.coef_[0]
#                 intercept = model.intercept_
#                 y_pred = model.predict(X)
#                 rmse = np.sqrt(mean_squared_error(y, y_pred))
#                 r2 = r2_score(y, y_pred)
#                 result[f'{var}_slope'] = slope
#                 result[f'{var}_intercept'] = intercept
#                 result[f'{var}_rmse'] = rmse
#                 result[f'{var}_r2'] = r2
#             else:
#                 # Not enough data to fit regression
#                 result[f'{var}_mean'] = y.mean() if len(y) > 0 else np.nan
#                 result[f'{var}_var'] = y.var() if len(y) > 0 else np.nan
#                 result[f'{var}_slope'] = np.nan
#                 result[f'{var}_intercept'] = np.nan
#                 result[f'{var}_rmse'] = np.nan
#                 result[f'{var}_r2'] = np.nan
#         feature_DAILY_VITAL.append(result)

#     # Convert the list of results into a DataFrame
#     feature_DAILY_VITAL = pd.DataFrame(feature_DAILY_VITAL)

#     # Merge all the features
#     features = feature_sepsis_DM.merge(feature_HIST, on='USUBJID', how='outer')
#     features = features.merge(feature_ES, on='USUBJID', how='outer')
#     features = features.merge(feature_CLI, on='USUBJID', how='outer')
#     features = features.merge(feature_DAILY_VITAL, on='USUBJID', how='outer')

#     # Load the labels
#     CLI = pd.read_excel(excel_data, 'CLI')
#     label_CLI = CLI[['USUBJID', 'DIED', 'HFNOV', 'MEVEN']]
#     label_CLI['DIED'] = label_CLI['DIED'].apply(lambda x: 1 if x == True else (0 if x == False else np.nan))
#     label_CLI['HFNOV'] = label_CLI['HFNOV'].apply(lambda x: 1 if x == True else (0 if x == False else np.nan))
#     label_CLI['MEVEN'] = label_CLI['MEVEN'].apply(lambda x: 1 if x == True else (0 if x == False else np.nan))
#     DIS = pd.read_excel(excel_data, 'DIS')
#     label_DIS = DIS[['USUBJID', 'OUTCOME', 'CULTURE_RESULT']]
#     labels = label_CLI.merge(label_DIS, on='USUBJID', how='outer')

#     return features.set_index('USUBJID'), labels.set_index('USUBJID')


# class DataPreprocessor:
#     def __init__(self, missing_method='median', encode=True, normalize=True):
#         self.features, self.labels = load_excel()
#         if missing_method is not None:
#             self.deal_with_missing(method=missing_method)
#         if encode:
#             self.encode_features()
#         if normalize:
#             self.normalize_features()
        
#     def deal_with_missing(self, method='median'):
#         if method == 'median':

#             # Fill numerical columns with the median
#             numerical_cols = self.features.select_dtypes(include=[np.number]).columns
#             for col in numerical_cols:
#                 self.features[col].fillna(self.features[col].median(), inplace=True)

#             # Fill categorical columns with the mode
#             categorical_cols = self.features.select_dtypes(include=['object']).columns
#             for col in categorical_cols:
#                 self.features[col].fillna(self.features[col].mode()[0], inplace=True)

#         else:
#             raise ValueError('Invalid method to deal with missing values')
        
                
#     def encode_features(self):
#         # One-hot encode categorical variables
#         self.features = pd.get_dummies(self.features, dummy_na=False)


#     def normalize_features(self):
#         numerical_cols = self.features.select_dtypes(include=[np.number]).columns
#         for col in numerical_cols:
#             mean = self.features[col].mean()
#             std = self.features[col].std()
#             if std != 0:
#                 self.features[col] = (self.features[col] - mean) / std
#             else:
#                 self.features[col] = 0

                
#     def prepare_dataset(self, label_name='DIED', feature_list=None, ID_list=None):
#         if feature_list is None:
#             feature_list = self.features.columns.tolist()

#         if ID_list is None:
#             # select ids with label_name not nan
#             ID_list = self.labels[label_name].dropna().index
#             # select ids with both features and labels
#             ID_list = list(set(ID_list) & set(self.features.index))

#         X = self.features[feature_list].loc[ID_list]
#         y = self.labels[label_name].loc[ID_list]
#         return X, y

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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
    variables = [
        'MONITORING_TEMP', 'MONITORING_PULSE', 'MONITORING_SYSBP',
        'MONITORING_DIABP', 'MONITORING_RESP', 'MONITORING_SPO2'
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

    # Load and process labels
    label_CLI = CLI[['USUBJID', 'DIED', 'HFNOV', 'MEVEN']]
    label_columns = ['DIED', 'HFNOV', 'MEVEN']
    for col in label_columns:
        label_CLI[col] = label_CLI[col].apply(lambda x: 1 if x is True else (0 if x is False else np.nan))

    DIS = pd.read_excel(excel_data, 'DIS')
    label_DIS = DIS[['USUBJID', 'OUTCOME', 'CULTURE_RESULT']]
    labels = label_CLI.merge(label_DIS, on='USUBJID', how='outer')

    return features.set_index('USUBJID'), labels.set_index('USUBJID')

class DataPreprocessor:
    def __init__(self, missing_method='median', encode=True, normalize=True):
        self.features, self.labels = load_excel()
        if missing_method:
            self.deal_with_missing(method=missing_method)
        if encode:
            self.encode_features()
        if normalize:
            self.normalize_features()

    def deal_with_missing(self, method='median'):
        if method == 'median':
            # Fill numerical columns with the median
            numerical_cols = self.features.select_dtypes(include=[np.number]).columns
            num_imputer = SimpleImputer(strategy='median')
            self.features[numerical_cols] = num_imputer.fit_transform(self.features[numerical_cols])

            # Fill categorical columns with the mode
            categorical_cols = self.features.select_dtypes(include=['object']).columns
            cat_imputer = SimpleImputer(strategy='most_frequent')
            self.features[categorical_cols] = cat_imputer.fit_transform(self.features[categorical_cols])
        else:
            raise ValueError('Invalid method to deal with missing values')

    def encode_features(self):
        # One-hot encode categorical variables
        categorical_cols = self.features.select_dtypes(include=['object']).columns
        self.features = pd.get_dummies(self.features, columns=categorical_cols, dummy_na=False)

    def normalize_features(self):
        numerical_cols = self.features.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        self.features[numerical_cols] = scaler.fit_transform(self.features[numerical_cols])

    def prepare_dataset(self, label_name='DIED', feature_list=None, ID_list=None):
        if feature_list is None:
            feature_list = self.features.columns.tolist()
        if ID_list is None:
            # Select IDs with non-missing labels and available features
            IDs_with_labels = self.labels[self.labels[label_name].notna()].index
            ID_list = IDs_with_labels.intersection(self.features.index)
        X = self.features.loc[ID_list, feature_list]
        y = self.labels.loc[ID_list, label_name]
        return X, y
