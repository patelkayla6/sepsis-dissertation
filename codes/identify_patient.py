"""
Utility to identify which patient ID an unknown raw PPG file belongs to.

Usage:
    python identify_patient.py <path_to_raw_ppg_csv>

Steps:
    1. Parses the recording timestamp from the filename
    2. Finds candidate patients whose monitoring window contains that timestamp (HIST sheet)
    3. If multiple candidates, runs per-column HRV correlation against existing PPG_old CSVs
       to find the best match
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

EXCEL_PATH = '../data/sepsis_01NVa_Sep2024_CHENYANG.xlsx'
PPG_OLD_PATH = '../data/PPG_old'
TIMEZONE_OFFSET_HOURS = 0  # set to 1 for BST (+0100), 0 for UTC


def parse_file_timestamp(raw_file_path):
    filename = os.path.basename(raw_file_path).split('.')[0]
    try:
        file_time = datetime.strptime(filename, "%Y%m%dT%H%M%S")
    except ValueError:
        raise ValueError(f"Cannot parse timestamp from filename: {filename}. Expected format: YYYYMMDDTHHmmSS")
    # Shift to UTC using known timezone offset
    file_time_utc = file_time - timedelta(hours=TIMEZONE_OFFSET_HOURS)
    return file_time_utc


def load_hist():
    xl = pd.ExcelFile(EXCEL_PATH)
    df = pd.read_excel(xl, 'HIST')
    df['start_dt'] = pd.to_datetime(df['MONITORING_START_DATE'].dt.date.astype(str) + ' ' +
                                    df['MONITORING_START_TIME'].dt.time.astype(str),
                                    errors='coerce')
    df['end_dt'] = pd.to_datetime(df['MONITOTING_END_DATE'].dt.date.astype(str) + ' ' +
                                  df['MONITORING_END_TIME'].dt.time.astype(str),
                                  errors='coerce')
    return df


def find_candidates_by_window(file_time, hist_df):
    candidates = []
    for _, row in hist_df.iterrows():
        start = row['start_dt']
        end = row['end_dt']
        if pd.isna(start):
            continue
        # If end is unknown, check only that file_time >= start (within 24h buffer)
        if pd.isna(end):
            if start <= file_time <= start + timedelta(hours=24):
                candidates.append(row['USUBJID'])
        else:
            if start <= file_time <= end:
                candidates.append(row['USUBJID'])
    return candidates


def extract_hrv_features_from_raw(raw_file_path, time_window=20):
    """Run pipeline 2 on the raw file and return the resulting DataFrame."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code_xg', 'preprocess'))
    from ppg_preprocess import process_ppg_files, resample_ppg_data
    from ppg_feature import PPG_feature
    from preprocessing_utils import flat_sqi
    from tqdm import tqdm

    ppg_files = [raw_file_path]
    concatenated_df = process_ppg_files(ppg_files)
    concatenated_df = resample_ppg_data(concatenated_df)

    df_ppg = concatenated_df.copy()
    df_ppg['Valid'] = df_ppg['ppg'].notna()
    df_ppg['Gap'] = df_ppg['Valid'] != df_ppg['Valid'].shift(1)
    df_ppg['Segment_ID'] = df_ppg['Gap'].cumsum()
    df_ppg = df_ppg[df_ppg['Valid']]
    start_time_ms = df_ppg['TIMESTAMP_MS'].min()

    sampling_rate = 100
    window_size = time_window * sampling_rate
    feature_list = []

    for segment_id, segment_df in df_ppg.groupby('Segment_ID'):
        segment_ppg = segment_df['ppg'].values
        segment_timestamps = segment_df['TIMESTAMP_MS'].values
        num_windows = len(segment_ppg) // window_size

        for i in tqdm(range(num_windows), desc=f'Segment {segment_id}', leave=False):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window_ppg = segment_ppg[start_idx:end_idx]
            if flat_sqi(window_ppg):
                continue
            window_start_ms = segment_timestamps[start_idx]
            time_elapsed = (window_start_ms - start_time_ms) // 1000
            feat = PPG_feature(window_ppg, fs=sampling_rate)
            feat['Time_Elapsed_Seconds'] = time_elapsed
            feature_list.append(feat)

    if not feature_list:
        return None
    return pd.concat(feature_list, ignore_index=True)


def run_hrv_correlation(new_df, candidate_id):
    old_path = os.path.join(PPG_OLD_PATH, f'{candidate_id}_hrv_20s.csv')
    if not os.path.exists(old_path):
        return None, 0

    old_df = pd.read_csv(old_path)
    hrv_cols = [c for c in new_df.columns if c.startswith('HRV_') and c in old_df.columns]

    merged = new_df[['Time_Elapsed_Seconds'] + hrv_cols].merge(
        old_df[['Time_Elapsed_Seconds'] + hrv_cols],
        on='Time_Elapsed_Seconds', suffixes=('_new', '_old')
    )

    correlations = []
    for col in hrv_cols:
        x = merged[f'{col}_new']
        y = merged[f'{col}_old']
        valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if valid.sum() > 10:
            r = np.corrcoef(x[valid], y[valid])[0, 1]
            if np.isfinite(r):
                correlations.append(r)

    mean_corr = np.mean(correlations) if correlations else np.nan
    overlap = len(merged)
    return mean_corr, overlap


def identify_patient(raw_file_path):
    print(f"\nRaw file: {raw_file_path}")

    file_time = parse_file_timestamp(raw_file_path)
    print(f"Parsed timestamp (UTC): {file_time}")

    hist_df = load_hist()
    candidates = find_candidates_by_window(file_time, hist_df)

    if not candidates:
        print("\nNo candidates found in HIST sheet for this timestamp.")
        return

    print(f"\nCandidates from HIST monitoring windows: {candidates}")

    if len(candidates) == 1:
        print(f"\nUnique match: {candidates[0]}")
        return candidates[0]

    print(f"\nMultiple candidates — running HRV correlation tiebreaker...")
    print("Extracting features from raw file (this may take a few minutes)...")
    new_df = extract_hrv_features_from_raw(raw_file_path)
    if new_df is None:
        print("Could not extract features from raw file.")
        return

    results = []
    for candidate in candidates:
        mean_corr, overlap = run_hrv_correlation(new_df, candidate)
        print(f"  {candidate}: mean_corr={mean_corr:.4f}, overlapping_windows={overlap}")
        results.append((candidate, mean_corr, overlap))

    results.sort(key=lambda x: x[1] if x[1] is not None else -999, reverse=True)
    best = results[0]
    print(f"\nBest match: {best[0]} (mean_corr={best[1]:.4f})")
    return best[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Identify patient ID for an unknown raw PPG file.')
    parser.add_argument('raw_file', help='Path to the raw PPG CSV file')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    identify_patient(args.raw_file)
