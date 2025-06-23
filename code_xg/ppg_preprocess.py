import pandas as pd
import numpy as np
import neurokit2 as nk
import os
from datetime import datetime
import re
import glob

from preprocessing_utils import flat_sqi
from ppg_feature import PPG_feature
from tqdm import tqdm

base_path = '/mnt/netshare1/BSP/bspprojects20/public_datasets/VitalDB/RAW_DATA/'

def find_patient_ppg_path(base_path, patient):
    # for folder in os.listdir(base_path):
    if patient in os.listdir(base_path):
        print(f"Preprocess Patient {patient}")
        return os.path.join(base_path, patient, '*', 'PPG')
    print(f"Patient {patient} not found.")
    return None


def sort_ppg_files(file_list):
    # Extract a key from each filename to sort
    def get_sort_key(file_path):
        base_name = os.path.basename(file_path)
        # Try to extract date
        try:
            start_time_str = base_name.split('.')[0]
            datetime.strptime(start_time_str, "%Y%m%dT%H%M%S")
            # Files with date in filename, sort by date
            return start_time_str
        except ValueError:
            # Files without date, extract the number at the end
            match = re.search(r'(\d+)\.csv$', base_name)
            if match:
                number = int(match.group(1))
                return "{:010d}".format(int(number))
            else:
                # No number, return the base name
                return base_name
    return sorted(file_list, key=get_sort_key)


def process_ppg_files(ppg_files):
    df_list = []
    cumulative_time_ms = 0  # Initialize cumulative time
    for file in ppg_files:
        temp_df = pd.read_csv(file)
        temp_df = temp_df[["TIMESTAMP_MS", "PLETH"]].rename(columns={"PLETH": "ppg"})
        filename = os.path.basename(file)
        try:
            # Try to parse the start time from the filename
            start_time_str = filename.split('.')[0]
            start_time = datetime.strptime(start_time_str, "%Y%m%dT%H%M%S")
            start_time_ms = int(start_time.timestamp() * 1000)
        except ValueError:
            # If parsing fails, use cumulative_time_ms
            start_time_ms = cumulative_time_ms

        # Ensure that start_time_ms is at least cumulative_time_ms to avoid overlaps
        start_time_ms = max(start_time_ms, cumulative_time_ms)
        # Adjust TIMESTAMP_MS to start from start_time_ms
        temp_df["TIMESTAMP_MS"] = start_time_ms + (temp_df["TIMESTAMP_MS"] - temp_df["TIMESTAMP_MS"].iloc[0])
        # Update cumulative_time_ms for the next file
        cumulative_time_ms = temp_df["TIMESTAMP_MS"].iloc[-1] + 10  # Add 10 ms to prevent overlap
        df_list.append(temp_df)
    concatenated_df = pd.concat(df_list, ignore_index=True)
    return concatenated_df


def resample_ppg_data(df, sampling_interval=10):
    # Drop duplicates to avoid reindexing errors
    df = df.drop_duplicates(subset='TIMESTAMP_MS')
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP_MS'], unit='ms')
    df = df.set_index('TIMESTAMP').resample(f'{sampling_interval}ms').asfreq().reset_index()
    df['TIMESTAMP_MS'] = (df['TIMESTAMP'].astype(np.int64) // 10**6)
    df = df.drop(columns='TIMESTAMP')
    return df


def save_ppg_features(patient, time_window=20):
    if os.path.exists(f'../data/PPG/{patient}_{time_window}s.csv'):
        return
    path = find_patient_ppg_path(base_path, patient)
    if path is None:
        return
    ppg_files = glob.glob(os.path.join(path, '*csv'))
    if len(ppg_files) == 0:
        return
    ppg_files = sort_ppg_files(ppg_files)
    concatenated_df = process_ppg_files(ppg_files)
    concatenated_df = resample_ppg_data(concatenated_df)

    df_ppg = concatenated_df.copy()
    df_ppg['Valid'] = df_ppg['ppg'].notna()
    df_ppg['Gap'] = df_ppg['Valid'] != df_ppg['Valid'].shift(1)
    df_ppg['Segment_ID'] = df_ppg['Gap'].cumsum()
    df_ppg = df_ppg[df_ppg['Valid']]
    start_time_ms = df_ppg["TIMESTAMP_MS"].min()

    feature_list = []
    sampling_rate = 1000 // 10  # Since sampling interval is 10ms
    window_size = int(time_window * sampling_rate)

    segment_number = df_ppg['Segment_ID'].max()
    print(f'Continuous Segment Number {segment_number}')
    for segment_id, segment_df in df_ppg.groupby("Segment_ID"):
        segment_ppg = segment_df["ppg"].values
        segment_timestamps = segment_df["TIMESTAMP_MS"].values
        num_samples_segment = len(segment_ppg)
        num_windows = num_samples_segment // window_size

        for i in tqdm(range(num_windows), desc=f"  Segment {segment_id}", leave=False):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window_ppg = segment_ppg[start_idx:end_idx]

            window_timestamps = segment_timestamps[start_idx:end_idx]
            window_start_time_ms = window_timestamps[0]
            time_elapsed_seconds = (window_start_time_ms - start_time_ms) // 1000

            if flat_sqi(window_ppg):
                continue

            ppg_feature = PPG_feature(window_ppg, fs=sampling_rate)
            ppg_feature["Time_Elapsed_Seconds"] = time_elapsed_seconds

            feature_list.append(ppg_feature)

    if feature_list:
        feature_df = pd.concat(feature_list, ignore_index=True)

        cols = feature_df.columns.tolist()
        cols.remove("Time_Elapsed_Seconds")
        feature_df = feature_df[["Time_Elapsed_Seconds"] + cols]

        output_path = '../data/PPG/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        feature_df.to_csv(os.path.join(output_path, f"{patient}_{time_window}s.csv"), index=False)
        print(f"PPG signal processing and HRV calculation complete for {patient}.")
    else:
        print(f"No valid HRV data was found in the PPG signal for {patient}.")


# # Example usage
if __name__ == '__main__':
    for i in range(1001, 1260): #45 is error
        save_ppg_features(f"01NVa-003-{i}", time_window=20)