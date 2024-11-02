import pandas as pd
import numpy as np
import neurokit2 as nk
import os
from datetime import datetime
from sqi import ppg_quality


def find_patient_ppg_path(base_path, patient):
    for folder in os.listdir(base_path):
        if patient in folder:
            return os.path.join(base_path, folder, 'PPG')
    print(f"Patient {patient} not found.")
    return None


def process_ppg_file(file_path):
    filename = os.path.basename(file_path)
    start_time_str = filename.split('.')[0]
    start_time = datetime.strptime(start_time_str, "%Y%m%dT%H%M%S")

    temp_df = pd.read_csv(file_path)
    temp_df["TIMESTAMP_MS"] += int(start_time.timestamp() * 1000)
    temp_df = temp_df[["TIMESTAMP_MS", "PLETH"]].rename(columns={"PLETH": "ppg"})
    return temp_df


def resample_ppg_data(df, sampling_interval=10):
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP_MS'], unit='ms')
    df = df.set_index('TIMESTAMP').resample(f'{sampling_interval}ms').asfreq().reset_index()
    df['TIMESTAMP_MS'] = (df['TIMESTAMP'].astype(np.int64) // 10**6)
    df = df.drop(columns='TIMESTAMP')
    return df


def remove_long_invalid_ppg(df, threshold_samples=1000):
    zero_mask = df['ppg'] == 0
    zero_groups = (zero_mask != zero_mask.shift()).cumsum()
    zero_group_sizes = zero_mask.groupby(zero_groups).transform('sum')
    df = df[~((zero_mask) & (zero_group_sizes > threshold_samples))]
    return df


def compute_hrv_metrics(peaks_dict, sampling_rate):
    hrv_metrics = pd.DataFrame()
    try:
        hrv_time = nk.hrv_time(peaks_dict, sampling_rate=sampling_rate)
        hrv_metrics = pd.concat([hrv_metrics, hrv_time], axis=1)
    except Exception as e:
        print(f"HRV time calculation error: {e}")
    try:
        hrv_freq = nk.hrv_frequency(peaks_dict, sampling_rate=sampling_rate)
        hrv_metrics = pd.concat([hrv_metrics, hrv_freq], axis=1)
    except Exception as e:
        print(f"HRV frequency calculation error: {e}")
    try:
        hrv_nonlinear = nk.hrv_nonlinear(peaks_dict, sampling_rate=sampling_rate)
        hrv_metrics = pd.concat([hrv_metrics, hrv_nonlinear], axis=1)
    except Exception as e:
        print(f"HRV nonlinear calculation error: {e}")
    return hrv_metrics


def save_ppg_features(patient, time_window=20):
    base_path = '../data/01NVa_Sepsis/RAW_DATA/'
    path = find_patient_ppg_path(base_path, patient)
    if path is None:
        return

    ppg_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    df_list = [process_ppg_file(file) for file in ppg_files]
    concatenated_df = pd.concat(df_list, ignore_index=True)
    concatenated_df = resample_ppg_data(concatenated_df)
    concatenated_df = remove_long_invalid_ppg(concatenated_df, threshold_samples=1000)

    df_ppg = concatenated_df.copy()
    df_ppg['Valid'] = df_ppg['ppg'].notna()
    df_ppg['Gap'] = df_ppg['Valid'] != df_ppg['Valid'].shift(1)
    df_ppg['Segment_ID'] = df_ppg['Gap'].cumsum()
    df_ppg = df_ppg[df_ppg['Valid']]
    start_time_ms = df_ppg["TIMESTAMP_MS"].min()

    hrv_list = []
    sampling_rate = 100
    window_size = int(time_window * sampling_rate)

    for segment_id, segment_df in df_ppg.groupby("Segment_ID"):
        segment_ppg = segment_df["ppg"].values
        segment_timestamps = segment_df["TIMESTAMP_MS"].values
        num_samples_segment = len(segment_ppg)
        num_windows = num_samples_segment // window_size

        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window_ppg = segment_ppg[start_idx:end_idx]
            window_timestamps = segment_timestamps[start_idx:end_idx]

            quality_flag, ppg_p = ppg_quality(window_ppg, sampling_rate)
            if quality_flag == 1:
                peaks_dict = nk.ppg_findpeaks(ppg_p, sampling_rate=sampling_rate)
                peaks = peaks_dict["PPG_Peaks"]

                if len(peaks) > 2:
                    hrv_metrics = compute_hrv_metrics(peaks_dict, sampling_rate)
                    pulse_amplitudes = ppg_p[peaks]
                    pav = np.std(pulse_amplitudes) / np.mean(pulse_amplitudes)
                    hrv_metrics["PAV"] = pav
                    window_start_time_ms = window_timestamps[0]
                    time_elapsed_seconds = (window_start_time_ms - start_time_ms) // 1000
                    hrv_metrics["Time_Elapsed_Seconds"] = time_elapsed_seconds
                    hrv_list.append(hrv_metrics)
            else:
                continue

    if hrv_list:
        hrv_df = pd.concat(hrv_list, ignore_index=True)
        cols = hrv_df.columns.tolist()
        cols.remove("Time_Elapsed_Seconds")
        hrv_df = hrv_df[["Time_Elapsed_Seconds"] + cols]
        output_path = '../data/HRV/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        hrv_df.to_csv(os.path.join(output_path, f"{patient}_hrv_{time_window}s.csv"), index=False)
        print(f"PPG signal processing and HRV calculation complete for {patient}.")
    else:
        print(f"No valid HRV data was found in the PPG signal for {patient}.")


# Example usage
save_ppg_features("003-1001", time_window=20)
