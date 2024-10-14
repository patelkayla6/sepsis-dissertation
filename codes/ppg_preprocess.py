import pandas as pd
import numpy as np
import neurokit2 as nk
import os
from datetime import datetime

def save_ppg_features(patient, time_window=120, sqi_threshold=0.5):

    # Load CSV data
    # patient = '01Nva-003-1007'
    path = f'../data/01NVa_Sepsis/RAW_DATA/{patient}/PPG/'

    # List all CSV files in the path
    ppg_files = [f for f in os.listdir(path) if f.endswith('.csv')]

    # Initialize an empty list to store dataframes
    df_list = []

    # Loop through each file, read and append to the list
    for file in ppg_files:
        # Extract the start time from the filename
        start_time_str = file.split('.')[0]
        start_time = datetime.strptime(start_time_str, "%Y%m%dT%H%M%S")

        # Read the CSV file
        temp_df = pd.read_csv(os.path.join(path, file))

        # Calculate the absolute timestamp based on the starting time of the file
        temp_df["ABSOLUTE_TIMESTAMP_MS"] = temp_df["TIMESTAMP_MS"] + int(start_time.timestamp() * 1000)

        # Extract necessary columns
        temp_df = temp_df[["ABSOLUTE_TIMESTAMP_MS", "PLETH"]]

        # Rename columns for consistency
        temp_df.rename(columns={"ABSOLUTE_TIMESTAMP_MS": "TIMESTAMP_MS", "PLETH": "ppg"}, inplace=True)

        # Set TIMESTAMP_MS as the index
        temp_df.set_index("TIMESTAMP_MS", inplace=True)

        # Resample the data to ensure 10 ms intervals
        # Fill in any missing values using forward fill method
        timestamp_range = pd.RangeIndex(start=temp_df.index.min(), stop=temp_df.index.max() + 10, step=10)
        temp_df = temp_df.reindex(timestamp_range, method='ffill')
        temp_df.index.name = "TIMESTAMP_MS"

        # Reset the index to have TIMESTAMP_MS as a column
        temp_df.reset_index(inplace=True)

        # Append to the list of dataframes
        df_list.append(temp_df)

    # Concatenate all dataframes
    concatenated_df = pd.concat(df_list, ignore_index=True)

    # Remove invalid PPG signals that last for more than 10 seconds (1000 samples at 100 Hz)
    invalid_zero_groups = (concatenated_df['ppg'] == 0).astype(int).groupby((concatenated_df['ppg'] != 0).cumsum()).cumsum()
    concatenated_df = concatenated_df[~((concatenated_df['ppg'] == 0) & (invalid_zero_groups > 1000))]

    # Extract PPG signal
    df_ppg = concatenated_df.copy()
    ppg_signal = df_ppg["ppg"].values

    # Clean the PPG signal before computing quality
    ppg_cleaned = nk.ppg_clean(ppg_signal, sampling_rate=100)

    # Compute PPG Quality Index on the cleaned signal
    ppg_quality = nk.ppg_quality(ppg_cleaned, sampling_rate=100)

    # Add cleaned PPG and Quality Index to df_ppg
    df_ppg["PPG_Clean"] = ppg_cleaned
    df_ppg["PPG_Quality"] = ppg_quality

    # Only keep rows with Quality Index greater than 0.5 and update timestamps accordingly
    df_ppg_filtered = df_ppg[df_ppg["PPG_Quality"] > sqi_threshold].reset_index(drop=True)

    # Compute time differences to identify gaps
    df_ppg_filtered["Time_Diff"] = df_ppg_filtered["TIMESTAMP_MS"].diff().fillna(10)

    # Identify gaps (assuming sampling rate is 100 Hz, i.e., 10 ms between samples)
    df_ppg_filtered["Gap"] = df_ppg_filtered["Time_Diff"] > 10

    # Assign segment IDs to continuous high-quality data
    df_ppg_filtered["Segment_ID"] = df_ppg_filtered["Gap"].cumsum()

    # Calculate the start time for elapsed time calculation
    start_time_ms = df_ppg_filtered["TIMESTAMP_MS"].iloc[0]

    # Initialize list to store HRV metrics
    hrv_list = []

    # Change window size from 10 seconds to 2 minutes (120 seconds)
    window_size = time_window * 100  # 2 minutes window with 100 Hz sampling rate

    # Loop through each continuous segment
    for segment_id, segment_df in df_ppg_filtered.groupby("Segment_ID"):
        segment_length = len(segment_df)
        num_windows = segment_length // window_size
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window_df = segment_df.iloc[start_idx:end_idx]
            window_signal = window_df["PPG_Clean"].values
            if len(window_signal) == window_size:
                # Process the window to extract HRV metrics
                # Find peaks in the PPG signal
                peaks_dict = nk.ppg_findpeaks(window_signal, sampling_rate=100)
                peaks = peaks_dict["PPG_Peaks"]

                # Ensure there are enough peaks to compute HRV
                if len(peaks) > 2:
                    # Compute HRV metrics
                    hrv_metrics = nk.hrv(peaks_dict, sampling_rate=100, show=False)
                    
                    # Add additional PPG features
                    # Example: Pulse Amplitude Variability (PAV)
                    pulse_amplitudes = window_signal[peaks]
                    pav = np.std(pulse_amplitudes) / np.mean(pulse_amplitudes)
                    hrv_metrics["PAV"] = pav

                    # Calculate elapsed time in seconds from the start time
                    time_elapsed_seconds = (window_df.iloc[0]["TIMESTAMP_MS"] - start_time_ms) / 1000.0
                    hrv_metrics["Time_Elapsed_Seconds"] = time_elapsed_seconds

                    # Collect the hrv_metrics in a list
                    hrv_list.append(hrv_metrics)

    # Concatenate HRV metrics into a DataFrame
    if hrv_list:
        hrv_df = pd.concat(hrv_list, ignore_index=True)
        # Reorder columns for readability
        cols = hrv_df.columns.tolist()
        cols.remove("Time_Elapsed_Seconds")
        hrv_df = hrv_df[["Time_Elapsed_Seconds"] + cols]
        # Save the HRV variables
        hrv_df.to_csv(path+f"hrv_{time_window}_{sqi_threshold}.csv", index=False)
        print(f"PPG signal processing and HRV calculation complete for {patient}.")
    else:
        print(f"No valid HRV data was found in the PPG signal for {patient}.")
