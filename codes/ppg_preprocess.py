import pandas as pd
import numpy as np
import neurokit2 as nk
import os
from datetime import datetime
from sqi import ppg_quality


def save_ppg_features(patient, time_window=20):
    # patient example: "003-1001"
    # Load CSV data
    path = f'../data/01NVa_Sepsis/RAW_DATA/'
    # find the folder name containing the patient name
    for folder in os.listdir(path):
        if patient in folder:
            path = os.path.join(path, folder)
            break
    if path == f'../data/01NVa_Sepsis/RAW_DATA/':
        print(f"Patient {patient} not found.")
        return
    path = os.path.join(path, 'PPG')

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

    # Get the starting timestamp of the entire signal
    start_time_ms = df_ppg["TIMESTAMP_MS"].min()

    # Compute time differences to identify gaps
    df_ppg["Time_Diff"] = df_ppg["TIMESTAMP_MS"].diff().fillna(10)
    # Identify gaps (assuming sampling rate is 100 Hz, i.e., 10 ms between samples)
    df_ppg["Gap"] = df_ppg["Time_Diff"] > 10
    # Assign segment IDs to continuous data
    df_ppg["Segment_ID"] = df_ppg["Gap"].cumsum()

    # Initialize list to store HRV metrics
    hrv_list = []

    sampling_rate = 100  # Hz
    window_size = int(time_window * sampling_rate)  # Number of samples in 20 seconds

    # Loop over each continuous segment
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
            if len(window_ppg) == window_size:
                # Use ppg_quality function to assess the window
                quality_flag, ppg_p = ppg_quality(window_ppg, sampling_rate)
                if quality_flag == 1:
                    # Valid window
                    # Find peaks in the PPG signal
                    peaks_dict = nk.ppg_findpeaks(ppg_p, sampling_rate=sampling_rate)
                    peaks = peaks_dict["PPG_Peaks"]

                    # Ensure there are enough peaks to compute HRV
                    if len(peaks) > 2:
                        # Initialize hrv_metrics as empty DataFrame
                        hrv_metrics = pd.DataFrame()

                        # Try to compute time-domain HRV metrics
                        try:
                            hrv_time = nk.hrv_time(peaks_dict, sampling_rate=sampling_rate)
                            hrv_metrics = pd.concat([hrv_metrics, hrv_time], axis=1)
                        except Exception as e:
                            print(f"HRV time calculation error: {e}")

                        # Try to compute frequency-domain HRV metrics
                        try:
                            hrv_freq = nk.hrv_frequency(peaks_dict, sampling_rate=sampling_rate)
                            hrv_metrics = pd.concat([hrv_metrics, hrv_freq], axis=1)
                        except Exception as e:
                            print(f"HRV frequency calculation error: {e}")

                        # Optionally, try to compute nonlinear HRV metrics
                        try:
                            hrv_nonlinear = nk.hrv_nonlinear(peaks_dict, sampling_rate=sampling_rate)
                            hrv_metrics = pd.concat([hrv_metrics, hrv_nonlinear], axis=1)
                        except Exception as e:
                            print(f"HRV nonlinear calculation error: {e}")

                        # Add additional PPG features if needed
                        # Example: Pulse Amplitude Variability (PAV)
                        pulse_amplitudes = ppg_p[peaks]
                        pav = np.std(pulse_amplitudes) / np.mean(pulse_amplitudes)
                        hrv_metrics["PAV"] = pav

                        # Get the starting timestamp of the window
                        window_start_time_ms = window_timestamps[0]

                        # Compute time elapsed in seconds from the start of the signal
                        time_elapsed_seconds = (window_start_time_ms - start_time_ms) / 1000.0

                        # Add the time_elapsed_seconds to hrv_metrics
                        hrv_metrics["Time_Elapsed_Seconds"] = time_elapsed_seconds

                        # Check if hrv_metrics is not empty
                        if not hrv_metrics.empty:
                            # Append to hrv_list
                            hrv_list.append(hrv_metrics)
                        else:
                            print("No HRV metrics were computed for this window.")
                    else:
                        # Not enough peaks, skip this window
                        continue
                else:
                    # Invalid window, skip
                    continue

    # Concatenate HRV metrics into a DataFrame
    if hrv_list:
        hrv_df = pd.concat(hrv_list, ignore_index=True)
        # Reorder columns for readability
        cols = hrv_df.columns.tolist()
        cols.remove("Time_Elapsed_Seconds")
        hrv_df = hrv_df[["Time_Elapsed_Seconds"] + cols]
        # Save the HRV variables
        output_path = '../data/HRV/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        hrv_df.to_csv(os.path.join(output_path, f"{patient}_hrv_{time_window}s.csv"), index=False)
        print(f"PPG signal processing and HRV calculation complete for {patient}.")
    else:
        print(f"No valid HRV data was found in the PPG signal for {patient}.")


# Example usage
save_ppg_features("003-1001", time_window=20)
