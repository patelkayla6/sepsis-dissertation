import pandas as pd
import numpy as np
import neurokit2 as nk
import pyedflib
import os
from ecg_features import from_Rpeaks, from_waves
from ppg_preprocess import compute_hrv_metrics


def find_patient_ecg_path(base_path, patient):
    for folder in os.listdir(base_path):
        if patient in folder:
            path = os.path.join(base_path, folder, 'ePatch', 'Extracted (from EDF converter)')
            if os.path.exists(path):
                return path
            else:
                print(f"Patient {patient} found but no ECG data found in {path}.")
                return None
    print(f"Patient {patient} not found.")
    return None


def process_ecg_files(ecg_path, lead=0):
    if ecg_path is None:
        return None
    ecg_files = [os.path.join(ecg_path, file) for file in os.listdir(ecg_path) if file.endswith('.edf')]
    if len(ecg_files) == 0:
        print(f"No ECG files found for patient {ecg_path}.")
        return None
    # Read edf files and concatenate them if there are multiple files
    ecg_data = []
    for file in ecg_files:
        with pyedflib.EdfReader(file) as f:
            ecg_data.append(f.readSignal(lead))
    ecg_data = np.concatenate(ecg_data)
    return (ecg_data, f.getSampleFrequency(0))


def save_ecg_features(patient, time_window=20, lead=0):
    output_file = f'../data/ECG/{patient}_ecg_lead{lead}_{time_window}s.csv'
    if os.path.exists(output_file):
        print(f"ECG features for patient {patient} already exist.")
        return
    
    print(f"Extracting ECG features for patient {patient} with lead {lead}.")

    base_path = '../data/01NVa_Sepsis/RAW_DATA/'
    ecg_path = find_patient_ecg_path(base_path, patient)
    ecg_files = process_ecg_files(ecg_path, lead=lead)
    if ecg_files is None:
        return
    ecg_data, sampling_rate = ecg_files

    window_size = int(time_window * sampling_rate)
    features_list = [] 
    for i in range(0, len(ecg_data), window_size):
        try:
            window_ecg = ecg_data[i:i+window_size]
            if len(window_ecg) < window_size:
                continue

            # Clean the ECG signal
            ecg_signal = nk.ecg_clean(window_ecg, sampling_rate=sampling_rate)
            
            quality = nk.ecg_quality(window_ecg, sampling_rate=sampling_rate, method='zhao2018')
            if quality == 'Unacceptable':
                continue

            # Detect R-peaks
            _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)

            if len(rpeaks['ECG_R_Peaks']) <= 3:
                continue

            # Delineate waves
            _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=sampling_rate, method="peak")

            # Extract morphological ECG features
            time_elapsed = i / sampling_rate
            feature_time = pd.DataFrame({'Time_Elapsed_Seconds': [time_elapsed]})
            features_rpeaks = from_Rpeaks(ecg_signal, rpeaks['ECG_R_Peaks'], sampling_rate=sampling_rate, prefix='ECG', average=True)
            features_rpeaks = pd.DataFrame(features_rpeaks, index=[0])
            features_waveforms = from_waves(ecg_signal, rpeaks['ECG_R_Peaks'], waves_peak, sampling_rate=sampling_rate, prefix='ECG', average=True)
            features_waveforms = pd.DataFrame(features_waveforms, index=[0])
            features_hrv = compute_hrv_metrics(rpeaks['ECG_R_Peaks'], sampling_rate=sampling_rate)
            features = pd.concat([feature_time, features_rpeaks, features_waveforms, features_hrv], axis=1)
            features_list.append(features)
        except Exception as e:
            print(f"Error processing patient {patient}: {e} for window {i}.")
            continue
    if features_list:
        features_df = pd.concat(features_list, ignore_index=True)
        features_df.to_csv(output_file, index=False)
    else:
        print(f"No ECG features extracted for patient {patient}.")


# # Example usage
# for i in range(1001, 1260):
#     for lead in range(2):
#         try:
#             save_ecg_features(f"003-{i}", time_window=20, lead=lead)
#         except Exception as e:
#             print(f"Error processing patient {i}: {e} for lead {lead}.")

                
        
