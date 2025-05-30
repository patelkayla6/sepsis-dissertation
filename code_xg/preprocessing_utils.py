import neurokit2 as nk
import pandas as pd
import numpy as np

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

def flat_sqi(signal, window=60, e=1e-5):
    if len(signal) < window:
        return 1  # Bad signal due to insufficient length
    for i in range(len(signal) - window):
        seg = signal[i:i + window]
        d = np.diff(seg)
        counter = sum([abs(a) < e for a in d])
        if counter == window - 1:
            return 1  # Bad signal
    return 0  # Good signal