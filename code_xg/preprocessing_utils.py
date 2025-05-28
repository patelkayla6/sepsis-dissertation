import neurokit2 as nk
import pandas as pd

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