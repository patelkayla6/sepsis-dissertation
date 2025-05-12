# The code is provided by Dr. Xiao Gu and then modified.

import pandas as pd
import numpy as np
import neurokit2 as nk
import heartpy as hp
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import skew

# Define necessary functions from the SQI code
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

def skewness_sqi(sig, sampling_rate, window=2, step=1, return_s=False):
    step_size = step * sampling_rate
    range_sample = window * sampling_rate
    if len(sig) < range_sample:
        if return_s:
            return 1, []  # Bad signal
        else:
            return 1  # Bad signal
    start = 0
    s = []
    while start < len(sig) - range_sample:
        end = start + range_sample
        seg = sig[start:end]
        start += step_size
        s.append(skew(seg))
    counter = [a > 0 for a in s]  # Positive skewness check
    if sum(counter) > 0.5 * len(s):  # More than 50% positive skewness
        if return_s:
            return 0, s  # Good signal
        else:
            return 0
    else:
        if return_s:
            return 1, s  # Bad signal
        else:
            return 1

def median_absolute_deviation(x):
    return np.median(np.abs(x - np.median(x)))

def hampel(ts, window_size=5, n=3, imputation=False):
    if not isinstance(ts, pd.Series):
        ts = pd.Series(ts)

    ts_cleaned = ts.copy()
    k = 1.4826  # Scaling factor for Gaussian distribution

    rolling_ts = ts_cleaned.rolling(window_size * 2 + 1, center=True)
    rolling_median = rolling_ts.median().bfill().ffill()
    mad = rolling_ts.apply(median_absolute_deviation).bfill().ffill()
    rolling_sigma = k * mad

    difference = np.abs(ts_cleaned - rolling_median)
    outlier_idx = difference > (n * rolling_sigma)
    outlier_indices = np.where(outlier_idx)[0]

    if imputation:
        ts_cleaned[outlier_indices] = rolling_median[outlier_indices]
        return outlier_indices, ts_cleaned.to_numpy()

    return outlier_indices

def ppg_quality(ppg, fs):
    if flat_sqi(ppg) == 1:
        return 0, ppg

    if ppg.std() == 0:
        return 0, ppg  # Bad signal due to lack of variability

    ppg = (ppg - ppg.mean()) / ppg.std()
    ppg_p = nk.ppg_clean(ppg, sampling_rate=fs, method='elgendi')

    try:
        wd, m = hp.process(ppg_p, fs, windowsize=1, reject_segmentwise=True)
    except hp.exceptions.BadSignalWarning:
        return 0, ppg

    if sum(wd['binary_peaklist']) < 5:
        return 0, ppg

    if skewness_sqi(ppg_p, fs) == 0:
        return 1, ppg_p  # Good signal
    else:
        return 0, ppg  # Bad signal