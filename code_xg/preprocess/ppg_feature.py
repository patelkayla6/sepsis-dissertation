from pyPPG import PPG, Fiducials, Biomarkers
from pyPPG.datahandling import load_data, plot_fiducials, save_data
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM
import pyPPG.ppg_sqi as SQI

import numpy as np
import sys
import json
import pandas as pd
from dotmap import DotMap
import scipy

import neurokit2 as nk

from preprocessing_utils import compute_hrv_metrics
import contextlib
import os
import sys

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def PPG_feature(sig, fs):
    signal = DotMap()

    signal.start_sig = 0
    signal.end_sig = len(sig)

    signal.fs=fs
    signal.v = sig

    signal.filtering = True  # whether or not to filter the PPG signal
    signal.fL = 0.5000001  # Lower cutoff frequency (Hz)
    signal.fH = 12  # Upper cutoff frequency (Hz)
    signal.order = 4  # Filter order
    signal.sm_wins = {'ppg': 50, 'vpg': 10, 'apg': 10,
                      'jpg': 10}  # smoothing windows in millisecond for the PPG_old, PPG_old', PPG_old", and PPG_old'"

    prep = PP.Preprocess(fL=signal.fL, fH=signal.fH, order=signal.order, sm_wins=signal.sm_wins)
    signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)

    # Initialise the correction for fiducial points
    corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
    correction = pd.DataFrame()
    correction.loc[0, corr_on] = True
    signal.correction = correction

    # Create a PPG class
    s = PPG(signal)

    with suppress_stdout():
        fpex = FP.FpCollection(s=s)
        fiducials = fpex.get_fiducials(s=s)
        # Create a fiducials class
        fp = Fiducials(fp=fiducials)

        bmex = BM.BmCollection(s=s, fp=fp)

        # Extract biomarkers
        bm_defs, bm_vals, bm_stats = bmex.get_biomarkers()
        ppgSQI = np.mean(SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fp.sp))

    # Assume `fp.sp` is your list of peak sample indices and `s.fs` is your sampling rate
    rpeaks_dict = {"PPG_Peaks": np.array(fp.sp, dtype=int)}

    # Compute HRV features
    hrv_df = compute_hrv_metrics(rpeaks_dict, sampling_rate=s.fs)
    # print(hrv_df.values)
    # Combine features
    biomarker_vals = np.array(bm_stats['ppg_sig'])[0]
    biomarker_names = list(bm_stats['ppg_sig'].columns)

    feature_vals = np.concatenate((biomarker_vals, [ppgSQI], hrv_df.values.squeeze()))
    feature_names = biomarker_names + ['SQI'] + list(hrv_df.columns)

    return pd.DataFrame([feature_vals], columns=feature_names)

if __name__ == '__main__':
    ppg = np.squeeze(scipy.io.loadmat('../../test/Sample_PPG_MAT_125Hz.mat').get('Data'))[0:125 * 20]
    feat = PPG_feature(ppg, 125)
    print(feat)

    '''
    feature name
    '''








