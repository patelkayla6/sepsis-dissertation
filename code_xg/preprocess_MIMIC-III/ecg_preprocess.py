#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from neurokit2.hrv.hrv_nonlinear import NeuroKitWarning

warnings.filterwarnings("ignore", category=NeuroKitWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from preprocessing_utils import flat_sqi
from ecg_feature  import extract_feature as ECG_feature

# ─── CONFIG ─────────────────────────────────────────────────────────────
WFDB_PATH     = '/media/engs2588/DATA/Data/physionet.org/files/mimic3wdb-matched/1.0'
KEEPERS_CSV   = 'mimic_overlap_keepers.csv'
SEPSIS3_CSV   = '/mnt/Data/engs2588/thirdparty/sepsis3-mimic/data/sepsis3-df.csv'

# Typical ECG leads in MIMIC-III waveform DB (tweak if needed)
ECG_CHANNELS  = [
    'II',
]
#     'II', 'I', 'III', 'MLII',
#     'V', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
#     'AVR', 'AVL', 'AVF'
# 125 Hz → 1000 ms / 125 = 8 ms grid
SAMPLING_MS   = 8      # grid interval in ms
TIME_WINDOW   = 20     # window length in s
OUT_DIR       = 'ecg_features_by_icu_v2'

os.makedirs(OUT_DIR, exist_ok=True)

ecg_feature_names = [
    # R-peaks:
    "ecg_a_R", "ecg_RR0", "ecg_RR1", "ecg_RR2",
    "ecg_RRm", "ecg_RR_0_1", "ecg_RR_2_1", "ecg_RR_m_1",

    # wave features:
    "ecg_t_PR", "ecg_t_QR", "ecg_t_RS", "ecg_t_RT",
    "ecg_t_PQ", "ecg_t_PS", "ecg_t_PT",
    "ecg_t_QS", "ecg_t_QT", "ecg_t_ST",
    "ecg_a_PQ", "ecg_a_QR", "ecg_a_RS", "ecg_a_ST",
    "ecg_a_PS", "ecg_a_PT", "ecg_a_QS", "ecg_a_QT",
    "ecg_a_ST_QS", "ecg_a_RS_QR",

    # HRV (NK2 time-domain)
    "ecg_hrv_MeanNN", "ecg_hrv_SDNN", "ecg_hrv_SDSD", "ecg_hrv_RMSSD",
    "ecg_hrv_CVNN", "ecg_hrv_CVSD", "ecg_hrv_MedianNN", "ecg_hrv_MadNN",
    "ecg_hrv_MCVNN", "ecg_hrv_IQRNN",
    "ecg_hrv_SDANN1", "ecg_hrv_SDANN2", "ecg_hrv_SDANN5", "ecg_hrv_SDANN10",
    "ecg_hrv_SDNN1", "ecg_hrv_SDNN2", "ecg_hrv_SDNN5", "ecg_hrv_SDNN10",
    "ecg_hrv_pNN50", "ecg_hrv_pNN20",
    "ecg_hrv_TINN", "ecg_hrv_HTI",

    # SQI
    "ecg_sqi_zhao2018"
]

# ─── LOAD KEEPERS & GROUP BY ICU ────────────────────────────────────────
ov_df = pd.read_csv(KEEPERS_CSV)
# ensure rec_start and rec_end are datetimes
ov_df['rec_start'] = pd.to_datetime(ov_df['rec_start'], format='%Y-%m-%d %H:%M:%S.%f')
ov_df['rec_end']   = pd.to_datetime(ov_df['rec_end'],   format='%Y-%m-%d %H:%M:%S.%f')
icu_groups = ov_df.groupby('icustay_id')['record_rel'].apply(list).to_dict()

# ─── LOAD SEPSIS3 DF & BUILD ICU-LEVEL ZERO POINTS ──────────────────────
sepsis3_df = pd.read_csv(SEPSIS3_CSV)
# ensure intime is datetime
sepsis3_df['intime'] = pd.to_datetime(sepsis3_df['intime'], format='%Y-%m-%d %H:%M:%S')

# 1) earliest ICU admission per icustay_id
icu_start_series = (
    sepsis3_df
      .groupby('icustay_id')['intime']
      .min()
)
# 2) convert to ms since epoch
icu_start_ms = (icu_start_series.astype('int64') // 10**6).to_dict()

# ─── BUILD RECORD-LEVEL START MAP ───────────────────────────────────────
rec_start_series = ov_df.set_index('record_rel')['rec_start']
rec_start_ms_map = (rec_start_series.astype('int64') // 10**6).to_dict()


# ─── EXTRACT & RESAMPLE ONE RECORD (ECG) ────────────────────────────────
def load_and_resample_ecg(record_rel, start_ms):
    """
    Read one WFDB record, pick an ECG channel, and resample onto a fixed
    SAMPLING_MS grid (8 ms → 125 Hz) using linear interpolation.
    """
    rec_path = os.path.join(WFDB_PATH, record_rel)
    rec = wfdb.rdrecord(rec_path)

    # choose first available ECG channel
    available_idxs = []
    for i, name in enumerate(rec.sig_name):
        nm = str(name).upper()
        if nm in ECG_CHANNELS or nm.startswith('ECG'):
            available_idxs.append(i)

    if not available_idxs:
        raise ValueError(f"No ECG channel in {record_rel}")

    idx = available_idxs[0]
    sig = rec.p_signal[:, idx]
    fs  = rec.fs

    # --- construct original absolute timestamps ---
    orig_time_ms = start_ms + np.arange(len(sig)) * (1000.0 / fs)

    # --- construct new regular timestamps (8 ms grid) ---
    new_time_ms = np.arange(
        orig_time_ms[0],
        orig_time_ms[-1] + SAMPLING_MS,
        SAMPLING_MS
    )

    # --- interpolation ---
    new_ecg = np.interp(new_time_ms, orig_time_ms, sig)

    return pd.DataFrame({
        'TIMESTAMP_MS': new_time_ms.astype(np.int64),
        'ecg': new_ecg
    })


# ─── PROCESS ONE ICU STAY ───────────────────────────────────────────────
def process_one_icu(icu_id, record_list):
    """
    For a given icustay_id, iterate over all waveform records,
    resample ECG, segment into TIME_WINDOW-second windows,
    compute ECG features, and save a single CSV.

    Output CSV schema:
      Time_Elapsed_s, icustay_id, record, <ECG feature columns...>
    """
    out_path = os.path.join(OUT_DIR, f'ICU_{icu_id}_ecg_features.csv')
    if os.path.exists(out_path):
        return icu_id, None

    zero_ms = icu_start_ms[icu_id]  # ICU admission time as t0
    feats = []

    for rec_rel in record_list:
        start_ms = rec_start_ms_map.get(rec_rel)
        if start_ms is None:
            print(f"[ICU {icu_id}] no rec_start for {rec_rel}, skipping")
            continue

        try:
            df_ecg = load_and_resample_ecg(rec_rel, start_ms)
        except Exception as e:
            print(f"[ICU {icu_id}] skip {rec_rel}: {e}")
            continue
        if df_ecg.empty:
            continue

        # align so ICU admission → 0 ms
        df_ecg['TIMESTAMP_MS'] -= zero_ms
        df_ecg['TIMESTAMP'] = pd.to_datetime(df_ecg['TIMESTAMP_MS'], unit='ms')

        # segment valid windows
        df_ecg['Valid']      = df_ecg['ecg'].notna()
        df_ecg['Gap']        = df_ecg['Valid'] != df_ecg['Valid'].shift(1)
        df_ecg['Segment_ID'] = df_ecg['Gap'].cumsum()
        df_val = df_ecg[df_ecg['Valid']].copy()
        if df_val.empty:
            continue

        # 125 Hz for 8 ms grid
        sr      = 1000 // SAMPLING_MS          # = 125
        win_len = TIME_WINDOW * sr             # samples per window

        for seg_id, seg in df_val.groupby('Segment_ID'):
            arr   = seg['ecg'].values
            times = seg['TIMESTAMP_MS'].values
            n_win = len(arr) // win_len

            for i in tqdm(
                range(n_win),
                desc=f"ICU {icu_id} | rec {rec_rel} | seg {seg_id}",
                leave=False
            ):
                w = arr[i * win_len:(i + 1) * win_len]

                # simple flat-signal rejection (same SQI helper)
                if flat_sqi(w):
                    continue

                try:
                    # assume ECG_feature returns a dict-like of features
                    feat = ECG_feature(w, fs=sr)
                    if feat is None:
                        continue
                except Exception as e:
                    print(f"[ICU {icu_id}][{rec_rel}][seg {seg_id}][win {i}] feat error: {e}")
                    continue

                t0_ms = int(times[i * win_len])
                feat['Time_Elapsed_s'] = t0_ms // 1000
                feat['icustay_id']     = icu_id
                feat['record']         = rec_rel
                feats.append(feat)

    return (icu_id, pd.concat(feats, ignore_index=True)) if feats else (icu_id, None)


# ─── MAIN: PARALLELIZE OVER ICU STAYS ────────────────────────────────────
if __name__ == '__main__':
    total = len(icu_groups)
    completed = 0

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {}

        for icu, recs in icu_groups.items():

            # 1. skip based on ICU id (same rule as in PPG script)
            if int(icu) < 0:
                print(f"Skipping ICU {icu} (< 247311)")
                continue

            # 2. skip if output file already exists
            out_path = os.path.join(OUT_DIR, f'ICU_{icu}_ecg_features.csv')
            if os.path.exists(out_path):
                print(f"Skipping ICU {icu}: already processed")
                continue

            # otherwise submit job
            futures[executor.submit(process_one_icu, icu, recs)] = icu

        # collect results
        for fut in as_completed(futures):
            icu_id = futures[fut]
            completed += 1

            try:
                icu_id, df_feats = fut.result()
                print(f"Processed {completed}/{total} (ICU {icu_id})")

                if df_feats is not None:
                    out_path = os.path.join(OUT_DIR, f'ICU_{icu_id}_ecg_features.csv')
                    df_feats.to_csv(out_path, index=False)
                    print(f"→ Saved ICU {icu_id}: {len(df_feats)} windows")
                else:
                    print(f"→ ICU {icu_id}: no valid windows")

            except Exception as e:
                print(f"!!! ICU {icu_id} failed: {e}")
