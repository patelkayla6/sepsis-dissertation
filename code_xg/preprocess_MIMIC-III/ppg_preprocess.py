#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from preprocessing_utils import flat_sqi
from ppg_feature        import PPG_feature

# ─── CONFIG ─────────────────────────────────────────────────────────────
WFDB_PATH    = '/media/engs2588/DATA/Data/physionet.org/files/mimic3wdb-matched/1.0'
KEEPERS_CSV  = 'mimic_overlap_keepers.csv'
SEPSIS3_CSV  = '/mnt/Data/engs2588/thirdparty/sepsis3-mimic/data/sepsis3-df.csv'
PPG_CHANNELS = ['PLETH', 'PLETH L', 'PLETH R']
SAMPLING_MS  = 10    # grid interval in ms
TIME_WINDOW  = 20    # window length in s
OUT_DIR      = 'ppg_features_by_icu_v2'

os.makedirs(OUT_DIR, exist_ok=True)

# ─── LOAD KEEPERS & GROUP BY ICU ──────────────────────────────────────────
ov_df = pd.read_csv(KEEPERS_CSV)
# ensure rec_start and rec_end are datetimes
ov_df['rec_start'] = pd.to_datetime(ov_df['rec_start'], format='%Y-%m-%d %H:%M:%S')
ov_df['rec_end']   = pd.to_datetime(ov_df['rec_end'],   format='%Y-%m-%d %H:%M:%S')
icu_groups = ov_df.groupby('icustay_id')['record_rel'].apply(list).to_dict()

# ─── LOAD SEPSIS3 DF & BUILD ICU-LEVEL ZERO POINTS ────────────────────────
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

# ─── BUILD RECORD-LEVEL START MAP ────────────────────────────────────────
rec_start_series = ov_df.set_index('record_rel')['rec_start']
rec_start_ms_map = (rec_start_series.astype('int64') // 10**6).to_dict()

# ─── EXTRACT & RESAMPLE ONE RECORD ────────────────────────────────────────
def load_and_resample_ppg(record_rel, start_ms):
    rec = wfdb.rdrecord(os.path.join(WFDB_PATH, record_rel))
    available = [ch for ch in PPG_CHANNELS if ch in rec.sig_name]
    if not available:
        raise ValueError(f"No PPG channel in {record_rel}")
    idx = rec.sig_name.index(available[0])
    sig = rec.p_signal[:, idx]
    fs  = rec.fs

    # absolute timestamps = record start_ms + per-sample offset
    times = (np.arange(len(sig)) * (1000.0 / fs)).astype(int) + start_ms
    df_ppg = pd.DataFrame({'TIMESTAMP_MS': times, 'ppg': sig})
    df_ppg['TIMESTAMP'] = pd.to_datetime(df_ppg['TIMESTAMP_MS'], unit='ms')

    df_ppg = (
        df_ppg
          .set_index('TIMESTAMP')
          .resample(f'{SAMPLING_MS}ms')
          .mean()
          .interpolate()
          .reset_index()
    )
    df_ppg['TIMESTAMP_MS'] = df_ppg['TIMESTAMP'].astype('int64') // 10**6
    return df_ppg[['TIMESTAMP_MS', 'ppg']]

# ─── PROCESS ONE ICU STAY ────────────────────────────────────────────────
def process_one_icu(icu_id, record_list):
    out_path = os.path.join(OUT_DIR, f'ICU_{icu_id}_ppg_features.csv')
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
            df_ppg = load_and_resample_ppg(rec_rel, start_ms)
        except Exception as e:
            print(f"[ICU {icu_id}] skip {rec_rel}: {e}")
            continue
        if df_ppg.empty:
            continue

        # align so ICU admission → 0
        df_ppg['TIMESTAMP_MS'] -= zero_ms
        df_ppg['TIMESTAMP']     = pd.to_datetime(df_ppg['TIMESTAMP_MS'], unit='ms')

        # segment valid windows
        df_ppg['Valid']      = df_ppg['ppg'].notna()
        df_ppg['Gap']        = df_ppg['Valid'] != df_ppg['Valid'].shift(1)
        df_ppg['Segment_ID'] = df_ppg['Gap'].cumsum()
        df_val = df_ppg[df_ppg['Valid']].copy()
        if df_val.empty:
            continue

        sr      = 1000 // SAMPLING_MS
        win_len = TIME_WINDOW * sr

        for seg_id, seg in df_val.groupby('Segment_ID'):
            arr   = seg['ppg'].values
            times = seg['TIMESTAMP_MS'].values
            n_win = len(arr) // win_len

            for i in tqdm(
                range(n_win),
                desc=f"ICU {icu_id} | rec {rec_rel} | seg {seg_id}",
                leave=False
            ):
                w = arr[i*win_len:(i+1)*win_len]
                if flat_sqi(w):
                    continue
                try:
                    feat = PPG_feature(w, fs=sr)
                except Exception as e:
                    print(f"[ICU {icu_id}][{rec_rel}][seg {seg_id}][win {i}] feat error: {e}")
                    continue
                feat['Time_Elapsed_s'] = times[i*win_len] // 1000
                feat['icustay_id']     = icu_id
                feat['record']         = rec_rel
                feats.append(feat)

    return (icu_id, pd.concat(feats, ignore_index=True)) if feats else (icu_id, None)

# ─── MAIN: PARALLELIZE OVER ICU STAYS ────────────────────────────────────
if __name__ == '__main__':
    total = len(icu_groups)
    completed = 0

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_one_icu, icu, recs): icu
            for icu, recs in icu_groups.items()
        }

        for fut in as_completed(futures):
            completed += 1
            icu_id = futures[fut]
            try:
                icu_id, df_feats = fut.result()
                print(f"Processed {completed}/{total} ICU stays (ICU {icu_id})")

                if df_feats is not None:
                    out_path = os.path.join(OUT_DIR, f'ICU_{icu_id}_ppg_features.csv')
                    df_feats.to_csv(out_path, index=False)
                    print(f"→ Saved ICU {icu_id}: {len(df_feats)} windows")
                else:
                    print(f"→ ICU {icu_id}: skipped")
            except Exception as e:
                print(f"!!! ICU {icu_id} failed: {e}")
