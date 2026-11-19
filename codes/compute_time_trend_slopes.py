"""
Computes, for each (evaluation_type, modality, stat_set, classifier) combination,
the linear slope of AUROC, AUPRC, and Sensitivity against numeric time cutoffs
(2h-24h, 'full' excluded since it has no fixed duration), plus the raw endpoint
values for reference. Outputs a single long-format CSV containing both the raw
per-combo rows AND group-level rollup rows (mean by stat_set, by classifier, by
modality, and overall, within each evaluation_type) -- rollups are tagged via the
`aggregation_level` column so summary numbers appear directly in the file
instead of only being derivable from it.
"""
import numpy as np
import pandas as pd

NUMERIC_CUTOFFS = {'2h': 2, '4h': 4, '6h': 6, '12h': 12, '24h': 24}
METRIC_COLS = [
    'AUROC_slope_per_hr', 'AUPRC_slope_per_hr', 'Sensitivity_slope_per_hr', 'Specificity_slope_per_hr',
    'F1-score_slope_per_hr',
    'AUROC_2h', 'AUROC_24h', 'AUPRC_2h', 'AUPRC_24h', 'Sensitivity_2h', 'Sensitivity_24h',
    'Specificity_2h', 'Specificity_24h', 'F1-score_2h', 'F1-score_24h',
]

SOURCES = {
    'symmetric':  ('../data/time_window_comparison_results.csv', 'time_cutoff_hours'),
    'asymmetric': ('../data/train_full_test_truncated_comparison_results.csv', 'test_time_cutoff_hours'),
}


def compute_slopes(path, cutoff_col, evaluation_type):
    df = pd.read_csv(path).rename(columns={cutoff_col: 'cutoff'})
    df = df[df['cutoff'].isin(NUMERIC_CUTOFFS)].copy()
    df['cutoff_num'] = df['cutoff'].map(NUMERIC_CUTOFFS)

    rows = []
    for (modality, stat_set, clf), g in df.groupby(['modality', 'stat_set', 'classifier']):
        g = g.sort_values('cutoff_num')
        if len(g) < 2:
            continue
        auroc_slope = np.polyfit(g['cutoff_num'], g['AUROC_mean'], 1)[0]
        auprc_slope = np.polyfit(g['cutoff_num'], g['AUPRC_mean'], 1)[0]
        sens_slope = np.polyfit(g['cutoff_num'], g['Sensitivity_mean'], 1)[0]
        spec_slope = np.polyfit(g['cutoff_num'], g['Specificity_mean'], 1)[0]
        f1_slope = np.polyfit(g['cutoff_num'], g['F1-score_mean'], 1)[0]
        rows.append({
            'evaluation_type': evaluation_type,
            'modality': modality,
            'stat_set': stat_set,
            'classifier': clf,
            'AUROC_slope_per_hr': auroc_slope,
            'AUPRC_slope_per_hr': auprc_slope,
            'Sensitivity_slope_per_hr': sens_slope,
            'Specificity_slope_per_hr': spec_slope,
            'F1-score_slope_per_hr': f1_slope,
            'AUROC_2h': g['AUROC_mean'].iloc[0],
            'AUROC_24h': g['AUROC_mean'].iloc[-1],
            'AUPRC_2h': g['AUPRC_mean'].iloc[0],
            'AUPRC_24h': g['AUPRC_mean'].iloc[-1],
            'Sensitivity_2h': g['Sensitivity_mean'].iloc[0],
            'Sensitivity_24h': g['Sensitivity_mean'].iloc[-1],
            'Specificity_2h': g['Specificity_mean'].iloc[0],
            'Specificity_24h': g['Specificity_mean'].iloc[-1],
            'F1-score_2h': g['F1-score_mean'].iloc[0],
            'F1-score_24h': g['F1-score_mean'].iloc[-1],
        })
    return pd.DataFrame(rows)


def add_aggregates(all_slopes):
    """Append group-level rollup rows (mean across the other grouping
    dimensions, within each evaluation_type) to the per-combo rows. Rollup
    rows leave the columns they're averaged over blank and are tagged via
    `aggregation_level` so they're distinguishable from the raw combo rows."""
    all_slopes = all_slopes.copy()
    all_slopes['aggregation_level'] = 'combo'

    group_specs = [
        ('by_stat_set',   ['evaluation_type', 'stat_set']),
        ('by_classifier', ['evaluation_type', 'classifier']),
        ('by_modality',   ['evaluation_type', 'modality']),
        ('overall',       ['evaluation_type']),
    ]
    rollups = []
    for level_name, group_cols in group_specs:
        grouped = all_slopes.groupby(group_cols)[METRIC_COLS].mean().reset_index()
        grouped['aggregation_level'] = level_name
        rollups.append(grouped)

    combined = pd.concat([all_slopes] + rollups, ignore_index=True)
    col_order = ['aggregation_level', 'evaluation_type', 'modality', 'stat_set', 'classifier'] + METRIC_COLS
    return combined.reindex(columns=col_order)


def main():
    all_slopes = pd.concat(
        [compute_slopes(path, cutoff_col, name) for name, (path, cutoff_col) in SOURCES.items()],
        ignore_index=True
    )
    all_slopes = add_aggregates(all_slopes)
    out_path = '../data/time_trend_slopes.csv'
    all_slopes.to_csv(out_path, index=False)
    print(f"Wrote {len(all_slopes)} rows to {out_path}")
    print(all_slopes['aggregation_level'].value_counts())


if __name__ == "__main__":
    main()
