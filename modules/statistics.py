import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency

# === Significance tests ===
def run_feature_tests(df: pd.DataFrame, features: list, target='Hospital_LOS_Category'):
    plos = df[df[target] == 1]
    normal = df[df[target] == 0]

    results = []
    for feature in features:
        if df[feature].dtype in [np.int64, np.float64]:
            stat, p_value = ttest_ind(plos[feature].dropna(), normal[feature].dropna())
            test_type = "t-test"
            mean_plos = plos[feature].mean()
            mean_normal = normal[feature].mean()
        else:
            table = pd.crosstab(df[feature], df[target])
            stat, p_value, _, _ = chi2_contingency(table)
            test_type = "chi2"
            mean_plos = plos[feature].value_counts(normalize=True).to_dict()
            mean_normal = normal[feature].value_counts(normalize=True).to_dict()

        results.append([feature, test_type, mean_plos, mean_normal, stat, p_value, p_value < 0.05])

    return pd.DataFrame(results, columns=[
        'Feature', 'Test', 'PLOS Value', 'Normal LOS Value', 'Statistic', 'P-value', 'Significant'
    ])

# === LOS Median & IQR ===
def summarize_los(df: pd.DataFrame, target='Hospital_LOS_Category', los_col='Hospital LOS Days'):
    grouped = df.groupby(target)[los_col].agg([
        'median',
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75)
    ])
    grouped.columns = ['Median', '25th Percentile', '75th Percentile']
    return grouped

# === Table 1 Generator ===
def generate_table1(df: pd.DataFrame, num_cols: list, cat_cols: list, target='Hospital_LOS_Category'):
    def summarize_num(series):
        med = series.median()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        return f"{med:.1f} ({q1:.1f}–{q3:.1f})"

    def summarize_cat(series):
        counts = series.value_counts(dropna=False)
        total = counts.sum()
        summary = {}
        for level in counts.index:
            label = f"{series.name}_{level}"
            pct = 100 * counts[level] / total
            summary[label] = f"{counts[level]} ({pct:.1f}%)"
        return summary

    def summarize_group(sub_df):
        result = {}
        for col in num_cols:
            result[col] = summarize_num(sub_df[col])
        for col in cat_cols:
            result.update(summarize_cat(sub_df[col]))
        return result

    full = summarize_group(df)
    typical = summarize_group(df[df[target] == 0])
    plos = summarize_group(df[df[target] == 1])

    all_keys = sorted(set(full) | set(typical) | set(plos))
    table = pd.DataFrame(index=all_keys)
    table['All Patients'] = table.index.map(full.get)
    table['Typical LOS'] = table.index.map(typical.get)
    table['Prolonged LOS'] = table.index.map(plos.get)

    return table
