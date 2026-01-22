import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('ggplot')  # apply ggplot aesthetics

# Use Arial font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

data = pd.read_csv('../../../data/dataset_index.csv')
data = data[
    (data['dataset_type']=='test') &
    (data['length']<=9) &
    (data['length']>=5)
]

data_VQE = pd.read_csv("VQE_time_memory.csv")
data_QAOA = pd.read_csv("QAOA_time_memory.csv")
data_QNN = pd.read_csv("QNN_time_memory.csv")

# Align lengths to method dataframes by index (NO truncation; must be equal and aligned)
lengths = data['length'].reset_index(drop=True)

def attach_lengths(df: pd.DataFrame, lengths: pd.Series) -> pd.DataFrame:
    if len(df) != len(lengths):
        raise ValueError(f"Length mismatch: df={len(df)} vs lengths={len(lengths)}; alignment is required.")
    df = df.copy()
    df['length'] = lengths.values
    return df

data_VQE = attach_lengths(data_VQE, lengths)
data_QAOA = attach_lengths(data_QAOA, lengths)
data_QNN = attach_lengths(data_QNN, lengths)
# QNN qubits fixed at 7
data_QNN['qubits'] = 7

# Assemble methods for comparison
methods = {
    'VQE': data_VQE[['length', 'time', 'memory', 'qubits']].copy(),
    'QAOA': data_QAOA[['length', 'time', 'memory', 'qubits']].copy(),
    'QSyncFold': data_QNN[['length', 'time', 'memory', 'qubits']].copy(),
}

# New: Draw one figure per metric comparing all methods
def plot_metric_comparison(methods: dict, metric: str, out_path: str, title: str):
    length_values = sorted(set().union(*[df['length'].dropna().unique().tolist() for df in methods.values()]))
    if not length_values:
        print(f"No length data to plot for {metric}")
        return

    k = len(methods)
    base = np.arange(len(length_values)) + 1
    group_width = 0.8
    delta = group_width / k
    offsets = (np.arange(k) - (k - 1) / 2) * delta

    # Dynamic width
    fig_w = max(8, 1.2 * len(length_values))
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 4))

    color_map = {
        'VQE': '#8ecae6',
        'QAOA': '#f4a261',
        'QSyncFold': '#90be6d',
    }

    handles = []
    for idx, (name, df) in enumerate(methods.items()):
        groups = [df.loc[df['length'] == L, metric].dropna().values for L in length_values]
        pos = base + offsets[idx]
        bp = ax.boxplot(
            groups,
            positions=pos,
            widths=delta * 0.8,
            patch_artist=True,
            showfliers=False
        )
        for patch in bp['boxes']:
            patch.set(facecolor=color_map[name], alpha=0.35, edgecolor=color_map[name])
        for median in bp['medians']:
            median.set(color=color_map[name], linewidth=2)

        means = [np.mean(g) if len(g) else np.nan for g in groups]
        line, = ax.plot(pos, means, color=color_map[name], marker='o', markersize=5, linewidth=2, label=name)
        handles.append(line)

    ax.set_xticks(base)
    ax.set_xticklabels([str(L) for L in length_values])
    ax.set_xlabel("Protein length")
    ylabel_map = {
        'time': 'Time (s)',
        'memory': 'Memory (MB)',
        'qubits': 'Qubits',
    }
    ax.set_ylabel(ylabel_map.get(metric, metric.capitalize()))
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(handles=handles, loc='upper left')
    # remove title

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# Replace per-metric comparison plots
plot_metric_comparison(
    methods, metric='time',
    out_path="compare_time_by_length.pdf",
    title=""
)
plot_metric_comparison(
    methods, metric='memory',
    out_path="compare_memory_by_length.pdf",
    title=""
)
plot_metric_comparison(
    methods, metric='qubits',
    out_path="compare_qubits_by_length.pdf",
    title=""
)


