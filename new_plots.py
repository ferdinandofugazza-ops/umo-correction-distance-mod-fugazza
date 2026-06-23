import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker


def plot_umo_mean_vs_bias(output_path, feature_names, feature_weights, sensitive_column_name=None, sensitive_attribute_value=None, num_iterations=None):
    """
    Plot mean UMO level vs bias coefficient for each feature across all pollution modes.
    For all attributes, plots separate lines for correct predictions (TP+TN) and incorrect predictions (FP+FN).
    
    Parameters:
    -----------
    output_path : str
        Path to the output directory containing pollution mode subdirectories
    feature_names : list
        List of feature names to plot
    feature_weights : dict
        Dictionary mapping feature names to their weights
    sensitive_column_name : str, optional
        Name of the sensitive column
    sensitive_attribute_value : int, optional
        Value of the sensitive attribute (0 or 1)
    num_iterations : int, optional
        Number of iterations used in the experiment
    """
    
    pollution_modes = ['linear_pollution', 'probabilistic_pollution', 'probabilistic_half_lies']
    # bias values used in iterate_process
    bias_values = np.round(np.arange(0.4, 1.6, 0.2), 1)
    
    # New behavior: one image per pollution mode, containing one subplot per feature
    for pollution_mode in pollution_modes:
        n = len(feature_names)
        if n == 0:
            continue
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        for idx, feat in enumerate(feature_names):
            ax = axes_flat[idx]
            raw_path = os.path.join(output_path, pollution_mode, f'{feat}_umo_raw.csv')
            if not os.path.exists(raw_path):
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(feat)
                continue

            df_raw = pd.read_csv(raw_path)
            per_bias_cols = [c for c in df_raw.columns if str(c).startswith("UMO_")]
            if not per_bias_cols:
                ax.text(0.5, 0.5, 'No per-bias UMO columns', ha='center', va='center', transform=ax.transAxes, fontsize=11)
                ax.set_title(feat)
                continue

            has_pred_cols = ('mean_pred_0_removed_ft' in df_raw.columns) and ('true_class' in df_raw.columns)
            has_iterations = 'iteration' in df_raw.columns

            def mean_sem_over_iterations(df_subset, cols):
                means, sems = [], []
                if df_subset is None or df_subset.empty:
                    return np.array([]), np.array([])
                if has_iterations:
                    grouped = df_subset.groupby('iteration')
                    for col in cols:
                        per_iter_means = grouped[col].mean().dropna()
                        if per_iter_means.empty:
                            means.append(np.nan)
                            sems.append(np.nan)
                        else:
                            means.append(per_iter_means.mean())
                            n_i = len(per_iter_means)
                            sems.append(per_iter_means.std(ddof=1) / np.sqrt(n_i) if n_i > 1 else 0.0)
                else:
                    for col in cols:
                        means.append(df_subset[col].mean())
                        sems.append(np.nan)
                return np.array(means), np.array(sems)

            def bias_from_col(c):
                try:
                    return float(c.split('_', 1)[1])
                except Exception:
                    return float('nan')

            per_bias_cols = sorted(per_bias_cols, key=bias_from_col)
            biases_found = [bias_from_col(c) for c in per_bias_cols]

            # For each sensitive attribute value (if present) plot correct/incorrect
            if sensitive_column_name in df_raw.columns:
                for attr in sorted(df_raw[sensitive_column_name].unique()):
                    df_attr = df_raw[df_raw[sensitive_column_name] == attr]
                    if attr == sensitive_attribute_value:
                        color_correct = 'C1'
                        color_incorrect = 'C3'
                        attr_label = f'Attr={attr} (Sensitive)'
                    else:
                        color_correct = 'C0'
                        color_incorrect = 'C2'
                        attr_label = f'Attr={attr}'

                    if has_pred_cols:
                        correct_mask = (df_attr['mean_pred_0_removed_ft'] == df_attr['true_class'])
                        df_correct = df_attr[correct_mask]
                        df_incorrect = df_attr[~correct_mask]

                        if not df_correct.empty:
                            y_correct, se_correct = mean_sem_over_iterations(df_correct, per_bias_cols)
                            ax.plot(biases_found, y_correct, marker='o', label=f'{attr_label} (Correct)', linewidth=2, markersize=5, linestyle='-', color=color_correct)
                            if has_iterations and y_correct.size:
                                ax.fill_between(biases_found, y_correct - se_correct, y_correct + se_correct, color=color_correct, alpha=0.2, linewidth=0)

                        if not df_incorrect.empty:
                            y_incorrect, se_incorrect = mean_sem_over_iterations(df_incorrect, per_bias_cols)
                            ax.plot(biases_found, y_incorrect, marker='s', label=f'{attr_label} (Incorrect)', linewidth=2, markersize=5, linestyle='--', color=color_incorrect)
                            if has_iterations and y_incorrect.size:
                                ax.fill_between(biases_found, y_incorrect - se_incorrect, y_incorrect + se_incorrect, color=color_incorrect, alpha=0.2, linewidth=0)
                    else:
                        y, se = mean_sem_over_iterations(df_attr, per_bias_cols)
                        ax.plot(biases_found, y, marker='o', label=attr_label, linewidth=2, markersize=5)
                        if has_iterations and y.size:
                            ax.fill_between(biases_found, y - se, y + se, alpha=0.2, linewidth=0)
            else:
                # No sensitive column: plot aggregate mean
                y, se = mean_sem_over_iterations(df_raw, per_bias_cols)
                ax.plot(biases_found, y, marker='o', label=f'{feat}', linewidth=2, markersize=5)
                if has_iterations and y.size:
                    ax.fill_between(biases_found, y - se, y + se, alpha=0.2, linewidth=0)

            ax.set_ylabel('Mean UMO Level', fontsize=9)
            ax.set_xlabel(f'Bias for {feat}', fontsize=9)
            weight_label = (feature_weights or {}).get(feat, '?')
            ax.set_title(f'w={weight_label}\n{feat}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

        # hide any unused subplots
        total_axes = len(axes_flat)
        for j in range(n, total_axes):
            axes_flat[j].axis('off')

        iterations_text = f'\nNumber of iterations: {num_iterations}' if num_iterations is not None else ''
        fig.suptitle(f'Mean UMO Level vs Bias Coefficient - Mode: {pollution_mode}{iterations_text}', fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        plots_dir = os.path.join(output_path, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        out_file = os.path.join(plots_dir, f'{pollution_mode}_umo_mean_vs_bias_{num_iterations}_iterations.png')
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved per-mode plot: {out_file}")


def plot_umo_statistics_summary(output_path, feature_names, feature_weights, stat_name='umo_mean',
                                mode_list=None, sensitive_column_name=None,
                                sensitive_attribute_value=None,
                                plot_type='step', colors=None, num_iterations=None):
   
    default_pollution_modes = ['linear_pollution', 'probabilistic_pollution', 'probabilistic_half_lies']
    if mode_list is not None:
        pollution_modes = mode_list
    else:
        pollution_modes = default_pollution_modes.copy()
    
    if not pollution_modes or len(pollution_modes) == 0:
        print(f"Warning: No pollution modes to plot. mode_list={mode_list}")
        return

    def get_palette(n):
        try:
            return sns.color_palette('tab10', n_colors=max(6, n))
        except Exception:
            prop_cycle = plt.rcParams.get('axes.prop_cycle')
            colors = prop_cycle.by_key().get('color', ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
            if len(colors) < n:
                colors = (colors * ((n // len(colors)) + 1))[:n]
            return colors

    plots_dir = os.path.join(output_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    for pollution_mode in pollution_modes:
        n = len(feature_names)
        if n == 0:
            continue

        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for idx, feat in enumerate(feature_names):
            ax = axes_flat[idx]
            raw_path = os.path.join(output_path, pollution_mode, f"{feat}_umo_raw.csv")
            if not os.path.exists(raw_path):
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(feat)
                continue

            df_raw = pd.read_csv(raw_path)
            per_bias_cols = [c for c in df_raw.columns if str(c).startswith("UMO_")]
            if not per_bias_cols:
                ax.text(0.5, 0.5, 'No UMO_* columns found', ha='center', va='center', transform=ax.transAxes, fontsize=11)
                ax.set_title(feat)
                continue

            def bias_from_col(c):
                try:
                    return float(c.split('_', 1)[1])
                except Exception:
                    return float('nan')

            per_bias_cols = sorted(per_bias_cols, key=bias_from_col)
            biases = [bias_from_col(c) for c in per_bias_cols]

            try:
                max_val = int(df_raw[per_bias_cols].max().max())
            except Exception:
                max_val = int(pd.to_numeric(df_raw[per_bias_cols].max().max(), errors='coerce') or 0)
            if np.isnan(max_val) or max_val is None:
                max_val = 0

            x_values = np.arange(0, max_val + 1)
            n_plots = len(per_bias_cols)
            palette = colors if (colors is not None and len(colors) >= n_plots) else get_palette(n_plots)
            bar_width = 0.8 / n_plots
            has_iterations = 'iteration' in df_raw.columns

            all_counts = []
            for i, col in enumerate(per_bias_cols):
                if (sensitive_column_name is not None) and (sensitive_attribute_value is not None) and (sensitive_column_name in df_raw.columns):
                    df_filtered = df_raw[df_raw[sensitive_column_name] == sensitive_attribute_value]
                else:
                    df_filtered = df_raw

                if has_iterations:
                    iteration_counts = []
                    for iter_num in df_filtered['iteration'].unique():
                        iter_data = df_filtered[df_filtered['iteration'] == iter_num]
                        series = iter_data[col].dropna()
                        try:
                            series = series.astype(int)
                        except Exception:
                            series = np.floor(pd.to_numeric(series, errors='coerce')).dropna().astype(int)

                        if series.empty:
                            counts = np.zeros_like(x_values, dtype=float)
                        else:
                            vc = series.value_counts().sort_index()
                            counts = np.array([float(vc.get(x, 0)) for x in x_values])
                        iteration_counts.append(counts)

                    if iteration_counts:
                        avg_counts = np.mean(iteration_counts, axis=0)
                    else:
                        avg_counts = np.zeros_like(x_values, dtype=float)

                    series_all = df_filtered[col].dropna()
                    try:
                        series_all = series_all.astype(int)
                    except Exception:
                        series_all = np.floor(pd.to_numeric(series_all, errors='coerce')).dropna().astype(int)
                else:
                    series = df_filtered[col].dropna()
                    series_all = series
                    try:
                        series = series.astype(int)
                    except Exception:
                        series = np.floor(pd.to_numeric(series, errors='coerce')).dropna().astype(int)
                    if series.empty:
                        avg_counts = np.zeros_like(x_values, dtype=float)
                    else:
                        vc = series.value_counts().sort_index()
                        avg_counts = np.array([float(vc.get(x, 0)) for x in x_values])

                all_counts.append((avg_counts, series_all))

            for i, (counts, series) in enumerate(all_counts):
                label = f"{biases[i]:.1f}" if not np.isnan(biases[i]) else per_bias_cols[i]
                color = palette[i % len(palette)]
                offset = (i - n_plots / 2 + 0.5) * bar_width
                bar_positions = x_values + offset
                ax.bar(bar_positions, counts, width=bar_width, label=label, color=color,
                      alpha=0.8, edgecolor='black', linewidth=0.5)
                if not series.empty:
                    ax.axvline(series.mean(), color=color, linestyle='--', linewidth=1.5, alpha=0.7)

            ax.set_xlabel('UMO level')
            ylabel = f'Average Count Across {num_iterations} Iterations' if has_iterations else 'Count'
            ax.set_ylabel(ylabel)
            weight_label = (feature_weights or {}).get(feat, '?')
            ax.set_title(f'w={weight_label}\n{feat}', fontsize=10)
            ax.set_xticks(x_values)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(title='Bias', fontsize=7)
            info_text = 'Dashed lines = mean'
            if has_iterations:
                info_text += f'\nAveraged over {num_iterations} iterations'
            ax.text(0.02, 0.98, info_text,
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        for j in range(n, len(axes_flat)):
            axes_flat[j].axis('off')

        iterations_text = f'\nNumber of iterations: {num_iterations}' if num_iterations is not None else ''
        fig.suptitle(f'UMO values count for sensitive={sensitive_attribute_value} — Mode: {pollution_mode}{iterations_text}', fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_file = os.path.join(plots_dir, f'{pollution_mode}_umo_distributions_by_bias_{num_iterations}_iterations.png')
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved UMO distributions plot for mode: {pollution_mode} -> {out_file}")

def plot_accuracy_by_group(
    output_path,
    feature_names,
    feature_weights=None,
    sensitive_column_name='dummy_sensitive',
    sensitive_attribute_value=0,
    num_iterations=None
):
    """
    For each pollution mode, plot accuracy and F1 across bias levels for
    each polluted feature. Produces one file per pollution mode.

    Parameters:
    -----------
    output_path : str
        Path to the output directory containing pollution mode subdirectories
    feature_names : list
        List of feature names to plot
    feature_weights : dict, optional
        Dictionary mapping feature names to their weights
    sensitive_column_name : str, optional
        Name of the sensitive column
    sensitive_attribute_value : int, optional
        Sensitive attribute value to treat as the target group
    num_iterations : int, optional
        Number of iterations used in the experiment
    """

    pollution_modes = ['linear_pollution', 'probabilistic_pollution', 'probabilistic_half_lies']
    pollution_labels = {
        'linear_pollution':           'Linear pollution',
        'probabilistic_pollution':    'Probabilistic pollution',
        'probabilistic_half_lies':    'Probabilistic half-lies'
    }

    bias_levels = [round(b, 1) for b in np.arange(0.4, 1.6, 0.2)]

    group_colors = {
        'group0': 'C1',   # blue  — minority
        'group1': 'C0',   # orange-red — majority
        'overall': 'C2'   # grey
    }
    metric_styles = {
        'accuracy': {'ls': '-',  'marker': 'o', 'label': 'Accuracy'},
        'f1':       {'ls': '--', 'marker': 's', 'label': 'F1'},
    }
    metrics_to_plot = ['accuracy', 'f1']

    plots_dir = os.path.join(output_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    for pollution_mode in pollution_modes:
        n = len(feature_names)
        if n == 0:
            continue

        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharey=True)
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for idx, feat in enumerate(feature_names):
            ax = axes_flat[idx]
            raw_csv = os.path.join(output_path, pollution_mode, f'{feat}_umo_raw.csv')
            if not os.path.isfile(raw_csv):
                ax.set_title(feat)
                ax.text(0.5, 0.5, 'No raw data', transform=ax.transAxes,
                        ha='center', va='center', color='grey')
                continue

            raw_df = pd.read_csv(raw_csv)
            if sensitive_column_name not in raw_df.columns:
                ax.set_title(feat)
                ax.text(0.5, 0.5, 'No sensitive column', transform=ax.transAxes,
                        ha='center', va='center', color='grey')
                continue

            group_vals = sorted(raw_df[sensitive_column_name].unique())
            group_keys = {gv: f'group{gv}' for gv in group_vals}
            per_bias = {gk: {m: [] for m in metrics_to_plot} for gk in group_keys.values()}

            for bias in bias_levels:
                umo_col = f'UMO_{bias}'
                if umo_col not in raw_df.columns:
                    for gk in group_keys.values():
                        for m in metrics_to_plot:
                            per_bias[gk][m].append(np.nan)
                    continue

                sub = raw_df.dropna(subset=[umo_col])
                for gv, gk in group_keys.items():
                    g_sub = sub[sub[sensitive_column_name] == gv]
                    if g_sub.empty or 'mean_pred_0_removed_ft' not in g_sub.columns:
                        for m in metrics_to_plot:
                            per_bias[gk][m].append(np.nan)
                        continue

                    y_pred = g_sub['mean_pred_0_removed_ft'].values.astype(int)
                    y_true = g_sub['true_class'].values.astype(int)
                    from sklearn.metrics import accuracy_score, f1_score
                    acc = accuracy_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    per_bias[gk]['accuracy'].append(acc)
                    per_bias[gk]['f1'].append(f1)

            for gv, gk in group_keys.items():
                color = group_colors.get(gk, '#888888')
                group_label = ('Sensitive attribute = 0' if gv == sensitive_attribute_value
                               else 'Sensitive attribute = 1')
                for metric in metrics_to_plot:
                    style = metric_styles[metric]
                    values = per_bias[gk][metric]
                    ax.plot(
                        bias_levels, values,
                        color=color,
                        ls=style['ls'],
                        marker=style['marker'],
                        markersize=5,
                        linewidth=1.8,
                        label=f"{group_label} — {style['label']}"
                    )

            ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
            weight_label = (feature_weights or {}).get(feat, '?')
            ax.set_title(f'w={weight_label}\n{feat}', fontsize=10)
            ax.set_xlabel('Bias coefficient')
            if idx % ncols == 0:
                ax.set_ylabel('Metric value')
            ax.set_ylim(0, 1.05)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.legend(fontsize=7)

        for j in range(n, len(axes_flat)):
            axes_flat[j].axis('off')

        iterations_text = f'\nNumber of iterations: {num_iterations}' if num_iterations is not None else ''
        fig.suptitle(f'Per-group performance metrics vs bias — Mode: {pollution_mode}{iterations_text}', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(plots_dir, f'{pollution_mode}_accuracy_by_group.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

def plot_umo_disparity_gap(output_path, feature_names, feature_weights, sensitive_column_name, sensitive_attribute_value=0, num_iterations=None):
    """
    Plots the difference in mean UMO between sensitive attribute group and the others,
    across bias levels. Produces one PNG per pollution mode and one line per feature.

    Parameters:
    -----------
    output_path : str
        Path to the output directory containing pollution mode subdirectories
    feature_names : list
        List of feature names to plot
    sensitive_column_name : str
        Name of the sensitive column
    feature_weights : dict,
        Dictionary mapping feature names to their weights
    sensitive_attribute_value : int, optional
        Value of the sensitive attribute to use for the sensitive group
    num_iterations : int, optional
        Number of iterations used in the experiment
    """
    pollution_modes = ['linear_pollution', 'probabilistic_pollution', 'probabilistic_half_lies']
    bias_levels = np.round(np.arange(0.4, 1.6, 0.2), 1)
    plots_dir = os.path.join(output_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    for mode in pollution_modes:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        plotted_any = False

        for feat in feature_names:
            raw_path = os.path.join(output_path, mode, f"{feat}_umo_raw.csv")
            if not os.path.exists(raw_path):
                continue

            df_raw = pd.read_csv(raw_path)
            if sensitive_column_name not in df_raw.columns:
                continue

            deltas = []
            valid_biases = []
            for bias in bias_levels:
                umo_col = f'UMO_{bias}'
                if umo_col not in df_raw.columns:
                    continue

                valid_data = df_raw.dropna(subset=[umo_col, sensitive_column_name])
                if valid_data.empty:
                    continue

                mean_sensitive = valid_data[valid_data[sensitive_column_name] == sensitive_attribute_value][umo_col].mean()
                mean_other = valid_data[valid_data[sensitive_column_name] != sensitive_attribute_value][umo_col].mean()
                if np.isnan(mean_sensitive) or np.isnan(mean_other):
                    continue

                delta = mean_sensitive - mean_other
                deltas.append(delta)
                valid_biases.append(bias)

            if not valid_biases:
                continue

            weight_label = (feature_weights or {}).get(feat)
            label = f'{feat} (w={weight_label})' if weight_label is not None else feat
            ax.plot(valid_biases, deltas, marker='o', linewidth=2, markersize=6, label=label)
            plotted_any = True

        ax.axhline(0, color='black', linestyle='--', alpha=0.6, label='No Gap')
        ax.axvline(1.0, color='grey', linestyle=':', alpha=0.6, label='No Pollution (Bias=1.0)')

        if not plotted_any:
            ax.text(0.5, 0.5, 'No data available for this pollution mode', ha='center', va='center', transform=ax.transAxes, fontsize=12)

        ax.set_xlabel('Bias Coefficient')
        ax.set_ylabel('Mean UMO Difference (Sensitive - Non-sensitive)')
        ax.set_title(mode.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        iterations_text = f'\nNumber of iterations: {num_iterations}' if num_iterations is not None else ''
        
        fig.suptitle(f'UMO Disparity Gap vs Bias - Mode: {mode}{iterations_text}', fontsize=13, fontweight='bold', y=1.03)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_file = os.path.join(plots_dir, f'{mode}_umo_disparity_gap.png')
        plt.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved UMO Disparity Gap plot for mode: {mode} -> {out_file}")
