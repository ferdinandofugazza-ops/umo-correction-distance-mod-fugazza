import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_umo_mean_vs_bias(output_path, feature_names, feature_weights, sensitive_column_name=None, sensitive_attribute_value=None, num_iterations=None):
    """
    Plot mean UMO level vs bias coefficient for each feature across all pollution modes.
    For the sensitive attribute, plots separate lines for correct predictions (TP+TN) and incorrect predictions (FP+FN).
    
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
    
    for feat in feature_names:
        # Create figure with one subplot per pollution mode
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes = axes.flatten()
        
        for mode_idx, pollution_mode in enumerate(pollution_modes):
            ax = axes[mode_idx]
            raw_path = os.path.join(output_path, pollution_mode, f'{feat}_umo_raw.csv')
            
            if not os.path.exists(raw_path):
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{pollution_mode.replace("_", " ").title()}')
                continue
            
            df_raw = pd.read_csv(raw_path)
            
            # Find columns corresponding to per-bias UMO means (e.g. "UMO_0.4")
            per_bias_cols = [c for c in df_raw.columns if str(c).startswith("UMO_")]
            if not per_bias_cols:
                ax.text(0.5, 0.5, 'No per-bias UMO columns', ha='center', va='center', transform=ax.transAxes, fontsize=11)
                ax.set_title(f'{pollution_mode.replace("_", " ").title()}')
                continue
            
            # Check if we have the necessary columns for prediction correctness
            has_pred_cols = ('mean_pred_0_removed_ft' in df_raw.columns) and ('true_class' in df_raw.columns)
            
            # sort per_bias_cols by numeric bias extracted from the column name
            def bias_from_col(c):
                try:
                    return float(c.split('_', 1)[1])
                except Exception:
                    return float('nan')
            per_bias_cols = sorted(per_bias_cols, key=bias_from_col)
            biases_found = [bias_from_col(c) for c in per_bias_cols]
            
            # For each sensitive attribute value
            for attr in df_raw[sensitive_column_name].unique() if sensitive_column_name in df_raw.columns else []:
                df_attr = df_raw[df_raw[sensitive_column_name] == attr]
                
                if attr == sensitive_attribute_value and has_pred_cols:
                    # For sensitive attribute, split into correct and incorrect predictions
                    # Correct: (pred == 1 and true == 1) OR (pred == 0 and true == 0)
                    correct_mask = (df_attr['mean_pred_0_removed_ft'] == df_attr['true_class'])
                    df_correct = df_attr[correct_mask]
                    df_incorrect = df_attr[~correct_mask]
                    
                    # Plot correct predictions (TP + TN)
                    if not df_correct.empty:
                        y_correct = [df_correct[col].mean() for col in per_bias_cols]
                        ax.plot(biases_found, y_correct, marker='o', label=f'Attr={attr} (Correct)', 
                               linewidth=2, markersize=6, linestyle='-', color='C1')
                    
                    # Plot incorrect predictions (FP + FN)
                    if not df_incorrect.empty:
                        y_incorrect = [df_incorrect[col].mean() for col in per_bias_cols]
                        ax.plot(biases_found, y_incorrect, marker='o', label=f'Attr={attr} (Incorrect)', 
                               linewidth=2, markersize=6, linestyle='-', color='C2')
                else:
                    # For non-sensitive attribute, plot as before
                    y = [df_attr[col].mean() for col in per_bias_cols]
                    ax.plot(biases_found, y, marker='s', label=f'Attr={attr}', linewidth=2, markersize=6, linestyle='--', color='C0')
            
            ax.set_ylabel('Mean UMO Level', fontsize=11)
            ax.set_xlabel(f'Noise Coefficient for {sensitive_attribute_value} individuals over {feat}', fontsize=11)
            ax.set_title(f'{pollution_mode.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        fig.suptitle(f'Mean UMO Level vs Bias Coefficient - Feature: {feat}, Feature weight: {feature_weights[feat]}\nNumber of iterations: {num_iterations}', fontsize=11, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        plots_dir = os.path.join(output_path, 'plots')
        if not os.path.isdir(plots_dir):
            os.makedirs(plots_dir)
        
        plt.savefig(os.path.join(plots_dir, f'{feat}_umo_mean_vs_bias_{num_iterations}_iterations.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved plot for feature: {feat}")


def plot_umo_statistics_summary(output_path, feature_names, feature_weights, stat_name='umo_mean',
                                mode_list=None, sensitive_column_name=None,
                                sensitive_attribute_value=None,
                                plot_type='step', colors=None, num_iterations=None):
   
    default_pollution_modes = ['linear_pollution', 'probabilistic_pollution', 'probabilistic_half_lies']
    # If mode_list is provided, use it directly; otherwise use defaults
    if mode_list is not None:
        pollution_modes = mode_list
    else:
        pollution_modes = default_pollution_modes.copy()
    
    # Safety check: ensure we have at least one mode to plot
    if not pollution_modes or len(pollution_modes) == 0:
        print(f"Warning: No pollution modes to plot. mode_list={mode_list}")
        return

    # palette helper
    def get_palette(n):
        try:
            return sns.color_palette('tab10', n_colors=max(6, n))
        except Exception:
            # fallback to matplotlib default cycle
            prop_cycle = plt.rcParams.get('axes.prop_cycle')
            colors = prop_cycle.by_key().get('color', ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
            if len(colors) < n:
                # repeat if needed
                colors = (colors * ((n // len(colors)) + 1))[:n]
            return colors

    for feat in feature_names:
        n_modes = len(pollution_modes)
        fig, axes = plt.subplots(1, n_modes, figsize=(5 * n_modes, 5))
        if n_modes == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax_idx, pollution_mode in enumerate(pollution_modes):
            ax = axes[ax_idx]
            raw_path = os.path.join(output_path, pollution_mode, f"{feat}_umo_raw.csv")
            if not os.path.exists(raw_path):
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(pollution_mode.replace("_", " ").title())
                ax.set_xlabel('UMO level')
                ax.set_ylabel('Average Count')
                continue

            df_raw = pd.read_csv(raw_path)
            per_bias_cols = [c for c in df_raw.columns if str(c).startswith("UMO_")]
            if not per_bias_cols:
                ax.text(0.5, 0.5, 'No UMO_* columns found', ha='center', va='center', transform=ax.transAxes, fontsize=11)
                ax.set_title(pollution_mode.replace("_", " ").title())
                continue

            def bias_from_col(c):
                try:
                    return float(c.split('_', 1)[1])
                except Exception:
                    return float('nan')

            per_bias_cols = sorted(per_bias_cols, key=bias_from_col)
            biases = [bias_from_col(c) for c in per_bias_cols]

            # determine UMO integer range across all bias columns
            try:
                max_val = int(df_raw[per_bias_cols].max().max())
            except Exception:
                max_val = int(pd.to_numeric(df_raw[per_bias_cols].max().max(), errors='coerce') or 0)
            if np.isnan(max_val) or max_val is None:
                max_val = 0
            
            x_values = np.arange(0, max_val + 1)
            n_plots = len(per_bias_cols)
            palette = colors if (colors is not None and len(colors) >= n_plots) else get_palette(n_plots)
            
            # Calculate bar width and positions for grouped bars
            bar_width = 0.8 / n_plots  # Total width of 0.8 divided by number of bias values
            
            # Check if we have iteration column for averaging
            has_iterations = 'iteration' in df_raw.columns
            
            # Collect all histograms data (averaged across iterations)
            all_counts = []
            for i, col in enumerate(per_bias_cols):
                # Filter by sensitive attribute if provided
                if (sensitive_column_name is not None) and (sensitive_attribute_value is not None) and (sensitive_column_name in df_raw.columns):
                    df_filtered = df_raw[df_raw[sensitive_column_name] == sensitive_attribute_value]
                else:
                    df_filtered = df_raw
                
                if has_iterations:
                    # Calculate average counts across iterations
                    # For each iteration, compute histogram, then average
                    iteration_counts = []
                    for iter_num in df_filtered['iteration'].unique():
                        iter_data = df_filtered[df_filtered['iteration'] == iter_num]
                        series = iter_data[col].dropna()
                        
                        # Convert to int levels
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
                    
                    # Average across iterations
                    if iteration_counts:
                        avg_counts = np.mean(iteration_counts, axis=0)
                    else:
                        avg_counts = np.zeros_like(x_values, dtype=float)
                    
                    # Calculate mean UMO across all iterations for vertical line
                    series_all = df_filtered[col].dropna()
                    try:
                        series_all = series_all.astype(int)
                    except Exception:
                        series_all = np.floor(pd.to_numeric(series_all, errors='coerce')).dropna().astype(int)
                    
                else:
                    # No iterations, compute directly
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
            
            # Plot grouped bars
            for i, (counts, series) in enumerate(all_counts):
                label = f"{biases[i]:.1f}" if not np.isnan(biases[i]) else per_bias_cols[i]
                color = palette[i % len(palette)]
                
                # Calculate offset for this bar group
                offset = (i - n_plots / 2 + 0.5) * bar_width
                bar_positions = x_values + offset
                
                # Plot bars
                ax.bar(bar_positions, counts, width=bar_width, label=label, color=color, 
                      alpha=0.8, edgecolor='black', linewidth=0.5)
                
                # Add vertical line for mean
                if not series.empty:
                    mean_umo = series.mean()
                    ax.axvline(mean_umo, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
                  
            ax.set_xlabel('UMO level')
            ylabel = f'Average Count Across {num_iterations} Iterations' if has_iterations else 'Count'
            ax.set_ylabel(ylabel)
            ax.set_title(pollution_mode.replace("_", " ").title())
            ax.set_xticks(x_values)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(title='Bias', fontsize=8)
            
            # Add text to explain the visualization elements
            info_text = 'Dashed lines = mean'
            if has_iterations:
                info_text += f'\nAveraged over {num_iterations} iterations'
            ax.text(0.02, 0.98, info_text,
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        title = f'UMO values count for points in class {sensitive_attribute_value} \nPolluted feature: {feat}'
        if has_iterations and num_iterations:
            title += f'\nAveraged over {num_iterations} iterations'
        title += f'\nFeature weight: {feature_weights[feat]}'
        fig.suptitle(title, fontsize=11, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        plots_dir = os.path.join(output_path, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        out_file = os.path.join(plots_dir, f'{feat}_umo_distributions_by_bias_{num_iterations}_iterations.png')
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved UMO distributions plot for feature: {feat} -> {out_file}")
