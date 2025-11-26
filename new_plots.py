import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_umo_mean_vs_bias(output_path, feature_names, feature_weights, sensitive_column_name=None, sensitive_attribute_value=None, num_iterations=None):
    """
    Plot mean UMO level vs bias coefficient for each feature across all pollution modes.
    
    Parameters:
    -----------
    output_path : str
        Path to the output directory containing pollution mode subdirectories
    feature_names : list
        List of feature names to plot
    sensitive_column_name : str, optional
        Name of the sensitive column
    sensitive_attribute_value : int, optional
        Value of the sensitive attribute (0 or 1)
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
            summary_path = os.path.join(output_path, pollution_mode, f'{feat}_summary.csv')
            
            if not os.path.exists(summary_path):
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{pollution_mode.replace("_", " ").title()}')
                continue
            
            summary_df = pd.read_csv(summary_path, index_col=0)
            
            # Find columns corresponding to per-bias UMO means (e.g. "UMO_0.4")
            per_bias_cols = [c for c in summary_df.columns if str(c).startswith("UMO_")]
            if not per_bias_cols:
                ax.text(0.5, 0.5, 'No per-bias UMO columns', ha='center', va='center', transform=ax.transAxes, fontsize=11)
                ax.set_title(f'{pollution_mode.replace("_", " ").title()}')
                continue
            
            # sort per_bias_cols by numeric bias extracted from the column name
            def bias_from_col(c):
                try:
                    return float(c.split('_', 1)[1])
                except Exception:
                    return float('nan')
            per_bias_cols = sorted(per_bias_cols, key=bias_from_col)
            biases_found = [bias_from_col(c) for c in per_bias_cols]
            
            # For each sensitive attribute value (index), plot mean UMO across bias columns
            for attr in summary_df.index:
                # extract values (may contain NaNs if some biases missing)
                y = summary_df.loc[attr, per_bias_cols].values.astype(float)
                # Align x: use biases_found (parsed from column names)
                ax.plot(biases_found, y, marker='o', label=f'Attr={attr}', linewidth=2, markersize=6)
            
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
                ax.set_ylabel('Count')
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

            for i, col in enumerate(per_bias_cols):
                # If a sensitive column and target sensitive attribute value are provided,
                # restrict the series to only those rows that get polluted (e.g. sensitive_attribute_value)
                # avoids plotting distributions for the non-polluted class whose UMO stays constant
                if (sensitive_column_name is not None) and (sensitive_attribute_value is not None) and (sensitive_column_name in df_raw.columns):
                    series = df_raw.loc[df_raw[sensitive_column_name] == sensitive_attribute_value, col].dropna()
                else:
                    series = df_raw[col].dropna()
                 # try to convert to int levels; if not possible, floor to int
                try:
                     series = series.astype(int)
                except Exception:
                     series = np.floor(pd.to_numeric(series, errors='coerce')).dropna().astype(int)

                if series.empty:
                    counts = np.zeros_like(x_values)
                else:
                    vc = series.value_counts().sort_index()
                    counts = np.array([int(vc.get(x, 0)) for x in x_values])

                y = counts.astype(float)
                
                label = f"{biases[i]:.1f}" if not np.isnan(biases[i]) else col
                color = palette[i % len(palette)]

                
                ax.step(x_values, y, where='mid', label=label, color=color, linewidth=2)
                
                # Calculate mean and std dev for the UMO levels
                if not series.empty:
                    mean_umo = series.mean()
                    std_umo = series.std()
                    
                    # Add vertical line for mean
                    ax.axvline(mean_umo, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
                  
            ax.set_xlabel('UMO level')
            ax.set_ylabel('Count')
            ax.set_title(pollution_mode.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
            ax.legend(title='Bias', fontsize=8)
            
            # Add text to explain the visualization elements
            ax.text(0.02, 0.98, 'Dashed lines = mean',
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        fig.suptitle(f'UMO values count for points in class {sensitive_attribute_value} \nPolluted feature: {feat}\nIterations: {num_iterations}\nFeature weight: {feature_weights[feat]}', fontsize=11, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        plots_dir = os.path.join(output_path, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        out_file = os.path.join(plots_dir, f'{feat}_umo_distributions_by_bias_{num_iterations}_iterations.png')
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved UMO distributions plot for feature: {feat} -> {out_file}")
