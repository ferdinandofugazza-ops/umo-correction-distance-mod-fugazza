import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import os


def make_plots(metric, output_path, feature_names, mode_list, sensitive_column_name=None, sensitive_attribute_value=None, noise_type_list=None):

    # TO ADD: altre visualizzazioni in cui si visualizzi anche il numero assoluto di TN, FP, FN, TP per ciascun gruppo
    # altre visualizzazioni con metriche ≠ media 
    
    # Default noise types if not provided
    if noise_type_list is None:
        noise_type_list = ['default']
    
    # determina le colonne effettive nei CSV in base all'attributo sensibile
    # se sensitive_attribute_value è 0, allora col_A=0 (biased) e col_B=1 (unbiased)
    # se sensitive_attribute_value è 1, allora col_A=1 (biased) e col_B=0 (unbiased)
    if sensitive_attribute_value is not None and sensitive_attribute_value in [0, 1]:
        col_A = str(sensitive_attribute_value)
        col_B = str(1 - sensitive_attribute_value)
    else:
        # Default: prova 'A' e 'B'
        col_A = 'A'
        col_B = 'B'

    for feat in feature_names:
        dictionary = {
            "baseline": [],
            "active_learning": []
        }

        for noise_type in noise_type_list:
            for mode in mode_list:
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                i = 0
                
                for group in ['tn', 'fp', 'fn', 'tp']:

                    row, col = divmod(i, 2)
                    
                    # Modifica il percorso per includere il tipo di rumore
                    csv_path = output_path + f'/{feat}_{group}_{mode}_{noise_type}.csv'
                    
                    # Controlla se il file esiste
                    if not os.path.exists(csv_path):
                        i = i + 1
                        continue

                    try:
                        aa = pd.read_csv(csv_path)
                    except Exception as e:
                        print(f"Errore nel leggere {csv_path}: {e}")
                        i = i + 1
                        continue

                    # Estrai righe UMO
                    df_UMO = aa[aa['Unnamed: 0'].str.contains("UMO", na=False)].copy()
                    if df_UMO.empty:
                        i = i + 1
                        continue

                    df_UMO['Unnamed: 0'] = df_UMO['Unnamed: 0'].str.replace("UMO_", "", regex=False)
                    df_UMO['Unnamed: 0'] = pd.to_numeric(df_UMO['Unnamed: 0'], errors='coerce')
                    
                    # Usa le colonne mappate (col_A e col_B) per accedere ai dati
                    if col_A in df_UMO.columns:
                        df_UMO['A'] = pd.to_numeric(df_UMO[col_A], errors='coerce')
                    else:
                        df_UMO['A'] = pd.Series(dtype=float)
                    
                    if col_B in df_UMO.columns:
                        df_UMO['B'] = pd.to_numeric(df_UMO[col_B], errors='coerce')
                    else:
                        df_UMO['B'] = pd.Series(dtype=float)

                    # Estrai righe count
                    df_count = aa[aa['Unnamed: 0'].str.contains("count", na=False)].copy()
                    if not df_count.empty:
                        df_count['Unnamed: 0'] = df_count['Unnamed: 0'].str.replace("count_", "", regex=False)
                        df_count['Unnamed: 0'] = pd.to_numeric(df_count['Unnamed: 0'], errors='coerce')
                        
                        if col_A in df_count.columns:
                            df_count['A'] = pd.to_numeric(df_count[col_A], errors='coerce')
                        else:
                            df_count['A'] = pd.Series(dtype=float)
                        
                        if col_B in df_count.columns:
                            df_count['B'] = pd.to_numeric(df_count[col_B], errors='coerce')
                        else:
                            df_count['B'] = pd.Series(dtype=float)
                        
                        try:
                            dictionary[mode].append(df_count.groupby(['Unnamed: 0']).agg({'A': metric,
                                                                           'B': metric})
                                             .rename(columns={'A': f'{metric}_{group}_A',
                                                              'B': f'{metric}_{group}_B'}))
                        except Exception:
                            pass

                    ax = axes[row, col]
                    
                    # Plot solo se ci sono dati 
                    if not df_UMO['A'].isna().all():
                        sns.lineplot(df_UMO, x='Unnamed: 0', y='A', label=f'A UMO {group}', ax=ax, estimator=metric)
                    if not df_UMO['B'].isna().all():
                        sns.lineplot(df_UMO, x='Unnamed: 0', y='B', label=f'B UMO {group}', ax=ax, estimator=metric)
                    
                    ax.grid(True)
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    ax.set_title(f'{group.upper()}', fontsize=12, fontweight='bold')

                    i = i + 1

                fig.suptitle(f'{mode} - {feat} ({noise_type})', fontsize=16)
                fig.supxlabel(f'noise coefficient for A individuals on {feat}', fontsize=14)
                fig.supylabel(f' {metric} UMO level', fontsize=14)
                fig.tight_layout(rect=[0, 0.05, 1, 0.95])
                plt.savefig(output_path + f'/plots/{feat}_{mode}_{metric}_{noise_type}.png')
                plt.close(fig)

    return



