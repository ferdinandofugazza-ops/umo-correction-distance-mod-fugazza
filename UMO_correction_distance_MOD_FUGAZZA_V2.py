import os
import re
import time
from itertools import combinations
import random

#import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from plots import make_plots
from new_plots import plot_umo_mean_vs_bias, plot_umo_statistics_summary
from scipy.special import expit


# funzione per calcolare y sul dataset sintetico
def score_computation (df):

    # Coefficienti dei componenti per interpretability
    # Ognuno di questi coefficienti è il peso di ciascun componente nel calcolo dello score
    beta = {"x1": 10, "x2": 1, "x3": 0, "x4": -10}

    comp_x1 = beta["x1"] * df['x1']
    comp_x2 = beta["x2"] * df['x2']
    comp_x3 = beta["x3"] * df['x3']
    comp_x4 = beta["x4"] * df['x4']

    # calcolo del punteggio generale, che si può scomporre e intepretare a livello di singoli componenti
    # lo score non tiene conto della variabile is_white
    score = comp_x1 + comp_x2 + comp_x3 + comp_x4

    # Trasforma i punteggi in probabilità [0, 1] con softmax
    score_probability = expit(score)

    # points with a score greater than 0.5 gets assigned 1, 0 otherwise
    y = (score_probability >= 0.5).astype(int)

    df["y"] = y

    # salva i componenti e i punteggi per ogni sample
    # contributions = pd.DataFrame({
    #    "comp_x1": comp_x1,
    #    "comp_x2": comp_x2,
    #    "comp_x3": comp_x3,
    #    "comp_x4": comp_x4,
    #    "score": score,
    #    "score_probability": score_probability,
    #    "y": y
    #})

    return df, beta

def generate_synthetic_dataset(
    n_samples=20000,
    random_state=0,
):
    rng = np.random.default_rng(random_state)

    x1 = rng.normal(loc=0, scale=1, size=n_samples)
    x2 = rng.normal(loc=0, scale=1, size=n_samples)
    x3 = rng.normal(loc=0, scale=1, size=n_samples)
    x4 = rng.normal(loc=0, scale=1, size=n_samples)

    # we treat this variable as already encoded
    # sensitive feature updated so that it is interchangeable with the other regression pipeline
    dummy_sensitive = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])

    df = pd.DataFrame({
        "x1": x1, "x2": x2, "x3": x3, "x4": x4, "dummy_sensitive": dummy_sensitive
    })

    scored_df, contributions = score_computation(df)

    return scored_df, contributions

def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Trova la soglia ottimale per convertire valori di regressione in classificazione binaria.

    Parameters:
    -----------
    y_true : array-like
        I valori reali (ground truth)
    y_pred_proba : array-like
        I valori predetti dal modello di regressione
    metric : str, default='f1'
        La metrica da ottimizzare ('accuracy', 'f1', 'precision', 'recall')

    Returns:
    --------
    float
        La soglia ottimale
    """

    # Normalizza i valori predetti se non sono già nell'intervallo [0, 1]
    y_pred_proba = (y_pred_proba - min(y_pred_proba)) / (max(y_pred_proba) - min(y_pred_proba))
    y_true_norm = (y_true - min(y_true)) / (max(y_true) - min(y_true))
    y_true_binary = (y_true_norm >= .5).astype(int)

    # Definisci la funzione di scoring in base alla metrica scelta
    if metric == 'accuracy':
        score_func = accuracy_score
    elif metric == 'f1':
        score_func = f1_score
    elif metric == 'precision':
        score_func = precision_score
    elif metric == 'recall':
        score_func = recall_score
    else:
        raise ValueError(f"Metrica '{metric}' non supportata.")

    # Prova diverse soglie e trova quella che massimizza la metrica scelta
    thresholds = np.linspace(0, 1, 100)
    scores = []

    for threshold in thresholds:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
        score = score_func(y_true_binary, y_pred_binary)
        scores.append(score)

    # Trova la soglia con il punteggio più alto
    best_score_idx = np.argmax(scores)
    best_threshold = thresholds[best_score_idx]

    return best_threshold

def train_model(dataset, sensitive_column):
    df = dataset  #.set_index('Unnamed: 0').sort_index()
    X = df.iloc[:, :-1]  # All columns except the last one
    input_features = df.iloc[:, :-1].columns.to_list()
    if sensitive_column != None:
        input_features.remove(sensitive_column)

    y = df.iloc[:, -1]  #last column

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=28)

    reg = XGBRegressor(random_state=0,
                       n_estimators=10,  # Number of boosting rounds
                       max_depth=5,  # Maximum depth of trees
                       objective="reg:squarederror",  # Loss function for regression
                       eval_metric="rmse",  # Evaluation metric
                       verbosity=1  # Controls output logs (set to 0 for silent mode)
                       )

    reg.fit(X_train, y_train)
    y_scores = reg.predict(X_test)
    optimal_threshold = find_optimal_threshold(y_test, y_scores, metric='f1')

    # Converti y_test in binario usando la stessa soglia
    predictions = (y_scores >= optimal_threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    return X_train, X_test, y_train, y_test, reg, input_features, optimal_threshold,

def cutoff_prediction(score, optimal_threshold):
    return (score >= optimal_threshold).astype(int)
    #return score

def train_model_with_threshold_normalization_and_binarization(df, sensitive_column):

    # DA AGGIUNGERE: possibilità di allenare il modello su dati biased
    # e quindi inserire la pipeline di data pollution PRIMA di allenare il modello qui sotto

    X = df.iloc[:, :-1]  # Tutte le colonne tranne l'ultima
    umo_input_features = X.columns.to_list() # All features including sensitive_column at this point

    if sensitive_column is not None and sensitive_column in umo_input_features:
        umo_input_features.remove(sensitive_column) 

    y = df.iloc[:, -1]  # Ultima colonna


    # Split del dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=28)

    model_trained_features = X_train.columns.to_list() # Store features model was trained on

    # Decide se è classificazione binaria o regressione
    # TO ADD: estendere al caso di classificazione multinomiale

    unique_vals = np.unique(y_train)
    is_binary = set(unique_vals).issubset({0, 1})

    model = None
    optimal_threshold = None

    if not is_binary:
        # Regressione: normalizza se i valori escono dall'intervallo [0,1]
        need_normalize = (y_train.min() < 0) or (y_train.max() > 1)

        if need_normalize:
            y_scaler = MinMaxScaler()
            y_train_normalized = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            y_test_normalized = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

            reg = XGBRegressor(random_state=0,
                               n_estimators=10,
                               max_depth=5,
                               objective="reg:squarederror",
                               eval_metric="rmse",
                               verbosity=1)
            reg.fit(X_train, y_train_normalized)
            y_scores_normalized = reg.predict(X_test)

            # soglia calcolata sullo spazio normalizzato
            optimal_threshold = find_optimal_threshold(y_test_normalized, y_scores_normalized, metric='accuracy')
            # keeping model and original y for downstream use
            model = reg
        else:
            reg = XGBRegressor(random_state=0,
                               n_estimators=10,
                               max_depth=5,
                               objective="reg:squarederror",
                               eval_metric="rmse",
                               verbosity=1)
            reg.fit(X_train, y_train)
            y_scores = reg.predict(X_test)
            optimal_threshold = find_optimal_threshold(y_test, y_scores, metric='accuracy')
            model = reg

    else:
        # Classificazione binaria
        clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)
        clf.fit(X_train, y_train)
        # For classification, we typically use a threshold of 0.5 on probabilities,
        # or we could find an optimal one if desired.
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        optimal_threshold = find_optimal_threshold(y_test, y_pred_proba, metric='accuracy') # Calculate optimal threshold for classification too
        model = clf

    return X_train, X_test, y_train, y_test, model, umo_input_features, model_trained_features, optimal_threshold

def calculate_bias_range_by_feat():
    dictionary = {}
    for feat in feature_names:
        dictionary[feat] = []
        dictionary[feat].append(X_test[feat].min())
        dictionary[feat].append(X_test[feat].max())

    return dictionary

def create_array(min):
    """
    Crea un array con attributi sensibili (0 e 1).
    0 è l'attributo sensibile biased, 1 è l'altro attributo.
    """
    n = X_test.shape[0]
    minority_rate = min
    majority_rate = 1 - minority_rate
    # Create array with values 0 (sensitive) and 1 (non-sensitive)
    arr = np.array([0] * int(n * minority_rate) + [1] * int(n * majority_rate) + [np.random.choice([0, 1])])
    np.random.shuffle(arr)  # Shuffle to randomize position
    return arr

def add_noise(row, biased_column, bias, sensitive_attribute, bias_range, sensitive_column, k, pollution_mode, biased_feature_threshold):
    output = row[biased_column] # Initialize output with the original value

    if pollution_mode == 'linear_pollution':
        if row[sensitive_column] == sensitive_attribute:
            output_biased = row[biased_column] * bias
            if output_biased < bias_range[biased_column][0]:
                output = bias_range[biased_column][0]
            elif output_biased > bias_range[biased_column][1]:
                output = bias_range[biased_column][1]
            else:
                output = output_biased

    elif pollution_mode == 'probabilistic_pollution':
        if row[sensitive_column] == sensitive_attribute:
            coin_toss = np.random.choice([1, 0], p=[0.6, 0.4])
            if coin_toss == 1:
                output_biased = row[biased_column] * bias
                if output_biased < bias_range[biased_column][0]:
                    output = bias_range[biased_column][0]
                elif output_biased > bias_range[biased_column][1]:
                    output = bias_range[biased_column][1]
                else:
                    output = output_biased

    elif pollution_mode == 'probabilistic_half_lies':
        if row[sensitive_column] == sensitive_attribute:
            if row[biased_column] >= biased_feature_threshold:
                coin_toss = np.random.choice([1, 0], p=[0.8, 0.2])
                if coin_toss == 1:
                    output_biased = row[biased_column] * bias
                    if output_biased < bias_range[biased_column][0]:
                        output = bias_range[biased_column][0]
                    elif output_biased > bias_range[biased_column][1]:
                        output = bias_range[biased_column][1]
                    else:
                        output = output_biased

    return output

def calculate_UMO(df, umo_input_features):
    start_time = time.time()  # Start timing
    
    df_selected = df[umo_input_features].copy() 
    baseline = df_selected.iloc[:, 0]  # First column as baseline 

    # Differenza minima per considerare uno switch
    switch_threshold = 0.05

    # Confronta ciascuna colonna con la baseline usando la soglia
    diffs = df_selected.subtract(baseline, axis=0).abs() > switch_threshold
    first_diff = diffs.values.argmax(axis=1) - 1
    # Get the first different column index
    # If no deviation, set to None
    first_diff[~diffs.any(axis=1)] = len(df_selected.columns) - 1 # Use df_selected.columns for correct length
    outpt = pd.Series(first_diff, index=df.index)

    end_time = time.time()  # End timing
    duration = end_time - start_time
    # print(f"Execution time: {duration:.4f} seconds")
    return outpt  # Return as a Series with row index

def predict_on_biased_dataset(X_test, sensitive_column_name, sensitive_attribute_value, bias, bias_range, biased_column, k, y_test, feat_to_remove, pollution_mode, mode, reg_model_features, umo_input_features, biased_feature_threshold=0):
    df_list = []
    data = X_test.copy().reset_index(drop=True)

    if sensitive_column_name is None:
        # If no sensitive column was provided, create a dummy one
        sens = pd.read_csv(output_path + f"/sensitive_feature_distr/iteration_{k}.csv").reset_index(drop=True)
        data['dummy_sensitive'] = sens.iloc[:, 1].astype(int)
        sensitive_column_name = 'dummy_sensitive'
        sensitive_attribute_value = 0

    # Apply noise to the biased_column based on the sensitive attribute
    data[biased_column] = data.apply(
        lambda x: add_noise(x, biased_column, bias, sensitive_attribute_value, bias_range, sensitive_column_name, k, pollution_mode, biased_feature_threshold), axis=1)

    if mode == 'baseline':
        if feat_to_remove:
            data[feat_to_remove] = np.nan

    if mode == 'active_learning':
        if feat_to_remove:
            data.loc[data[sensitive_column_name] != sensitive_attribute_value, feat_to_remove] = np.nan

    for i in range(len(umo_input_features) + 1):
        temp_predictions_df = pd.DataFrame(index=data.index)

        for subset_idx, subset in enumerate(combinations(umo_input_features, i)):
            data_copy = data.copy()

            # Evitato di rimuovere la feature che sta venendo inquinata (deve sempre essere presente in prediction)
            if i > 0 and (biased_column in subset):
                continue

            if i == 0:
                pass
            else:
                data_copy[list(subset)] = np.nan

            data_for_prediction = data_copy[reg_model_features].copy()

            if isinstance(reg, XGBClassifier):
                prediction_scores = reg.predict_proba(data_for_prediction)[:, 1]
            else:
                prediction_scores = reg.predict(data_for_prediction)

            prediction_scores = np.nan_to_num(prediction_scores, nan=0.5)

            col_name_for_subset_prediction = f'pred_i{i}_s{subset_idx}'
            # Store continuous predictions (no binarization yet)
            temp_predictions_df[col_name_for_subset_prediction] = prediction_scores

        mean_pred_col_name = f'mean_pred_{i}_removed_ft'
        if not temp_predictions_df.empty:
            mean_predictions = temp_predictions_df.agg("mean", axis="columns")
            mean_predictions = np.nan_to_num(mean_predictions, nan=0.5)
            # Store continuous mean predictions
            df_list.append(pd.DataFrame({mean_pred_col_name: mean_predictions}))

    total = pd.concat(df_list, axis=1, ignore_index=False)

    total['true_class'] = y_test.reset_index(drop=True).astype(int)
    
    cols = [f'mean_pred_{i}_removed_ft' for i in range(len(umo_input_features) + 1) if f'mean_pred_{i}_removed_ft' in total.columns]
    # Calculate UMO on continuous values
    total['UMO'] = calculate_UMO(total, cols)

    # Binarize predictions after UMO calculation
    for col in cols:
        total[col] = cutoff_prediction(total[col], optimal_threshold)

    total[sensitive_column_name] = data[sensitive_column_name]

    return total

def create_sensitive_column(k):
  
    col = create_array(minority_rate)
    pd.DataFrame(col).to_csv(output_path + f"/sensitive_feature_distr/iteration_{k}.csv")
    return col

def calculate_performance(sensitive_column_name, res):
    dictionary = {}
    for attr in res[sensitive_column_name].unique():
        new_df = res[res[sensitive_column_name] == attr]
        y_pred_for_cm = new_df['mean_pred_0_removed_ft']
        y_true_for_cm = new_df['true_class']

        dictionary[attr] = confusion_matrix(y_true_for_cm, y_pred_for_cm, labels=[0, 1]).ravel()

    overall = pd.DataFrame.from_dict(dictionary, orient='index', columns=['tn', 'fp', 'fn', 'tp'])
    return overall

def create_sensitive_columns_dict(m, min):
   
    diz = {}
    for a in range(m):
        diz[a] = create_array(min)
        pd.DataFrame(diz[a]).to_csv(output_path + f"/sensitive_feature_distr/iteration_{a}.csv")
    return

def calculate_UMO_stats(feature, df_raw, performance_list, pollution_folder, sensitive_column_name):
    """
    Compute UMO statistics across collected biased runs and merge with aggregated confusion-matrix counts.
    Saves a single CSV per feature inside pollution_folder containing tn, fp, fn, tp and all UMO stats.
    """

    # Prepare aggregated confusion matrix (sum across biases / iterations)
    if performance_list:
        aggregated_cm = pd.concat(performance_list, axis=0).groupby(level=0).sum()
    else:
        aggregated_cm = pd.DataFrame(columns=['tn', 'fp', 'fn', 'tp'])

    # Prepare UMO values collected across biases (long form)
    if (df_raw is not None) and (not df_raw.empty):
        umo_cols = [c for c in df_raw.columns if str(c).startswith("UMO_") or c == "UMO"]
        # stack to long series and drop NaNs
        umo_series = df_raw[umo_cols].stack().reset_index(drop=True).astype(float)
    else:
        umo_series = pd.Series(dtype=float)

    # Compute statistics on UMO distribution
    if not umo_series.empty:
        umo_mean = float(umo_series.mean())
        umo_std = float(umo_series.std(ddof=0))
        umo_median = float(umo_series.median())
        umo_min = float(umo_series.min())
        umo_max = float(umo_series.max())
        umo_range = float(umo_max - umo_min)
    else:
        umo_mean = umo_std = umo_median = umo_min = umo_max = umo_range = np.nan

    # Merge stats into aggregated_cm. Stats repeated per sensitive attribute row.
    if aggregated_cm.empty:
        summary_df = pd.DataFrame({
            'tn': [],
            'fp': [],
            'fn': [],
            'tp': []
        })
    else:
        summary_df = aggregated_cm.copy()

    # Add UMO global statistics columns (same values for each sensitive-attr row)
    summary_df['umo_mean'] = umo_mean
    summary_df['umo_std'] = umo_std
    summary_df['umo_median'] = umo_median
    summary_df['umo_min'] = umo_min
    summary_df['umo_max'] = umo_max
    summary_df['umo_range'] = umo_range

    # compute per-bias mean UMO grouped by sensitive attribute and insert into summary_df
    if (df_raw is not None) and (not df_raw.empty):
        # Identify columns that correspond to per-bias UMO values
        per_bias_cols = [c for c in df_raw.columns if str(c).startswith("UMO_")]
        if per_bias_cols:
            # group by sensitive attribute and compute mean for each per-bias column
            bias_means = df_raw.groupby(sensitive_column_name)[per_bias_cols].mean()
            
            for col in bias_means.columns:
                # add column with same name as bias column (keeps "UMO_0.4" naming)
                # align by index; if summary_df lacks rows for some attrs, reindex to include them
                if summary_df.empty:
                    summary_df = bias_means.copy()
                else:
                    # assign values per sensitive attr; fill missing with NaN
                    for attr in bias_means.index:
                        if attr not in summary_df.index:
                            # create row for attr with NaNs for existing columns
                            summary_df.loc[attr] = [ np.nan ] * summary_df.shape[1]
                    # now assign column values
                    summary_df[col] = bias_means[col]

    # save raw UMO dataframe for plotting
    try:
        if (df_raw is not None) and (not df_raw.empty):
            raw_outpath = os.path.join(pollution_folder, f"{feature}_umo_raw.csv")
            # save UMO_* columns plus sensitive column and prediction correctness columns
            cols_to_save = [c for c in df_raw.columns if str(c).startswith("UMO_")]
            if sensitive_column_name in df_raw.columns:
                cols_to_save = [sensitive_column_name] + cols_to_save
            # add prediction and true_class columns for filtering
            if 'mean_pred_0_removed_ft' in df_raw.columns:
                cols_to_save.append('mean_pred_0_removed_ft')
            if 'true_class' in df_raw.columns:
                cols_to_save.append('true_class')
            # add iteration column for averaging across iterations
            if 'iteration' in df_raw.columns:
                cols_to_save.append('iteration')
            # fallback: if no UMO_* columns, save entire df_raw
            if not cols_to_save:
                df_raw.to_csv(raw_outpath, index=False)
            else:
                df_raw[cols_to_save].to_csv(raw_outpath, index=False)
    except Exception as e:
        print(f"Warning: could not save raw UMO CSV for feature '{feature}' in '{pollution_folder}': {e}")

    # Save the combined summary CSV for this feature inside the pollution folder
    outpath = os.path.join(pollution_folder, f"{feature}_summary.csv")
    summary_df.to_csv(outpath)

    return summary_df

def iterate_process(X_test, y_test, feat_to_remove, sensitive_column_name, sensitive_attribute_value, num_iterations, minority, reg_model_features, umo_input_features):
    features_bias_ranges = calculate_bias_range_by_feat()
    pollution_modes = ['linear_pollution', 'probabilistic_pollution', 'probabilistic_half_lies']

    if sensitive_column_name is None:
        create_sensitive_columns_dict(num_iterations, minority)
        sensitive_column_name = 'dummy_sensitive'

    # Create pollution_mode subfolders
    for pollution_mode in pollution_modes:
        pollution_folder = output_path + f'/{pollution_mode}'
        if not os.path.isdir(pollution_folder):
            os.makedirs(pollution_folder)

    for feat in feature_names:
        
        for pollution_mode in pollution_modes:
            # Create lists for this specific pollution_mode
            lista_tp_baseline = []
            lista_tn_baseline = []
            lista_fp_baseline = []
            lista_fn_baseline = []
            lista_all_baseline = []
            performance_list_baseline = []
            raw_res_baseline = []  # collect raw results before further grouping (for stats)

            for k in range(num_iterations):
                for bias in np.arange(0.4, 1.6, 0.2):
                    bias = round(bias, 1)
                    res = predict_on_biased_dataset(X_test.copy(), sensitive_column_name, sensitive_attribute_value, bias, features_bias_ranges, feat, k, y_test, feat_to_remove, pollution_mode, mode='baseline', reg_model_features=reg_model_features, umo_input_features=umo_input_features)
                    res['feature'] = feat
                    res['iteration'] = k  # Add iteration number

                    # keep a copy of the raw result before renaming/grouping for stats aggregation
                    res_for_stats = res.copy()
                    if 'UMO' in res_for_stats.columns:
                        res_for_stats = res_for_stats.rename(columns={"UMO": f"UMO_{bias}"})
                    raw_res_baseline.append(res_for_stats)

                    res = res.rename(columns={"UMO": f"UMO_{bias}"})
                    performance_list_baseline.append(calculate_performance(sensitive_column_name, res))
                    zz = res.groupby([sensitive_column_name]).agg({
                        f"UMO_{bias}": "mean",
                        sensitive_column_name: "count"
                    }).rename(columns={sensitive_column_name:f"count_{bias}"})

                    lista_all_baseline.append(zz)
                    res = res.groupby([sensitive_column_name, 'mean_pred_0_removed_ft', 'true_class']).agg({
                        f"UMO_{bias}": "mean",
                        sensitive_column_name: "count"
                    }).rename(columns={sensitive_column_name:f"count_{bias}"}).reset_index()
                    tp = res[(res['mean_pred_0_removed_ft'] == 1) & (res['true_class'] == 1)][
                                        [sensitive_column_name, f"UMO_{bias}", f"count_{bias}"]].set_index(sensitive_column_name)
                    tn = res[(res['mean_pred_0_removed_ft'] == 0) & (res['true_class'] == 0)][
                        [sensitive_column_name, f"UMO_{bias}", f"count_{bias}"]].set_index(sensitive_column_name)
                    fp = res[(res['mean_pred_0_removed_ft'] == 1) & (res['true_class'] == 0)][
                                        [sensitive_column_name, f"UMO_{bias}", f"count_{bias}"]].set_index(sensitive_column_name)
                    fn = res[(res['mean_pred_0_removed_ft'] == 0) & (res['true_class'] == 1)][
                                        [sensitive_column_name, f"UMO_{bias}", f"count_{bias}"]].set_index(sensitive_column_name)
                    lista_tp_baseline.append(tp)
                    lista_tn_baseline.append(tn)
                    lista_fp_baseline.append(fp)
                    lista_fn_baseline.append(fn)

            pollution_folder = output_path + f'/{pollution_mode}'

            # usa la funzione di riepilogo che scrive UN SOLO CSV per feature (confusion matrix aggregata + statistiche UMO)
            if raw_res_baseline:
                df_raw = pd.concat(raw_res_baseline, axis=0, ignore_index=True)
            else:
                df_raw = pd.DataFrame()
            calculate_UMO_stats(feat, df_raw, performance_list_baseline, pollution_folder, sensitive_column_name)

    return

# for dataset_name in ['winequality-white']:
#     for minority_rate in [0.5, 0.9]:
#         sensitive_column = None
#         sensitive_attribute = None
#         output_path = "../data/output/" + str(minority_rate) + '/' + dataset_name
#         df = pd.read_csv("../data/" + dataset_name + ".csv")
#         # # # df = df[df['race'].isin([0,1])]
#         X_train, X_test, y_train, y_test, reg, input_features, optimal_threshold = train_model_with_threshold_normalization_and_binarization(df, sensitive_column)
#         feature_names = input_features
#         iterate_process(X_test, y_test, sensitive_column=sensitive_column, sensitive_attribute=sensitive_attribute, num_iterations=5, minority=minority_rate)
#         make_plots('mean', output_path, feature_names)
#         #make_plots('median', output_path, feature_names)

for minority_rate in [0.5]:#np.arange(0.5, 1, 0.1):

    pipeline_mode = 'classification'

    if pipeline_mode == 'classification':

        sensitive_column_name = 'dummy_sensitive'
        sensitive_attribute_value = 0

        df, feature_weights = generate_synthetic_dataset(n_samples=10000, random_state=42)
        X_train, X_test, y_train, y_test, cls, umo_input_features, model_trained_features, optimal_threshold = train_model_with_threshold_normalization_and_binarization(df, sensitive_column_name)
        feat_to_remove = [] # Assuming feat_to_remove is defined globally or passed as an argument

        feature_names = [x for x in umo_input_features if (x not in feat_to_remove)] # feature_names for iterate_process
        output_path = "data/output/" + "synthetic_dataset_" + str(minority_rate) # Define output_path
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        subfolder_1 = output_path + '/plots'
        if not os.path.isdir(subfolder_1):
            os.makedirs(subfolder_1)

        subfolder_2 = output_path + '/sensitive_feature_distr'
        if not os.path.isdir(subfolder_2):
            os.makedirs(subfolder_2)

        reg = cls # Assign the trained model to 'reg'
        reg_model_features = model_trained_features # Get the actual feature names the model was trained with
        num_iterations = 1

        iterate_process(X_test, y_test, feat_to_remove, sensitive_column_name=sensitive_column_name, sensitive_attribute_value=sensitive_attribute_value, num_iterations=num_iterations, minority=minority_rate, reg_model_features=reg_model_features, umo_input_features=umo_input_features)
        # mode_list = ['baseline']
        # make_plots('mean', output_path, feature_names, mode_list=mode_list, sensitive_column_name=sensitive_column_name, sensitive_attribute_value=sensitive_attribute_value)
        plot_umo_mean_vs_bias(output_path, feature_names, feature_weights, sensitive_column_name=sensitive_column_name, sensitive_attribute_value=sensitive_attribute_value, num_iterations=num_iterations)
        plot_umo_statistics_summary(output_path, feature_names, feature_weights, sensitive_column_name=sensitive_column_name, sensitive_attribute_value=sensitive_attribute_value, num_iterations=num_iterations)

    else:
        for dataset_name in ['winequality-red']:

        # create output folder and 2 subfolders: plots, sensitive_feature_distr
            output_path = "data/output/" + dataset_name + str(minority_rate)
            mypath = output_path
            if not os.path.isdir(mypath):
                os.makedirs(mypath)

            subfolder_1 = output_path + '/plots'
            if not os.path.isdir(subfolder_1):
                os.makedirs(subfolder_1)

            subfolder_2 = output_path + '/sensitive_feature_distr'
            if not os.path.isdir(subfolder_2):
                os.makedirs(subfolder_2)

            sensitive_column_name = None
            sensitive_attribute_value = None
            df = pd.read_csv("data/" + dataset_name + ".csv")

            X_train, X_test, y_train, y_test, reg, umo_input_features, model_trained_features, optimal_threshold = train_model_with_threshold_normalization_and_binarization(df, sensitive_column_name)
            feat_to_remove = ['fixed acidity']

            feature_names = [x for x in umo_input_features if (x not in feat_to_remove)]
            reg_model_features = model_trained_features
            iterate_process(X_test, y_test, feat_to_remove, sensitive_column_name=sensitive_column_name, sensitive_attribute_value=sensitive_attribute_value, num_iterations=1, minority=minority_rate, reg_model_features=reg_model_features, umo_input_features=umo_input_features)
            mode_list = ['baseline']
            
            # make_plots('mean', output_path, feature_names, mode_list=mode_list, sensitive_column_name=sensitive_column_name, sensitive_attribute_value=sensitive_attribute_value)
            plot_umo_mean_vs_bias(output_path, feature_names, sensitive_column_name=sensitive_column_name, sensitive_attribute_value=sensitive_attribute_value)
