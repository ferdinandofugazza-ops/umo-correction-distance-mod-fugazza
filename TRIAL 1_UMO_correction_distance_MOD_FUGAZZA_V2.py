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
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from plots import make_plots
from new_plots import plot_umo_mean_vs_bias, plot_umo_statistics_summary, plot_accuracy_by_group, plot_umo_disparity_gap
from scipy.special import expit
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# funzione per calcolare y sul dataset sintetico
def score_computation (df):

    # Coefficienti dei componenti per interpretability
    # Ognuno di questi coefficienti è il peso di ciascun componente nel calcolo dello score
    beta = {"x1": 3.5, "x2": 1.5, "x3": 0.1, "x4": -3.5}

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

    x1 = rng.normal(loc=0, scale=4, size=n_samples)
    x2 = rng.normal(loc=0, scale=1, size=n_samples)
    x3 = rng.normal(loc=0, scale=1, size=n_samples)
    x4 = rng.normal(loc=0, scale=3.5, size=n_samples)

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

def train_model_with_threshold_normalization_and_binarization(df, sensitive_column, learner):

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
        if learner not in ['DecisionTree', 'KNN']:
            # Classificazione binaria
            clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)
            clf.fit(X_train, y_train)
            # For classification, we typically use a threshold of 0.5 on probabilities,
            # or we could find an optimal one if desired.
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            optimal_threshold = find_optimal_threshold(y_test, y_pred_proba, metric='accuracy') # Calculate optimal threshold for classification too
            model = clf

        elif learner == 'KNN':
            clf = KNeighborsClassifier()
            clf.fit(X_train, y_train)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            optimal_threshold = find_optimal_threshold(y_test, y_pred_proba, metric='accuracy')
            model = clf

        elif learner == 'DecisionTree': 
            clf = DecisionTreeClassifier(random_state=0)
            clf.fit(X_train, y_train)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            optimal_threshold = find_optimal_threshold(y_test, y_pred_proba, metric='accuracy')
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
            coin_toss = np.random.choice([1, 0], p=[0.8, 0.2])
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

def calculate_UMO_continuous(df, umo_input_features):
    start_time = time.time()  # Start timing
    
    df_selected = df[umo_input_features].copy() 
    baseline = df_selected.iloc[:, 0]  # First column as baseline 

    # Differenza minima per considerare uno switch
    switch_threshold = float(np.std(df_selected, axis=1).mean()) # moving one std from baseline is an approximation of a prediction switch
    #switch_threshold = 0.05

    # Confronta ciascuna colonna con la baseline usando la soglia
    diffs = df_selected.subtract(baseline, axis=0).abs() > switch_threshold
    first_diff = diffs.values.argmax(axis=1) 
    # Get the first different column index
    # If no deviation, set to None
    first_diff[~diffs.any(axis=1)] = len(df_selected.columns) - 1 # Use df_selected.columns for correct length
    outpt = pd.Series(first_diff, index=df.index)

    end_time = time.time()  # End timing
    duration = end_time - start_time
    # print(f"Execution time: {duration:.4f} seconds")
    return outpt  # Return as a Series with row index

def calculate_UMO_discrete(df, umo_input_features):
    start_time = time.time()  # Start timing
    
    df_selected = df[umo_input_features].copy() 
    baseline = df_selected.iloc[:, 0]  # First column as baseline 
    diffs = df_selected.ne(baseline, axis=0)  # Compare each column with the baseline
    first_diff = diffs.values.argmax(axis=1) 
    # Get the first different column index
    # If no deviation, set to None
    first_diff[~diffs.any(axis=1)] = len(df_selected.columns) - 1 # Use df_selected.columns for correct length
    outpt = pd.Series(first_diff, index=df.index)

    end_time = time.time()  # End timing
    duration = end_time - start_time
    # print(f"Execution time: {duration:.4f} seconds")
    return outpt  # Return as a Series with row index

def predict_on_biased_dataset(X_test, sensitive_column_name, sensitive_attribute_value, bias, bias_range, biased_column, k, y_test, feat_to_remove, pollution_mode, UMO_mode, mode, reg_model_features, umo_input_features, biased_feature_threshold=0):
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

    if UMO_mode == 'continuous': # compute UMO over continuous predictions

        for i in range(len(umo_input_features) + 1):
            temp_predictions_df = pd.DataFrame(index=data.index)

            for subset_idx, subset in enumerate(combinations(umo_input_features, i)):
                data_copy = data.copy()

                # Evita di rimuovere la feature che sta venendo inquinata (deve sempre essere presente in prediction)
                if i > 0 and (biased_column in subset):
                    continue

                if i == 0:
                    pass
                else:
                    data_copy[list(subset)] = np.nan

                data_for_prediction = data_copy[reg_model_features].copy()

                if isinstance(reg, XGBClassifier):
                    prediction_scores = reg.predict_proba(data_for_prediction)[:, 1]
                elif isinstance(reg, DecisionTreeClassifier):
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
        total['UMO'] = calculate_UMO_continuous(total, cols)

        # Binarize predictions AFTER UMO calculation
        for col in cols:
            total[col] = cutoff_prediction(total[col], optimal_threshold)

        total[sensitive_column_name] = data[sensitive_column_name]

        return total
            
    elif UMO_mode == 'discrete': # compute UMO over binarized predictions

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
                temp_predictions_df[col_name_for_subset_prediction] = cutoff_prediction(prediction_scores, optimal_threshold)

            mean_pred_col_name = f'mean_pred_{i}_removed_ft'
            if not temp_predictions_df.empty:
                mean_predictions = temp_predictions_df.agg("mean", axis="columns")
                mean_predictions = np.nan_to_num(mean_predictions, nan=0.5)
                df_list.append(pd.DataFrame({mean_pred_col_name: cutoff_prediction(mean_predictions, optimal_threshold)}))

        total = pd.concat(df_list, axis=1, ignore_index=False)

        total['true_class'] = y_test.reset_index(drop=True).astype(int)
            
        cols = [f'mean_pred_{i}_removed_ft' for i in range(len(umo_input_features) + 1) if f'mean_pred_{i}_removed_ft' in total.columns]
        total['UMO'] = calculate_UMO_discrete(total, cols)

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

def plot_probability_contours(model, feature_names, X_train, X_test=None, y_train=None, y_test=None, optimal_threshold=None, output_path=None, learner='model', grid_points=100, figsize=(12, 10)):

    # Register contour plot directory
    contour_dir = os.path.join(output_path, 'probability_contours') if output_path is not None else os.path.join('plots', 'probability_contours')
    if not os.path.isdir(contour_dir):
        os.makedirs(contour_dir)

    # Convert to DataFrame if necessary
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train, columns=feature_names)
    if X_test is not None and isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    # Prepare data for PCA (fit on combined train+test if available for consistent projection)
    try:
        pca = PCA(n_components=2)
        if X_test is not None:
            combined_df = pd.concat([X_train[feature_names], X_test[feature_names]], ignore_index=True)
            X_pca_all = pca.fit_transform(combined_df.values)
            n_train = X_train.shape[0]
            X_train_pca = X_pca_all[:n_train]
            X_test_pca = X_pca_all[n_train:]
            X_pca_for_ranges = X_pca_all
        else:
            X_feat = X_train[feature_names].values
            X_train_pca = pca.fit_transform(X_feat)
            X_test_pca = None
            X_pca_for_ranges = X_train_pca

        # Determine ranges in PCA space
        x1_min, x1_max = X_pca_for_ranges[:, 0].min(), X_pca_for_ranges[:, 0].max()
        x2_min, x2_max = X_pca_for_ranges[:, 1].min(), X_pca_for_ranges[:, 1].max()

        # Create grid in PCA space
        x1_range = np.linspace(x1_min, x1_max, grid_points)
        x2_range = np.linspace(x2_min, x2_max, grid_points)
        X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)

        grid_points_combined = np.vstack([X1_grid.ravel(), X2_grid.ravel()]).T

        # Inverse transform grid points back to original feature space for prediction
        grid_original = pca.inverse_transform(grid_points_combined)
        grid_df = pd.DataFrame(grid_original, columns=feature_names)

        # Ensure column order matches model training features
        grid_df = grid_df[feature_names]

        # Get model predictions for grid, train and test
        try:
            if hasattr(model, 'predict_proba'):
                grid_preds = model.predict_proba(grid_df)[:, 1]
                train_preds = model.predict_proba(X_train[feature_names])[:, 1]
                test_preds = model.predict_proba(X_test[feature_names])[:, 1] 
            else:
                grid_preds = model.predict(grid_df)
                train_preds = model.predict(X_train[feature_names])
                test_preds = model.predict(X_test[feature_names]) 

                # Normalize regression outputs to [0,1] so maps and colors are comparable
                scaler = MinMaxScaler()
                combined_preds = grid_preds.reshape(-1, 1)
                combined_preds = np.vstack([combined_preds, train_preds.reshape(-1, 1)])
                if test_preds is not None:
                    combined_preds = np.vstack([combined_preds, test_preds.reshape(-1, 1)])
                scaler.fit(combined_preds)
                grid_preds = scaler.transform(grid_preds.reshape(-1, 1)).flatten()
                train_preds = scaler.transform(train_preds.reshape(-1, 1)).flatten()
                if test_preds is not None:
                    test_preds = scaler.transform(test_preds.reshape(-1, 1)).flatten()

            # Reshape grid predictions to grid
            Z = grid_preds.reshape(X1_grid.shape)

            # Plot in PCA-space (PC1 vs PC2)
            fig, ax = plt.subplots(figsize=figsize)
            contour = ax.contourf(X1_grid, X2_grid, Z, levels=20, cmap='RdYlBu_r', alpha=0.8)
            contour_lines = ax.contour(X1_grid, X2_grid, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
            ax.clabel(contour_lines, inline=True, fontsize=8)

            # Add colorbar (shared for contour and scatter)
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label('Predicted Probability', fontsize=12)

            # Overlay training and test points (projected into PCA space)
            ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_preds, cmap='RdYlBu_r', s=30, edgecolors='k', linewidths=0.4, alpha=0.9)
            ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_preds, cmap='RdYlBu_r', s=30, edgecolors='k', linewidths=0.4, alpha=0.90)

            # Add decision boundary at 0.5 probability when meaningful
            try:
                ax.contour(X1_grid, X2_grid, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
            except Exception:
                pass

            # Labels and title
            ax.set_xlabel('PC1', fontsize=12)
            ax.set_ylabel('PC2', fontsize=12)
            ax.set_title(f'Probability Contour Plot: PC1 vs PC2 (PCA-reduced) - model: {learner}', fontsize=14, fontweight='bold')

            # Legend
            ax.legend(loc='upper right')

            # Compute and display performance metrics (use F1 and accuracy) if true labels are provided
            metrics_text = ''
            if y_train is not None:
                # Binarize predictions using provided optimal_threshold (fallback 0.5)
                train_pred_bin = (np.array(train_preds) >= optimal_threshold).astype(int)
            y_train_arr = np.array(y_train).astype(int)
            train_acc = accuracy_score(y_train_arr, train_pred_bin)
            train_f1 = f1_score(y_train_arr, train_pred_bin, zero_division=0)
            metrics_text += f'Train — Acc: {train_acc:.3f}, F1: {train_f1:.3f}\n'

            if (y_test is not None) and (X_test is not None):
                test_pred_bin = (np.array(test_preds) >= optimal_threshold).astype(int)
            y_test_arr = np.array(y_test).astype(int)
            test_acc = accuracy_score(y_test_arr, test_pred_bin)
            test_f1 = f1_score(y_test_arr, test_pred_bin, zero_division=0)
            metrics_text += f'Test  — Acc: {test_acc:.3f}, F1: {test_f1:.3f}'

            if metrics_text:
                # place a semi-transparent text box in upper-left
                ax.text(0.01, 0.99, metrics_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Save figure
            plot_filename = os.path.join(contour_dir, f'contour_pca_pc1_pc2_{learner}.png')
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved PCA contour plot: {plot_filename}")

        except Exception as e:
            print(f"Error computing predictions for contour grid or points: {e}")

    except Exception as e:
        print(f"Error fitting PCA or preparing grid for contour plotting: {e}")
        
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
        umo_median = float(umo_series.median())
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
    summary_df['umo_median'] = umo_median

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

def compute_accuracy_old(results_list, sensitive_column_name):
    """
    Compute accuracy and F1 score for each iteration.
    Returns a dictionary with iteration -> (accuracy, f1_score)
    """
    iteration_metrics = {}
    
    for result_df in results_list:
        if result_df.empty:
            continue
        
        # Extract iteration number if available
        iteration = result_df['iteration'].iloc[0] if 'iteration' in result_df.columns else 0
        
        # Get predictions and true labels
        y_pred = result_df['mean_pred_0_removed_ft'].values
        y_true = result_df['true_class'].values
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if iteration not in iteration_metrics:
            iteration_metrics[iteration] = {'accuracy': [], 'f1': []}
        
        iteration_metrics[iteration]['accuracy'].append(accuracy)
        iteration_metrics[iteration]['f1'].append(f1)
    
    # Average across biases for each iteration
    for iteration in iteration_metrics:
        iteration_metrics[iteration]['accuracy'] = np.mean(iteration_metrics[iteration]['accuracy'])
        iteration_metrics[iteration]['f1'] = np.mean(iteration_metrics[iteration]['f1'])
    
    return iteration_metrics

def compute_accuracy(results_list, sensitive_column_name):
    """
    Compute accuracy and F1 score for each iteration, both overall and per sensitive group.
    Returns a dictionary with iteration -> {
        'overall':   {'accuracy': float, 'f1': float},
        group_value: {'accuracy': float, 'f1': float, 'tpr': float, 'fpr': float},
        ...
    }
    """
    iteration_metrics = {}

    for result_df in results_list:
        if result_df.empty:
            continue

        iteration = result_df['iteration'].iloc[0] if 'iteration' in result_df.columns else 0

        y_pred = result_df['mean_pred_0_removed_ft'].values
        y_true = result_df['true_class'].values

        if iteration not in iteration_metrics:
            iteration_metrics[iteration] = {}
            iteration_metrics[iteration]['overall'] = {'accuracy': [], 'f1': []}

        # overall metrics (unchanged behaviour)
        iteration_metrics[iteration]['overall']['accuracy'].append(accuracy_score(y_true, y_pred))
        iteration_metrics[iteration]['overall']['f1'].append(f1_score(y_true, y_pred, zero_division=0))

        # per-group metrics
        if sensitive_column_name in result_df.columns:
            for group_val in result_df[sensitive_column_name].unique():
                mask = result_df[sensitive_column_name] == group_val
                y_pred_g = result_df.loc[mask, 'mean_pred_0_removed_ft'].values
                y_true_g = result_df.loc[mask, 'true_class'].values

                if group_val not in iteration_metrics[iteration]:
                    iteration_metrics[iteration][group_val] = {
                        'accuracy': [], 'f1': [], 'tpr': [], 'fpr': []
                    }

                iteration_metrics[iteration][group_val]['accuracy'].append(
                    accuracy_score(y_true_g, y_pred_g)
                )
                iteration_metrics[iteration][group_val]['f1'].append(
                    f1_score(y_true_g, y_pred_g, zero_division=0)
                )

                # TPR (equal opportunity) and FPR (equalized odds components)
                tn, fp, fn, tp = confusion_matrix(y_true_g, y_pred_g, labels=[0, 1]).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # recall / sensitivity
                fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan  # fall-out

                iteration_metrics[iteration][group_val]['tpr'].append(tpr)
                iteration_metrics[iteration][group_val]['fpr'].append(fpr)

    # average across bias levels for each iteration
    for iteration in iteration_metrics:
        for group_key in iteration_metrics[iteration]:
            for metric in iteration_metrics[iteration][group_key]:
                values = iteration_metrics[iteration][group_key][metric]
                iteration_metrics[iteration][group_key][metric] = np.nanmean(values)

    return iteration_metrics

def save_accuracy_csv_old(metrics_by_feature_pollution, output_path, feature_names, pollution_modes, num_iterations):
    """
    Aggregate and save accuracy and F1 scores by iteration to CSV files.
    Creates one CSV per pollution mode with columns for each feature's accuracy and F1.
    """
    for pollution_mode in pollution_modes:
        # Initialize dataframe columns for this pollution mode
        iteration_data = {}
        
        for iteration in range(num_iterations):
            for feat in feature_names:
                key = (feat, pollution_mode)
                if key in metrics_by_feature_pollution:
                    metrics = metrics_by_feature_pollution[key]
                    if iteration in metrics:
                        acc = metrics[iteration]['accuracy']
                        f1 = metrics[iteration]['f1']
                        
                        # Create column names with feature and metric type
                        acc_col = f"{feat}_iter{iteration}_accuracy"
                        f1_col = f"{feat}_iter{iteration}_f1"
                        
                        # Initialize row 0 if not yet done
                        if 0 not in iteration_data:
                            iteration_data[0] = {}
                        
                        iteration_data[0][acc_col] = acc
                        iteration_data[0][f1_col] = f1
        
        # Create and save dataframe
        if iteration_data:
            df_metrics = pd.DataFrame(iteration_data).T
            pollution_folder = os.path.join(output_path, pollution_mode)
            if not os.path.isdir(pollution_folder):
                os.makedirs(pollution_folder)
            
            csv_path = os.path.join(pollution_folder, "metrics_by_iteration.csv")
            df_metrics.to_csv(csv_path, index=False)

def save_accuracy_csv(metrics_by_feature_pollution, output_path, feature_names, pollution_modes, num_iterations):
    """
    Aggregate and save accuracy, F1, TPR and FPR — overall and per sensitive group — to CSV.
    Creates one CSV per pollution mode. Each row is one iteration; columns cover all
    features × groups × metrics.
    """
    for pollution_mode in pollution_modes:
        rows = {}  # iteration -> {col_name: value}

        for iteration in range(num_iterations):
            rows[iteration] = {}

            for feat in feature_names:
                key = (feat, pollution_mode)
                if key not in metrics_by_feature_pollution:
                    continue

                feat_metrics = metrics_by_feature_pollution[key]
                if iteration not in feat_metrics:
                    continue

                iter_data = feat_metrics[iteration]

                for group_key, group_metrics in iter_data.items():
                    # group_key is either 'overall' or a sensitive attribute value (0, 1, ...)
                    group_label = f"group{group_key}" if group_key != 'overall' else 'overall'

                    for metric_name, metric_val in group_metrics.items():
                        col = f"{feat}_{group_label}_{metric_name}"
                        rows[iteration][col] = metric_val

        if rows:
            df_metrics = pd.DataFrame.from_dict(rows, orient='index')
            df_metrics.index.name = 'iteration'

            pollution_folder = os.path.join(output_path, pollution_mode)
            if not os.path.isdir(pollution_folder):
                os.makedirs(pollution_folder)

            csv_path = os.path.join(pollution_folder, "metrics_by_iteration.csv")
            df_metrics.to_csv(csv_path)

def iterate_process_old(X_test, y_test, feat_to_remove, UMO_mode, sensitive_column_name, sensitive_attribute_value, num_iterations, minority, reg_model_features, umo_input_features):
    features_bias_ranges = calculate_bias_range_by_feat()
    pollution_modes = ['linear_pollution', 'probabilistic_pollution', 'probabilistic_half_lies']
    
    # Dictionary to store metrics for each feature and pollution mode
    metrics_by_feature_pollution = {}

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
            results_per_iteration = {}  # Collect results for each iteration

            for k in range(num_iterations):
                results_per_iteration[k] = []  # Store all results for this iteration
                
                for bias in np.arange(0.4, 1.6, 0.2):
                    bias = round(bias, 1)
                    res = predict_on_biased_dataset(X_test.copy(), sensitive_column_name, sensitive_attribute_value, bias, features_bias_ranges, feat, k, y_test, feat_to_remove, pollution_mode, UMO_mode, mode='baseline', reg_model_features=reg_model_features, umo_input_features=umo_input_features)
                    res['feature'] = feat
                    res['iteration'] = k  # Add iteration number

                    # Store results for this iteration (for later metrics computation)
                    results_per_iteration[k].append(res.copy())

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
                
                # Compute accuracy and F1 for this iteration
                iteration_metrics = compute_accuracy(results_per_iteration[k], sensitive_column_name)
                key = (feat, pollution_mode)
                if key not in metrics_by_feature_pollution:
                    metrics_by_feature_pollution[key] = {}
                metrics_by_feature_pollution[key].update(iteration_metrics)

            pollution_folder = output_path + f'/{pollution_mode}'

            # usa la funzione di riepilogo che scrive UN SOLO CSV per feature (confusion matrix aggregata + statistiche UMO)
            if raw_res_baseline:
                df_raw = pd.concat(raw_res_baseline, axis=0, ignore_index=True)
            else:
                df_raw = pd.DataFrame()
            calculate_UMO_stats(feat, df_raw, performance_list_baseline, pollution_folder, sensitive_column_name)

    # Save aggregated accuracy and F1 metrics to CSV files
    save_accuracy_csv(metrics_by_feature_pollution, output_path, feature_names, pollution_modes, num_iterations)

    return


def iterate_process(X_test, y_test, feat_to_remove, UMO_mode, sensitive_column_name,
                    sensitive_attribute_value, num_iterations, minority,
                    reg_model_features, umo_input_features):
    features_bias_ranges = calculate_bias_range_by_feat()
    pollution_modes = ['linear_pollution', 'probabilistic_pollution', 'probabilistic_half_lies']
    metrics_by_feature_pollution = {}

    if sensitive_column_name is None:
        create_sensitive_columns_dict(num_iterations, minority)
        sensitive_column_name = 'dummy_sensitive'

    for pollution_mode in pollution_modes:
        pollution_folder = output_path + f'/{pollution_mode}'
        if not os.path.isdir(pollution_folder):
            os.makedirs(pollution_folder)

    for feat in feature_names:
        for pollution_mode in pollution_modes:

            lista_tp_baseline = []
            lista_tn_baseline = []
            lista_fp_baseline = []
            lista_fn_baseline = []
            lista_all_baseline = []
            performance_list_baseline = []
            raw_res_baseline = []
            results_per_iteration = {}

            for k in range(num_iterations):
                results_per_iteration[k] = []

                # ── NEW: collect one result per bias for this iteration ──────────
                bias_dfs = {}
                # ────────────────────────────────────────────────────────────────

                for bias in np.arange(0.4, 1.6, 0.2):
                    bias = round(bias, 1)
                    res = predict_on_biased_dataset(
                        X_test.copy(), sensitive_column_name, sensitive_attribute_value,
                        bias, features_bias_ranges, feat, k, y_test, feat_to_remove,
                        pollution_mode, UMO_mode, mode='baseline',
                        reg_model_features=reg_model_features,
                        umo_input_features=umo_input_features
                    )
                    res['feature'] = feat
                    res['iteration'] = k

                    results_per_iteration[k].append(res.copy())

                    # rename UMO column for this bias level
                    res_renamed = res.rename(columns={"UMO": f"UMO_{bias}"})

                    # ── NEW: store in dict instead of appending directly ─────────
                    bias_dfs[bias] = res_renamed
                    # ────────────────────────────────────────────────────────────

                    # all downstream aggregations are unchanged
                    performance_list_baseline.append(
                        calculate_performance(sensitive_column_name, res_renamed)
                    )
                    zz = res_renamed.groupby([sensitive_column_name]).agg({
                        f"UMO_{bias}": "mean",
                        sensitive_column_name: "count"
                    }).rename(columns={sensitive_column_name: f"count_{bias}"})
                    lista_all_baseline.append(zz)

                    res_grouped = res_renamed.groupby(
                        [sensitive_column_name, 'mean_pred_0_removed_ft', 'true_class']
                    ).agg({
                        f"UMO_{bias}": "mean",
                        sensitive_column_name: "count"
                    }).rename(columns={sensitive_column_name: f"count_{bias}"}).reset_index()

                    tp = res_grouped[
                        (res_grouped['mean_pred_0_removed_ft'] == 1) & (res_grouped['true_class'] == 1)
                    ][[sensitive_column_name, f"UMO_{bias}", f"count_{bias}"]].set_index(sensitive_column_name)
                    tn = res_grouped[
                        (res_grouped['mean_pred_0_removed_ft'] == 0) & (res_grouped['true_class'] == 0)
                    ][[sensitive_column_name, f"UMO_{bias}", f"count_{bias}"]].set_index(sensitive_column_name)
                    fp = res_grouped[
                        (res_grouped['mean_pred_0_removed_ft'] == 1) & (res_grouped['true_class'] == 0)
                    ][[sensitive_column_name, f"UMO_{bias}", f"count_{bias}"]].set_index(sensitive_column_name)
                    fn = res_grouped[
                        (res_grouped['mean_pred_0_removed_ft'] == 0) & (res_grouped['true_class'] == 1)
                    ][[sensitive_column_name, f"UMO_{bias}", f"count_{bias}"]].set_index(sensitive_column_name)

                    lista_tp_baseline.append(tp)
                    lista_tn_baseline.append(tn)
                    lista_fp_baseline.append(fp)
                    lista_fn_baseline.append(fn)


                # Use bias=1.0 as anchor for static columns (same sample order,
                # true labels and neutral prediction are the reference condition)
                neutral_bias = 1.0
                anchor = bias_dfs[neutral_bias][
                    [sensitive_column_name, 'mean_pred_0_removed_ft', 'true_class', 'iteration', 'feature']
                ].copy()

                for bias_val, df_b in bias_dfs.items():
                    umo_col = f'UMO_{bias_val}'
                    anchor[umo_col] = df_b[umo_col].values

                raw_res_baseline.append(anchor)

                iteration_metrics = compute_accuracy(results_per_iteration[k], sensitive_column_name)
                key = (feat, pollution_mode)
                if key not in metrics_by_feature_pollution:
                    metrics_by_feature_pollution[key] = {}
                metrics_by_feature_pollution[key].update(iteration_metrics)

            pollution_folder = output_path + f'/{pollution_mode}'

            if raw_res_baseline:
                df_raw = pd.concat(raw_res_baseline, axis=0, ignore_index=True)
            else:
                df_raw = pd.DataFrame()

            calculate_UMO_stats(feat, df_raw, performance_list_baseline,
                                pollution_folder, sensitive_column_name)

    save_accuracy_csv(metrics_by_feature_pollution, output_path,
                      feature_names, pollution_modes, num_iterations)
    return


for minority_rate in [0.5]:

    pipeline_mode = 'classification'

    if pipeline_mode == 'classification':

        sensitive_column_name = 'dummy_sensitive'
        sensitive_attribute_value = 0
        learner = 'DecisionTree'

        df, feature_weights = generate_synthetic_dataset(n_samples=10000, random_state=42)
        X_train, X_test, y_train, y_test, cls, umo_input_features, model_trained_features, optimal_threshold = train_model_with_threshold_normalization_and_binarization(df, sensitive_column_name, learner)
        feat_to_remove = [] # Assuming feat_to_remove is defined globally or passed as an argument

        feature_names = [x for x in umo_input_features if (x not in feat_to_remove)] # feature_names for iterate_process
        output_path = "data/output/" + "synthetic_dataset_" + 'TRIAL_1' # Define output_path
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
        num_iterations = 10
        UMO_mode = 'discrete'

        # Generate contour plots for model probability forecasts (overlay train and test points)
        # plot_probability_contours(reg, reg_model_features, X_train, X_test, y_train, y_test, optimal_threshold=optimal_threshold, output_path=output_path, learner=learner, grid_points=100)

        iterate_process(X_test, y_test, feat_to_remove, UMO_mode, sensitive_column_name=sensitive_column_name, sensitive_attribute_value=sensitive_attribute_value, num_iterations=num_iterations, minority=minority_rate, reg_model_features=reg_model_features, umo_input_features=umo_input_features)
        # mode_list = ['baseline']
        
        plot_umo_mean_vs_bias(output_path, feature_names, feature_weights, sensitive_column_name=sensitive_column_name, sensitive_attribute_value=sensitive_attribute_value, num_iterations=num_iterations)
        plot_umo_statistics_summary(output_path, feature_names, feature_weights, sensitive_column_name=sensitive_column_name, sensitive_attribute_value=sensitive_attribute_value, num_iterations=num_iterations)
        plot_accuracy_by_group(output_path, feature_names, feature_weights, sensitive_column_name=sensitive_column_name, sensitive_attribute_value=sensitive_attribute_value, num_iterations=num_iterations)
        plot_umo_disparity_gap(output_path, feature_names, feature_weights, sensitive_column_name, sensitive_attribute_value=0)
        

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
            iterate_process(X_test, y_test, feat_to_remove, UMO_mode, sensitive_column_name=sensitive_column_name, sensitive_attribute_value=sensitive_attribute_value, num_iterations=1, minority=minority_rate, reg_model_features=reg_model_features, umo_input_features=umo_input_features)
            mode_list = ['baseline']
            
            # make_plots('mean', output_path, feature_names, mode_list=mode_list, sensitive_column_name=sensitive_column_name, sensitive_attribute_value=sensitive_attribute_value)
            plot_umo_mean_vs_bias(output_path, feature_names, sensitive_column_name=sensitive_column_name, sensitive_attribute_value=sensitive_attribute_value)
