import os
import re
import time
from itertools import combinations
import random

#import seaborn as sns
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from plots import make_plots

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

def shap_values_compute(reg, X_test, output_path):

    shap_explainer = shap.Explainer(reg.predict, X_test)
    shap_values = shap_explainer(X_test)

    shap.summary_plot(shap_values, X_test.sample(200), plot_type="bar", show=False)
    plt.savefig("shap_summary_plot.png", bbox_inches='tight', dpi=300)


    shap.summary_plot(shap_values, X_test.sample(200).reset_index(inplace=True), plot_type="violin", show=False)
    plt.savefig("shap_violin_plot.png", bbox_inches='tight', dpi=300)

    return

def train_model(dataset, sensitive_column):
    df = dataset  #.set_index('Unnamed: 0').sort_index()
    X = df.iloc[:, :-1]  # All columns except the last one
    input_features = df.iloc[:, :-1].columns.to_list()
    if sensitive_column != None:
        input_features.remove(sensitive_column)

    y = df.iloc[:, -1]  #last column
    print(df.iloc[:, -1].min())
    print(df.iloc[:, -1].max())
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
    print(f"Soglia ottimale: {optimal_threshold}")

    # Converti y_test in binario usando la stessa soglia
    predictions = (y_scores >= optimal_threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuratezza classificazione binaria: {accuracy:.4f}")

    f1 = f1_score(y_test, predictions)
    print(f"F1 Score: {f1:.4f}")

    shap_values_compute(reg, X_test, output_path)
    #shap.plots.waterfall(shap_values[0])
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_scores, pos_label=1)
    # roc_auc = metrics.roc_auc_score(y_test, y_scores, pos_label=1)
    #
    # # # Compute accuracy for each threshold
    # # accuracies = []
    # # for threshold in thresholds:
    # #     y_pred = (y_scores >= threshold).astype(int)  # Convert probabilities to class labels
    # #     acc = np.sqrt(mean_squared_error(y_test, y_pred))
    # #     accuracies.append(acc)
    # #
    # # # Find the best threshold
    # # best_idx = np.argmax(accuracies)
    # # best_threshold = thresholds[best_idx]
    # # best_accuracy = accuracies[best_idx]
    # #
    # # print('Optimal threshold: ' + str(best_threshold))
    # # print('Accuracy:' + str(best_accuracy))
    #
    # # Plot ROC Curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random classifier)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc='lower right')
    #
    # # Save the plot
    # plt.savefig(output_path + "/roc_curve.png", dpi=300)  # Save as PNG with high resolution
    # plt.show()  # Display the plot

    return X_train, X_test, y_train, y_test, reg, input_features, optimal_threshold,


def cutoff_prediction(score, optimal_threshold):
    return (score >= optimal_threshold).astype(int)
    #return score


def train_model_with_threshold_normalization_and_binarization(df, sensitive_column):
    X = df.iloc[:, :-1]  # Tutte le colonne tranne l'ultima
    input_features = df.iloc[:, :-1].columns.to_list()
    if sensitive_column is not None:
        input_features.remove(sensitive_column)

    y = df.iloc[:, -1]  # Ultima colonna
    print(f"Target min: {y.min()}")
    print(f"Target max: {y.max()}")

    # Split del dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=28)

    # Normalizza la variabile target
    y_scaler = MinMaxScaler()
    y_train_normalized = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_normalized = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

    # Addestra il modello di regressione
    reg = XGBRegressor(random_state=0,
                       n_estimators=10,
                       max_depth=5,
                       objective="reg:squarederror",
                       eval_metric="rmse",
                       verbosity=1)

    reg.fit(X_train, y_train_normalized)
    y_scores_normalized = reg.predict(X_test)

    # Calcola la soglia ottimale
    optimal_threshold = find_optimal_threshold(y_test_normalized, y_scores_normalized, metric='accuracy')
    print(f"Soglia ottimale: {optimal_threshold}")

    # Applica la soglia per ottenere predizioni binarie
    y_binary_pred = (y_scores_normalized >= optimal_threshold).astype(int)

    # Calcola le metriche sia per la regressione che per la classificazione
    # Denormalizza per l'RMSE
    y_scores = y_scaler.inverse_transform(y_scores_normalized.reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_scores))

    # Converti y_test in binario usando la stessa soglia
    y_test_binary = (y_test_normalized >= optimal_threshold).astype(int)

    # Calcola accuratezza e F1 per la classificazione binaria
    accuracy = accuracy_score(y_test_binary, y_binary_pred)
    f1 = f1_score(y_test_binary, y_binary_pred)

    print(f"RMSE: {rmse}")
    print(f"Accuratezza classificazione binaria: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    #shap_values_compute(reg, X_test, input_features)
    return X_train, X_test, y_train_normalized, y_test_normalized, reg, input_features, optimal_threshold


def calculate_bias_range_by_feat():
    dictionary = {}
    for feat in feature_names:
        dictionary[feat] = []
        dictionary[feat].append(X_test[feat].min())
        dictionary[feat].append(X_test[feat].max())
    print(dictionary)
    return dictionary


def create_array(min):
    n = X_test.shape[0]
    minority_rate = min
    majority_rate = 1 - minority_rate
    arr = np.array(['A'] * int(n * minority_rate) + ['B'] * int(n * majority_rate) + random.choice([['A'], ['B']]))
    np.random.shuffle(arr)  # Shuffle to randomize position
    return arr


def add_noise(row, biased_column, bias, sensitive_attribute, bias_range, sensitive_column, k):
    if row[sensitive_column] == sensitive_attribute:
        output = row[biased_column] * bias
        if output < bias_range[biased_column][0]:
            output = bias_range[biased_column][0]
        elif output > bias_range[biased_column][1]:
            output = bias_range[biased_column][1]
    else:
        output = row[biased_column]
    return output


def calculate_UMO_old(row):
    umo_level = 0
    for i in range(1, len(input_features) + 1):
        print(i)
        print(row)
        print(row[row.index[i]])
        print(row[row.index[i - 1]])
        if row[row.index[i]] == row[row.index[i - 1]]:
            umo_level += 1
        else:
            break
    return umo_level

def calculate_UMO(df, input_features):
    start_time = time.time()  # Start timing
    df = df.iloc[:, :-1]
    baseline = df.iloc[:, 0]  # First column as baseline
    diffs = df.ne(baseline, axis=0)  # Compare each column with the baseline
    first_diff = diffs.values.argmax(axis=1) - 1
    # Get the first different column index
    # If no deviation, set to None
    first_diff[~diffs.any(axis=1)] = len(df.columns) - 1
    outpt = pd.Series(first_diff, index=df.index)

    end_time = time.time()  # End timing
    duration = end_time - start_time
    print(f"Execution time: {duration:.4f} seconds")
    return outpt  # Return as a Series with row index


def predict_on_biased_dataset(X_test, sensitive_column, sensitive_attribute, bias, bias_range, biased_column, k, y_test, feat_to_remove, mode):
    df_list = []
    data = X_test.copy().reset_index(drop=True)
    if mode == 'baseline':
        # remove all values of feat_to_remove
        data[feat_to_remove] = np.NAN

    if sensitive_column == None:
        sens = pd.read_csv(output_path + f"/sensitive_feature_distr/iteration_{k}.csv").reset_index(drop=True)
        data['dummy_sensitive'] = sens[['0']]
        sensitive_column = 'dummy_sensitive'
        sensitive_attribute = 'A'

    data[biased_column] = data.apply(
        lambda x: add_noise(x, biased_column, bias, sensitive_attribute, bias_range, sensitive_column, k), axis=1)

    if mode == 'active_learning':
        # remove values of feat_to_remove for privileged datapoints
        data.loc[data['dummy_sensitive'] != sensitive_attribute, feat_to_remove] = np.NAN

    for i in range(len(input_features) + 1):
        subdict = {}

        for subset in combinations(input_features, i):

            data_copy = data.copy()
            if sensitive_column == 'dummy_sensitive':
                data_copy = data_copy.drop(columns=['dummy_sensitive'])
            if i == 0:
                subset = 'baseline'
            else:
                data_copy[list(subset)] = np.NAN
            # Predict
            subdict[subset] = cutoff_prediction(reg.predict(data_copy), optimal_threshold)
            #accuracy = accuracy_score(cutoff_prediction(y_test, optimal_threshold), cutoff_prediction(reg.predict(data_copy), optimal_threshold))
            #print(f"Accuratezza classificazione binaria: {accuracy:.4f}")

        sub_df = pd.DataFrame.from_dict(subdict)
        sub_df[f'mean_pred_{i}_removed_ft'] = sub_df.agg("mean", axis="columns")  #mean of predictions
        sub_df[f'mean_pred_{i}_removed_ft'] = cutoff_prediction(sub_df[f'mean_pred_{i}_removed_ft'],
                                                                optimal_threshold)  #thresholding the mean prediction

        df_list.append(sub_df)

    total = pd.concat(df_list, axis=1, ignore_index=False).filter(like='removed', axis=1)
    total['true_class'] = cutoff_prediction(y_test, optimal_threshold)
    total[f'UMO'] = calculate_UMO(total, input_features)
    total['dummy_sensitive'] = data['dummy_sensitive']
    #total.to_csv(f'check_this_bias_{bias}_{biased_column}.csv')
    #print(total.groupby(['true_class', 'dummy_sensitive']).count())
    #print(total[total['true_class']==1].groupby(['dummy_sensitive', 'mean_pred_0_removed_ft']).count())
    #print(total[total['true_class']==1].groupby(['dummy_sensitive'])['UMO'].mean())
    #print(total.groupby(['dummy_sensitive','mean_pred_0_removed_ft', 'true_class'])['UMO'].mean())
    #total.to_csv('../data/output/UMO_diabetes.csv')
    return total


def create_sensitive_column(k):
    col = create_array()
    pd.DataFrame(col).to_csv(output_path + f"/sensitive_feature_distr/iteration_{k}.csv")
    return col


def calculate_performance(sensitive_column, res):
    dictionary = {}
    for attr in res[sensitive_column].unique():
        new_df = res[res[sensitive_column] == attr]
        dictionary[attr] = confusion_matrix(new_df['mean_pred_0_removed_ft'], new_df['true_class'],
                                            labels=[0, 1]).ravel()
    #print(dictionary)
    overall = pd.DataFrame.from_dict(dictionary, orient='index', columns=['tn', 'fp', 'fn', 'tp'])
    return overall

def create_sensitive_columns_dict(m, min):
    diz = {}
    for a in range(m):
        diz[a] = create_array(min)
        pd.DataFrame(diz[a]).to_csv(output_path + f"/sensitive_feature_distr/iteration_{a}.csv")
    return

def iterate_process(X_test, y_test, feat_to_remove, sensitive_column, sensitive_attribute, num_iterations, minority):
    bias_range = calculate_bias_range_by_feat()
    regression_list_baseline = []
    regression_list_active_learning = []
    if sensitive_column == None:
        create_sensitive_columns_dict(num_iterations, minority)
    for feat in feature_names:
        lista_tp_baseline = []
        lista_tn_baseline = []
        lista_fp_baseline = []
        lista_fn_baseline = []
        lista_all_baseline = []
        performance_list_baseline = []

        lista_tp_active_learning = []
        lista_tn_active_learning = []
        lista_fp_active_learning = []
        lista_fn_active_learning = []
        lista_all_active_learning = []
        performance_list_active_learning = []
        for k in range(num_iterations):
            for bias in np.arange(0.4, 1.6, 0.2):
                bias = round(bias, 1)
                res = predict_on_biased_dataset(X_test.copy(), sensitive_column, str(sensitive_attribute), bias, bias_range, feat, k, y_test, feat_to_remove, mode='baseline')
                res['feature'] = feat
                regression_list_baseline.append(res)
                res = res.rename(columns={"UMO": f"UMO_{bias}"})
                #lista_raw_data.append(res)
                print(res['dummy_sensitive'].head())

                performance_list_baseline.append(calculate_performance('dummy_sensitive', res))
                zz = res.groupby(['dummy_sensitive']).agg({
                    f"UMO_{bias}": "mean",
                    'dummy_sensitive': "count"
                }).rename(columns={'dummy_sensitive':f"count_{bias}"})
                print(zz.head())
                lista_all_baseline.append(zz)
                res = res.groupby(['dummy_sensitive', 'mean_pred_0_removed_ft', 'true_class']).agg({
                    f"UMO_{bias}": "mean",
                    'dummy_sensitive': "count"
                }).rename(columns={'dummy_sensitive':f"count_{bias}"}).reset_index()
                tp = res[(res['mean_pred_0_removed_ft'] == 1) & (res['true_class'] == 1)][
                                    ['dummy_sensitive', f"UMO_{bias}", f"count_{bias}"]].set_index('dummy_sensitive')
                tn = res[(res['mean_pred_0_removed_ft'] == 0) & (res['true_class'] == 0)][
                    ['dummy_sensitive', f"UMO_{bias}", f"count_{bias}"]].set_index('dummy_sensitive')
                fp = res[(res['mean_pred_0_removed_ft'] == 1) & (res['true_class'] == 0)][
                                    ['dummy_sensitive', f"UMO_{bias}", f"count_{bias}"]].set_index('dummy_sensitive')
                fn = res[(res['mean_pred_0_removed_ft'] == 0) & (res['true_class'] == 1)][
                                    ['dummy_sensitive', f"UMO_{bias}", f"count_{bias}"]].set_index('dummy_sensitive')
                lista_tp_baseline.append(tp)
                lista_tn_baseline.append(tn)
                lista_fp_baseline.append(fp)
                lista_fn_baseline.append(fn)
                
        #         res = predict_on_biased_dataset(X_test.copy(), sensitive_column, str(sensitive_attribute), bias, bias_range, feat, k, y_test, feat_to_remove, mode='active_learning')
        #         res['feature'] = feat
        #         regression_list_active_learning.append(res)
        #         res = res.rename(columns={"UMO": f"UMO_{bias}"})
        #         #lista_raw_data.append(res)
        #         print(res['dummy_sensitive'].head())
        #
        #         performance_list_active_learning.append(calculate_performance('dummy_sensitive', res))
        #         zz = res.groupby(['dummy_sensitive']).agg({
        #             f"UMO_{bias}": "mean",
        #             'dummy_sensitive': "count"
        #         }).rename(columns={'dummy_sensitive':f"count_{bias}"})
        #         print(zz.head())
        #         lista_all_active_learning.append(zz)
        #         res = res.groupby(['dummy_sensitive', 'mean_pred_0_removed_ft', 'true_class']).agg({
        #             f"UMO_{bias}": "mean",
        #             'dummy_sensitive': "count"
        #         }).rename(columns={'dummy_sensitive':f"count_{bias}"}).reset_index()
        #         tp = res[(res['mean_pred_0_removed_ft'] == 1) & (res['true_class'] == 1)][
        #                             ['dummy_sensitive', f"UMO_{bias}", f"count_{bias}"]].set_index('dummy_sensitive')
        #         tn = res[(res['mean_pred_0_removed_ft'] == 0) & (res['true_class'] == 0)][
        #             ['dummy_sensitive', f"UMO_{bias}", f"count_{bias}"]].set_index('dummy_sensitive')
        #         fp = res[(res['mean_pred_0_removed_ft'] == 1) & (res['true_class'] == 0)][
        #                             ['dummy_sensitive', f"UMO_{bias}", f"count_{bias}"]].set_index('dummy_sensitive')
        #         fn = res[(res['mean_pred_0_removed_ft'] == 0) & (res['true_class'] == 1)][
        #                             ['dummy_sensitive', f"UMO_{bias}", f"count_{bias}"]].set_index('dummy_sensitive')
        #         lista_tp_active_learning.append(tp)
        #         lista_tn_active_learning.append(tn)
        #         lista_fp_active_learning.append(fp)
        #         lista_fn_active_learning.append(fn)
        # pd.concat(lista_tp_active_learning, axis=1).transpose().to_csv(output_path + f'/{feat}_tp_active_learning.csv')
        # pd.concat(lista_tn_active_learning, axis=1).transpose().to_csv(output_path + f'/{feat}_tn_active_learning.csv')
        # pd.concat(lista_fp_active_learning, axis=1).transpose().to_csv(output_path + f'/{feat}_fp_active_learning.csv')
        # pd.concat(lista_fn_active_learning, axis=1).transpose().to_csv(output_path + f'/{feat}_fn_active_learning.csv')
        # pd.concat(lista_all_active_learning, axis=1).transpose().to_csv(output_path + f'/{feat}_all_active_learning.csv')
        # pd.concat(regression_list_active_learning, axis=0).to_csv(output_path + '/regression_active_learning.csv')
        # pd.concat(performance_list_active_learning, axis=1).transpose().to_csv(output_path + '/performance_active_learning.csv')

        pd.concat(lista_tp_baseline, axis=1).transpose().to_csv(output_path + f'/{feat}_tp_baseline.csv')
        pd.concat(lista_tn_baseline, axis=1).transpose().to_csv(output_path + f'/{feat}_tn_baseline.csv')
        pd.concat(lista_fp_baseline, axis=1).transpose().to_csv(output_path + f'/{feat}_fp_baseline.csv')
        pd.concat(lista_fn_baseline, axis=1).transpose().to_csv(output_path + f'/{feat}_fn_baseline.csv')
        pd.concat(lista_all_baseline, axis=1).transpose().to_csv(output_path + f'/{feat}_all_baseline.csv')
        pd.concat(regression_list_baseline, axis=0).to_csv(output_path + '/regression_baseline.csv')
        pd.concat(performance_list_baseline, axis=1).transpose().to_csv(output_path + '/performance_baseline.csv')

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
#



for minority_rate in [0.5]:#np.arange(0.5, 1, 0.1):
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

        sensitive_column = None
        sensitive_attribute = None
        df = pd.read_csv("data/" + dataset_name + ".csv")

        X_train, X_test, y_train, y_test, reg, input_features, optimal_threshold = train_model_with_threshold_normalization_and_binarization(df, sensitive_column)
        feat_to_remove = ['fixed acidity']
        feature_names = [x for x in input_features if (x not in feat_to_remove)]
        iterate_process(X_test, y_test, feat_to_remove, sensitive_column=sensitive_column, sensitive_attribute=sensitive_attribute, num_iterations=1, minority=minority_rate)
        mode_list = ['baseline']
        make_plots('mean', output_path, feature_names, mode_list)