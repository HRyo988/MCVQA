# -*- coding: utf-8 -*-
"""
This script shows how to evaluate the performance of a model using the best parameters
"""
import pandas
import scipy.io
import argparse
import time
import math
import os, sys
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from scipy.optimize import curve_fit
from sklearn.svm import LinearSVR
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ----------------------- Set System logger ------------- #
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='RAPIQUE',
                        help='Evaluated BVQA model name.')
    parser.add_argument('--dataset_name', type=str, default='mc_study',
                        help='Evaluation dataset.')
    parser.add_argument('--feature_file', type=str,
                        default='feat_files/mc_RAPIQUE_feats.mat',
                        help='Pre-computed feature matrix.')
    parser.add_argument('--mos_file', type=str,
                        default='mos_files/mc_us.csv',
                        help='Dataset MOS scores.')
    parser.add_argument('--out_file', type=str,
                        default='result/mc_RAPIQUE_SVR_corr.mat',
                        help='Output correlation results')
    parser.add_argument('--log_file', type=str,
                        default='logs/McVQA.log',
                        help='Log files.')
    parser.add_argument('--color_only', action='store_true',
                        help='Evaluate color values only. (Only for YouTube UGC)')
    parser.add_argument('--log_short', action='store_true',
                        help='Whether log short')
    parser.add_argument('--use_parallel', action='store_true',
                        help='Use parallel for iterations.')
    parser.add_argument('--num_iterations', type=int, default=1,
                        help='Number of iterations of train-test splits')
    parser.add_argument('--max_thread_count', type=int, default=4,
                        help='Number of threads.')
    args = parser.parse_args()
    return args

# logistic_func
def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    # 4-parameter logistic function
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

# compute_metrics
def compute_metrics(y_pred, y):
    '''
  compute metrics btw predictions & labels
  '''
    # compute SRCC & KRCC
    SRCC = scipy.stats.spearmanr(y, y_pred)[0]
    try:
        KRCC = scipy.stats.kendalltau(y, y_pred)[0]
    except:
        KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

    # logistic regression btw y_pred & y
    beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
    popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
    y_pred_logistic = logistic_func(y_pred, *popt)

    # compute  PLCC RMSE
    PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
    RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
    return [SRCC, KRCC, PLCC, RMSE]

# print formatting
def formatted_print(snapshot, duration):
    print('======================================================')
    print('SRCC_train: ', snapshot[0])
    print('KRCC_train: ', snapshot[1])
    print('PLCC_train: ', snapshot[2])
    print('RMSE_train: ', snapshot[3])
    print('======================================================')
    print(' -- ' + str(duration) + ' seconds elapsed...\n\n')


# calculate average scores
def final_avg(snapshot):
    def formatted(args, pos):
        mean = np.mean(list(map(lambda x: x[pos], snapshot)))
        stdev = np.std(list(map(lambda x: x[pos], snapshot)))
        print('{}: {} (std: {})'.format(args, mean, stdev))

    print('======================================================')
    print('')
    formatted("SRCC Train", 0)
    formatted("KRCC Train", 1)
    formatted("PLCC Train", 2)
    formatted("RMSE Train", 3)
    print('\n\n')


def evaluate_bvqa_one_split(i, X, y, log_short):
    if not log_short:
        print('{} th repeated holdout test'.format(i))
        t_start = time.time()

    # Step1:only training 100% data
    X_train, y_train = X, y
    print(f'X_train shape: {X_train.shape}')

    # 欠損値処理
    # 欠損値を平均値で補完
    from scipy import stats
    X_train = np.where(np.isnan(X_train), np.nanmean(X_train, axis=0), X_train)

    # Step2:fix parameters
    print(f'{X_train.shape[1]}-dim features, using LinearSVR')
    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    # init model
    best_params = {'C':0.01, 'epsilon':0.01}
    regressor = LinearSVR(C=best_params['C'], epsilon=best_params['epsilon'], loss='squared_epsilon_insensitive')
    regressor.fit(X_train, y_train)
    # Step3:predictions
    y_train_pred = regressor.predict(X_train)

    # Calculate RMSE
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    RMSE = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print('RMSE:', RMSE)

    # Step4:compute metrics and plot figure
    metrics_train = compute_metrics(y_train_pred, y_train)

    # Step5:compute metrics and plot figure
    if not log_short:
        t_end = time.time()
        formatted_print(metrics_train,(t_end - t_start))
    
    # Step6:save SVR model trained by using joblib at same directory
    import joblib
    joblib.dump(regressor, 'McVQA.pkl')
    return metrics_train


def main(args):
    df = pandas.read_csv(args.mos_file, skiprows=[], header=None)
    array = df.values
    if args.dataset_name == 'mc_study':
        y = array[1:, 9]
    y = np.array(list(y), dtype=np.float64)
    X_mat = scipy.io.loadmat(args.feature_file)
    X = np.asarray(X_mat['feats_mat'], dtype=np.float64)

    # preprocessing
    X[np.isinf(X)] = np.nan
    imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
    X = imp.transform(X)
    X = X[:, 1360:3408] # CNN features
    # Extract a dataset of labels 1 and 5
    idx_list = []
    idx_list1 = []
    for idx in range(198):
        if y[idx] !=1.0 and y[idx] !=5.0:
            # print(idx, y[idx])
            idx_list.append(idx)
        else:
            idx_list1.append(idx)
    X_del = np.delete(X, idx_list,0)
    y_del = np.delete(y, idx_list)
    all_iterations = []
    t_overall_start = time.time()

    # Calculate average score
    for i in range(0, args.num_iterations):
        metrics_train = evaluate_bvqa_one_split(i, X_del, y_del, args.log_short)
        all_iterations.append(metrics_train)
    
    # Averages the scores of the number of num_iterations
    final_avg(all_iterations)
    print('Overall {} secs lapsed..'.format(time.time() - t_overall_start))

    # save logs
    dir_path = os.path.dirname(args.out_file)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    scipy.io.savemat(args.out_file,
                     mdict={'all_iterations': np.asarray(all_iterations, dtype=np.float64)})


if __name__ == '__main__':
    args = arg_parser()
    log_file = args.log_file
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = Logger(log_file)
    print(args)
    main(args)