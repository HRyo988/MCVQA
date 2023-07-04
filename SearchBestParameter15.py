# -*- coding: utf-8 -*-
"""
This script shows how to apply 80-20 holdout train and validate regression model to predict
MOS from the features
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
                        default='logs/McRAPIQUELinearSVR15Default.log',
                        help='Log files.')
    parser.add_argument('--log_short', action='store_true',
                        help='Whether log short')
    parser.add_argument('--use_parallel', action='store_true',
                        help='Use parallel for iterations.')
    parser.add_argument('--num_iterations', type=int, default=20,
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
def compute_metrics(y_pred, y, flag):
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

    # Plot (X, y)=(Predicted Score, MOS)
    # if flag:
    #     plt.scatter(y_pred_logistic, y, c='blue', alpha=1, marker='+')
    #     y_pred_logistic_sorted = np.sort(y_pred_logistic)
    #     plt.plot(y_pred_logistic_sorted, logistic_func(y_pred_logistic_sorted, *popt), color='r')
    #     plt.xlabel('Predicted Score')
    #     plt.ylabel('MOS')
    #     plt.savefig('mc_study_15.jpg')
    return [SRCC, KRCC, PLCC, RMSE]

# print formatting
def formatted_print(snapshot, params, duration):
    print('======================================================')
    print('params: ', params)
    print('SRCC_train: ', snapshot[0])
    print('KRCC_train: ', snapshot[1])
    print('PLCC_train: ', snapshot[2])
    print('RMSE_train: ', snapshot[3])
    print('======================================================')
    print('SRCC_test: ', snapshot[4])
    print('KRCC_test: ', snapshot[5])
    print('PLCC_test: ', snapshot[6])
    print('RMSE_test: ', snapshot[7])
    print('======================================================')
    print(' -- ' + str(duration) + ' seconds elapsed...\n\n')

# print average scores
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
    print('======================================================')
    print('')
    formatted("SRCC Test", 4)
    formatted("KRCC Test", 5)
    formatted("PLCC Test", 6)
    formatted("RMSE Test", 7)
    print('\n\n')


def evaluate_bvqa_one_split(i, X, y, log_short):
    if not log_short:
        print('{} th repeated holdout test'.format(i))
        t_start = time.time()

    # Step1:train test split
    # (train, test) = (9, 1)
    X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.1, random_state=math.ceil(8.8 * i))

    # Step2:grid search CV on the training set
    print(f'{X_train.shape[1]}-dim features, using LinearSVR')

    # grid search on liblinear
    param_grid = {'C': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.],
                  'epsilon': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.]}
    grid = RandomizedSearchCV(LinearSVR(), param_grid, cv=9)  # cv=9:(train, validation) = (8, 1) in train(0)
    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    
    # grid search fit and search best parameter
    grid.fit(X_train, y_train)
    best_params = grid.best_params_

    # init model
    print('best_params: ', best_params)
    regressor = LinearSVR(C=best_params['C'], epsilon=best_params['epsilon'])

    # Step3:re-train the model using the best parameter
    # (train, test) = (9,1)
    regressor.fit(X_train, y_train)

    # Step4:predictions
    y_train_pred = regressor.predict(X_train)
    X_test = scaler.transform(X_test)
    y_test_pred = regressor.predict(X_test)

    # Step5:compute metrics and plot figure
    metrics_train = compute_metrics(y_train_pred, y_train, flag=False)
    metrics_test = compute_metrics(y_test_pred, y_test, flag=True)

    # Step6:compute and print scores
    if not log_short:
        t_end = time.time()
        formatted_print(metrics_train + metrics_test, best_params, (t_end - t_start))
    return best_params, metrics_train, metrics_test


def main(args):
    df = pandas.read_csv(args.mos_file, skiprows=[], header=None)
    array = df.values
    if args.dataset_name == 'mc_study':
        y = array[1:, 9]
    y = np.array(list(y), dtype=np.float)
    X_mat = scipy.io.loadmat(args.feature_file)
    X = np.asarray(X_mat['feats_mat'], dtype=np.float)

    # preprocessing
    X[np.isinf(X)] = np.nan
    imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
    X = imp.transform(X)

    # Extract a dataset of labels 1 and 5
    idx_list = []
    for idx in range(198):
        if y[idx] !=1.0 and y[idx] !=5.0:
            # print(idx, y[idx])
            idx_list.append(idx)
    X_del = np.delete(X, idx_list,0)
    y_del = np.delete(y, idx_list)
    all_iterations = []
    t_overall_start = time.time()

    # Calculate average score
    for i in range(0, args.num_iterations):
        best_params, metrics_train, metrics_test = evaluate_bvqa_one_split(i, X_del, y_del, args.log_short)
        all_iterations.append(metrics_train + metrics_test)
    
    # Averages the scores of the number of num_iterations
    final_avg(all_iterations)
    print('Overall {} secs lapsed..'.format(time.time() - t_overall_start))

    # save logs
    dir_path = os.path.dirname(args.out_file)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    scipy.io.savemat(args.out_file,
                     mdict={'all_iterations': np.asarray(all_iterations, dtype=np.float)})


if __name__ == '__main__':
    args = arg_parser()
    log_file = args.log_file
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = Logger(log_file)
    print(args)
    main(args)

'''

python evaluate_bvqa_features_regression.py \
  --model_name RAPIQUE \
  --dataset_name mc_study \
  --feature_file feat_files/mc_RAPIQUE_feats.mat \
  --mos_file mos_files/mc_us.csv \
  --out_file result/mc_RAPIQUE_SVR_corr.mat\
  --use_parallel


'''