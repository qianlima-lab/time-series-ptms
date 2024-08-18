import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


import numpy as np
import matplotlib.pyplot as plt
from spot import dSPOT
import numpy as np
import time
import datetime
import datautils
from sklearn.metrics import f1_score, precision_score, recall_score
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score
from other_anomaly_baselines.metrics.affiliation.metrics import pr_from_events
from other_anomaly_baselines.metrics.vus.metrics import get_range_vus_roc
from other_anomaly_baselines.metrics.affiliation.generics import convert_vector_to_events
from tadpak import evaluate


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

# consider delay threshold and missing segments
def get_range_proba(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict



parser = argparse.ArgumentParser()
# parser.add_argument('dataset', help='The dataset name')
# parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
# parser.add_argument('--dataset', default='kpi', help='The dataset name, yahoo, kpi')
parser.add_argument('--dataset', default='kpi',
                help='The dataset name, yahoo, kpi')  ##  SMD, MSL, SMAP, PSM, SWAT, NIPS_TS_Swan, UCR, NIPS_TS_Water
parser.add_argument('--is_multi', default=False, help='The dataset name, yahoo, kpi')
parser.add_argument('--datapath', default='./datasets/', help='')
parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
parser.add_argument('--index', type=int, default=143, help='')
parser.add_argument('--batch_size', type=int, default=8, help='The batch size (defaults to 8)')
parser.add_argument('--save_dir', type=str, default='/dev_data/lz/tsm_ptms_anomaly_detection/result/')
parser.add_argument('--save_csv_name', type=str, default='dspot_0719.csv')

args = parser.parse_args()

print("Dataset:", args.dataset)
print("Arguments:", str(args))

if args.is_multi:
    from datasets.data_loader import get_loader_segment

    data_path = args.datapath + args.dataset + '/'
    print("data_path = ", data_path)
    _, train_data_loader = get_loader_segment(args.index, data_path, args.batch_size, win_size=100, step=100,
                                              mode='train',
                                              dataset=args.dataset)

    all_train_data = train_data_loader.train
    all_train_labels = None
    all_train_timestamps = None
    all_test_data = train_data_loader.test
    all_test_labels = train_data_loader.test_labels
    all_test_timestamps = None
    delay = 5

    all_train_data = np.squeeze(all_train_data)
    all_test_data = np.squeeze(all_test_data)

    print("all_train_data test_data, test_labels.shape = ", all_train_data.shape, all_test_data.shape,
          all_test_labels.shape)
    # train_data = np.expand_dims(all_train_data, axis=0)
    # print("train_data.shape = ", train_data.shape)
    print("Read Success!!!")

else:

    # dataset = 'kpi' # yahoo, kpi
    print('Loading data... ', end='')
    all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)



labels = []
pred = []
scores = []

if args.is_multi:
    train_data = all_train_data  # initial batch
    train_labels = all_train_labels

    test_data = all_test_data # stream
    test_labels = all_test_labels
    test_timestamps = all_test_timestamps

    q = 1e-4  # risk parameter # yahoo: 1e-3
    d = 50  # depth
    s = dSPOT(q, d)  # DSPOT object
    s.fit(train_data, test_data)  # data import
    s.initialize()  # initialization step
    results = s.run()  # run

    test_thresholds = results['thresholds']
    idx_anoamly = results['alarms']

    test_pred = np.zeros(len(test_thresholds))
    test_pred[idx_anoamly] = 1

    test_pred = get_range_proba(test_pred, test_labels, delay)

    labels.append(test_labels)
    pred.append(test_pred)
    scores.append(results['scores'])
else:
    for k in all_test_data:
        train_data = all_train_data[k] # initial batch
        train_labels = all_train_labels[k]
        train_timestamps = all_train_timestamps[k]

        test_data = all_test_data[k] # stream
        test_labels = all_test_labels[k]
        test_timestamps = all_test_timestamps[k]

        q = 1e-4            # risk parameter # yahoo: 1e-3
        d = 50              # depth
        s = dSPOT(q, d)  		# DSPOT object
        s.fit(train_data, test_data) 	# data import
        s.initialize() 		# initialization step
        results = s.run() 	# run

        test_thresholds = results['thresholds']
        idx_anoamly = results['alarms']

        test_pred = np.zeros(len(test_thresholds))
        test_pred[idx_anoamly] = 1

        test_pred = get_range_proba(test_pred, test_labels, delay)

        labels.append(test_labels)
        pred.append(test_pred)
        scores.append(results['scores'])

labels = np.concatenate(labels)
pred = np.concatenate(pred)

scores = np.concatenate(scores)

if args.is_multi:
    # labels = np.asarray(labels_log, np.int64)[0]
    # pred = np.asarray(res_log, np.int64)[0]
    # print("labels.shape = ", labels.shape, labels[:5])
    # print("pred.shape = ", pred.shape, pred[:5])

    labels, pred = adjustment(labels, pred)

    events_pred = convert_vector_to_events(pred)
    events_gt = convert_vector_to_events(labels)

    Trange = (0, len(labels))
    affiliation = pr_from_events(events_pred, events_gt, Trange)
    vus_results = get_range_vus_roc(labels, pred, 100)  # default slidingWindow = 100

    pred_scores = scores
    results_f1_pa_k_10 = evaluate.evaluate(pred_scores, labels, k=10)
    # results_f1_pa_k_30 = evaluate.evaluate(pred, labels, k=30)
    results_f1_pa_k_50 = evaluate.evaluate(pred_scores, labels, k=50)
    results_f1_pa_k_90 = evaluate.evaluate(pred_scores, labels, k=90)

    eval_res = {
        'f1': f1_score(labels, pred),
        'precision': precision_score(labels, pred),
        'recall': recall_score(labels, pred),
        "Affiliation precision": affiliation['precision'],
        "Affiliation recall": affiliation['recall'],
        "R_AUC_ROC": vus_results["R_AUC_ROC"],
        "R_AUC_PR": vus_results["R_AUC_PR"],
        "VUS_ROC": vus_results["VUS_ROC"],
        "VUS_PR": vus_results["VUS_PR"],
        'f1_pa_10': results_f1_pa_k_10['best_f1_w_pa'],
        'f1_pa_50': results_f1_pa_k_50['best_f1_w_pa'],
        'f1_pa_90': results_f1_pa_k_90['best_f1_w_pa'],
    }
    print("eval_res = ", eval_res)
else:

    print('\nf1:', f1_score(labels, pred))
    print('precision:', precision_score(labels, pred))
    print('recall:', recall_score(labels, pred))

    events_pred = convert_vector_to_events(pred)
    events_gt = convert_vector_to_events(labels)

    Trange = (0, len(labels))
    affiliation = pr_from_events(events_pred, events_gt, Trange)
    vus_results = get_range_vus_roc(labels, pred, 100)  # default slidingWindow = 100

    eval_res = {
        'f1': f1_score(labels, pred),
        'precision': precision_score(labels, pred),
        'recall': recall_score(labels, pred),
        "Affiliation precision": affiliation['precision'],
        "Affiliation recall": affiliation['recall'],
        "R_AUC_ROC": vus_results["R_AUC_ROC"],
        "R_AUC_PR": vus_results["R_AUC_PR"],
        "VUS_ROC": vus_results["VUS_ROC"],
        "VUS_PR": vus_results["VUS_PR"]
    }

    # results_f1_pa_k_10 = evaluate.evaluate(scores, labels, k=10)
    # # results_f1_pa_k_30 = evaluate.evaluate(pred, labels, k=30)
    # results_f1_pa_k_50 = evaluate.evaluate(scores, labels, k=50)
    # results_f1_pa_k_90 = evaluate.evaluate(scores, labels, k=90)
    #
    # eval_res['f1_pa_10'] = results_f1_pa_k_10['best_f1_w_pa']
    # eval_res['f1_pa_50'] = results_f1_pa_k_50['best_f1_w_pa']
    # eval_res['f1_pa_90'] = results_f1_pa_k_90['best_f1_w_pa']


eval_res['dataset'] = args.dataset + str(args.index)
import pandas as pd
import os

# 转换字典为 DataFrame
df = pd.DataFrame([eval_res])
# 指定保存路径
save_path = args.save_dir + args.save_csv_name

# 转换字典为 DataFrame
df_new = pd.DataFrame([eval_res])

# 检查文件是否存在
if os.path.exists(save_path):
    # 文件存在，读取现有数据
    df_existing = pd.read_csv(save_path, index_col=0)
    # 将新数据附加到现有数据框中
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
else:
    # 文件不存在，创建新的数据框
    df_combined = df_new

# 保存 DataFrame 为 CSV 文件
df_combined.to_csv(save_path, index=True, index_label="id")

print("Finished.")