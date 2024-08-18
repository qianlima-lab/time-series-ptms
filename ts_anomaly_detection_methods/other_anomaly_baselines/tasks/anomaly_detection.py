import numpy as np
import time
import bottleneck as bn
from sklearn.metrics import f1_score, precision_score, recall_score
from other_anomaly_baselines.metrics.affiliation.metrics import pr_from_events
from other_anomaly_baselines.metrics.vus.metrics import get_range_vus_roc
from other_anomaly_baselines.metrics.affiliation.generics import convert_vector_to_events

from sklearn.metrics import f1_score, precision_score, recall_score
import bottleneck as bn
import pdb

from tadpak import evaluate


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


# set missing = 0
def reconstruct_label(timestamp, label):
    timestamp = np.asarray(timestamp, np.int64)
    index = np.argsort(timestamp)

    timestamp_sorted = np.asarray(timestamp[index])
    interval = np.min(np.diff(timestamp_sorted))

    label = np.asarray(label, np.int64)
    label = np.asarray(label[index])

    idx = (timestamp_sorted - timestamp_sorted[0]) // interval

    new_label = np.zeros(shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,), dtype=np.int)
    new_label[idx] = label

    return new_label


def eval_ad_result(test_pred_list, test_labels_list, test_timestamps_list, delay, pred_scores=None):
    labels = []
    pred = []
    ts_scores = []
    if pred_scores is not None:
        for test_pred, test_labels, test_timestamps, test_score in zip(test_pred_list, test_labels_list, test_timestamps_list, pred_scores):
            # assert test_pred.shape == test_labels.shape == test_timestamps.shape
            min_len = min(min(test_pred.shape[0], test_labels.shape[0]), test_timestamps.shape[0])
            test_pred = test_pred[:min_len]
            test_labels = test_labels[:min_len]
            test_timestamps = test_timestamps[:min_len]
            test_score = test_score[:min_len]
            min_len = min(min(test_pred.shape[0], test_labels.shape[0]), test_timestamps.shape[0])
            test_pred = test_pred[:min_len]
            test_labels = test_labels[:min_len]
            test_timestamps = test_timestamps[:min_len]
            test_labels = reconstruct_label(test_timestamps, test_labels)
            test_pred = reconstruct_label(test_timestamps, test_pred)
            test_pred = get_range_proba(test_pred, test_labels, delay)
            labels.append(test_labels)
            pred.append(test_pred)
            ts_scores.append(test_score)
    else:
        for test_pred, test_labels, test_timestamps in zip(test_pred_list, test_labels_list, test_timestamps_list):
            # assert test_pred.shape == test_labels.shape == test_timestamps.shape
            test_labels = reconstruct_label(test_timestamps, test_labels)
            test_pred = reconstruct_label(test_timestamps, test_pred)
            test_pred = get_range_proba(test_pred, test_labels, delay)
            labels.append(test_labels)
            pred.append(test_pred)
    labels = np.concatenate(labels)
    pred = np.concatenate(pred)
    if pred_scores is not None:
        ts_scores = np.concatenate(ts_scores)

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
    if pred_scores is not None:
        # pred_scores = np.asarray(res_log_socres, np.float64)[0]
        # labels = np.asarray(labels_log, np.int64)[0]
        min_len1 = min(ts_scores.shape[0], labels.shape[0])
        results_f1_pa_k_10 = evaluate.evaluate(ts_scores[:min_len1], labels[:min_len1], k=10)
        # results_f1_pa_k_30 = evaluate.evaluate(pred, labels, k=30)
        results_f1_pa_k_50 = evaluate.evaluate(ts_scores[:min_len1], labels[:min_len1], k=50)
        results_f1_pa_k_90 = evaluate.evaluate(ts_scores[:min_len1], labels[:min_len1], k=90)

        eval_res['f1_pa_10'] = results_f1_pa_k_10['best_f1_w_pa']
        eval_res['f1_pa_50'] = results_f1_pa_k_50['best_f1_w_pa']
        eval_res['f1_pa_90'] = results_f1_pa_k_90['best_f1_w_pa']

    return eval_res




def np_shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


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


def eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay, is_multi=False, ucr_index=None):
    t = time.time()
    
    all_train_repr = {}
    all_test_repr = {}
    all_train_repr_wom = {}
    all_test_repr_wom = {}

    if is_multi:
        train_data = all_train_data
        test_data = all_test_data
        if test_data.shape[-1] > 2:
            re_t = test_data.shape[-1]
        else:
            re_t = 1
        full_repr = model.encode(
            np.concatenate([train_data, test_data]).reshape(1, -1, re_t),
            mask='mask_last',
            casual=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256
        ).squeeze()
        all_train_repr[0] = full_repr[:len(train_data)]  # (n_timestamps, repr-dims)
        all_test_repr[0] = full_repr[len(train_data):]  # (n_timestamps, repr-dims)

        full_repr_wom = model.encode(
            np.concatenate([train_data, test_data]).reshape(1, -1, re_t),
            casual=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256
        ).squeeze()
        all_train_repr_wom[0] = full_repr_wom[:len(train_data)]  # (n_timestamps, repr-dims)
        all_test_repr_wom[0] = full_repr_wom[len(train_data):]  # (n_timestamps, repr-dims)
    else:
        for k in all_train_data:
            train_data = all_train_data[k]
            test_data = all_test_data[k]

            full_repr = model.encode(
                np.concatenate([train_data, test_data]).reshape(1, -1, 1),
                mask='mask_last',
                casual=True,
                sliding_length=1,
                sliding_padding=200,
                batch_size=256
            ).squeeze()
            all_train_repr[k] = full_repr[:len(train_data)] # (n_timestamps, repr-dims)
            all_test_repr[k] = full_repr[len(train_data):] # (n_timestamps, repr-dims)

            full_repr_wom = model.encode(
                np.concatenate([train_data, test_data]).reshape(1, -1, 1),
                casual=True,
                sliding_length=1,
                sliding_padding=200,
                batch_size=256
            ).squeeze()
            all_train_repr_wom[k] = full_repr_wom[:len(train_data)] # (n_timestamps, repr-dims)
            all_test_repr_wom[k] = full_repr_wom[len(train_data):] # (n_timestamps, repr-dims)

            # print(np.shape(all_train_repr[k]))
            # print(np.shape(all_test_repr[k]))
            # print(np.shape(all_train_repr_wom[k]))
            # print(np.shape(all_test_repr_wom[k]))
            # print("#####################")
            # raise Exception('my personal exception!')
        
    res_log = []
    res_log_socres = []
    labels_log = []
    timestamps_log = []
    if is_multi:

        test_labels = all_test_labels
        test_timestamps = all_test_timestamps

        train_err = np.abs(all_train_repr_wom[0] - all_train_repr[0]).sum(axis=1)
        test_err = np.abs(all_test_repr_wom[0] - all_test_repr[0]).sum(axis=1)

        ma = np_shift(bn.move_mean(np.concatenate([train_err, test_err]), 21), 1)
        train_err_adj = (train_err - ma[:len(train_err)]) / ma[:len(train_err)]
        test_err_adj = (test_err - ma[len(train_err):]) / ma[len(train_err):]
        train_err_adj = train_err_adj[22:]

        thr = np.mean(train_err_adj) + 4 * np.std(train_err_adj)
        test_res = (test_err_adj > thr) * 1
        res_log_socres.append(test_err_adj)

        for i in range(len(test_res)):
            if i >= delay and test_res[i - delay:i].sum() >= 1:
                test_res[i] = 0

        res_log.append(test_res)
        labels_log.append(test_labels)
        timestamps_log.append(test_timestamps)
    else:
        for k in all_train_data:
            test_labels = all_test_labels[k]
            test_timestamps = all_test_timestamps[k]

            train_err = np.abs(all_train_repr_wom[k] - all_train_repr[k]).sum(axis=1)
            test_err = np.abs(all_test_repr_wom[k] - all_test_repr[k]).sum(axis=1)

            ma = np_shift(bn.move_mean(np.concatenate([train_err, test_err]), 21), 1)
            train_err_adj = (train_err - ma[:len(train_err)]) / ma[:len(train_err)]
            test_err_adj = (test_err - ma[len(train_err):]) / ma[len(train_err):]
            train_err_adj = train_err_adj[22:]

            thr = np.mean(train_err_adj) + 4 * np.std(train_err_adj)
            test_res = (test_err_adj > thr) * 1
            res_log_socres.append(test_err_adj)

            for i in range(len(test_res)):
                if i >= delay and test_res[i-delay:i].sum() >= 1:
                    test_res[i] = 0

            res_log.append(test_res)
            labels_log.append(test_labels)
            timestamps_log.append(test_timestamps)
    t = time.time() - t



    if is_multi:
        labels = np.asarray(labels_log, np.int64)[0]
        pred = np.asarray(res_log, np.int64)[0]
        # print("labels.shape = ", labels.shape, labels[:5])
        # print("pred.shape = ", pred.shape, pred[:5])



        events_pred = convert_vector_to_events(pred)
        events_gt = convert_vector_to_events(labels)

        Trange = (0, len(labels))

        # print("labels.shape = ", labels.shape, "pred.shape = ", pred.shape)
        # print("events_pred.shape = ", len(events_pred), ", events_gt.shape = ", len(events_gt), ", Trange = ", Trange)
        if ucr_index == 79 or ucr_index == 108 or ucr_index == 187 or ucr_index == 203:
            pred_scores = np.asarray(res_log_socres, np.float64)[0]

            # results_f1_pa_k_10 = evaluate.evaluate(pred_scores, labels, k=10)
            # # results_f1_pa_k_30 = evaluate.evaluate(pred, labels, k=30)
            # results_f1_pa_k_50 = evaluate.evaluate(pred_scores, labels, k=50)
            # results_f1_pa_k_90 = evaluate.evaluate(pred_scores, labels, k=90)

            labels, pred = adjustment(labels, pred)

            eval_res = {
                'f1': f1_score(labels, pred),
                'precision': precision_score(labels, pred),
                'recall': recall_score(labels, pred),
                "Affiliation precision": None,
                "Affiliation recall": None,
                "R_AUC_ROC": None,
                "R_AUC_PR": None,
                "VUS_ROC": None,
                "VUS_PR": None,
                'f1_pa_10': None,
                # 'results_f1_pa_k_10_th_w_pa': results_f1_pa_k_10['pa_f1_scores'],
                'f1_pa_50': None,
                # 'results_f1_pa_k_50_th_w_pa': results_f1_pa_k_50['pa_f1_scores'],
                'f1_pa_90': None,
                # 'results_f1_pa_k_90_th_w_pa': results_f1_pa_k_90['pa_f1_scores'],

                # 'results_f1_pa_k_10_wpa': f1_score(labels, results_f1_pa_k_10),
                # # 'results_f1_pa_k_10_th_w_pa': results_f1_pa_k_10['best_f1_th_w_pa'],
                # 'results_f1_pa_k_50_wpa': f1_score(labels, results_f1_pa_k_50),
                # # 'results_f1_pa_k_50_th_w_pa': results_f1_pa_k_50['best_f1_th_w_pa'],
                # 'results_f1_pa_k_90_wpa': f1_score(labels, results_f1_pa_k_90),
                # 'results_f1_pa_k_90_th_w_pa': results_f1_pa_k_90['best_f1_th_w_pa'],
            }
        else:

            affiliation = pr_from_events(events_pred, events_gt, Trange)
            vus_results = get_range_vus_roc(labels, pred, 100)  # default slidingWindow = 100

            pred_scores = np.asarray(res_log_socres, np.float64)[0]

            # print("pred_scores.shape = ", pred_scores.shape, labels.shape)
            # print("pred_scores.shape = ", pred_scores[:10])
            # print("labels.shape = ", labels[:10])

            results_f1_pa_k_10 = evaluate.evaluate(pred_scores, labels, k=10)
            # results_f1_pa_k_30 = evaluate.evaluate(pred, labels, k=30)
            results_f1_pa_k_50 = evaluate.evaluate(pred_scores, labels, k=50)
            results_f1_pa_k_90 = evaluate.evaluate(pred_scores, labels, k=90)

            labels, pred = adjustment(labels, pred)

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
                # 'results_f1_pa_k_10_th_w_pa': results_f1_pa_k_10['pa_f1_scores'],
                'f1_pa_50': results_f1_pa_k_50['best_f1_w_pa'],
                # 'results_f1_pa_k_50_th_w_pa': results_f1_pa_k_50['pa_f1_scores'],
                'f1_pa_90': results_f1_pa_k_90['best_f1_w_pa'],
                # 'results_f1_pa_k_90_th_w_pa': results_f1_pa_k_90['pa_f1_scores'],

                # 'results_f1_pa_k_10_wpa': f1_score(labels, results_f1_pa_k_10),
                # # 'results_f1_pa_k_10_th_w_pa': results_f1_pa_k_10['best_f1_th_w_pa'],
                # 'results_f1_pa_k_50_wpa': f1_score(labels, results_f1_pa_k_50),
                # # 'results_f1_pa_k_50_th_w_pa': results_f1_pa_k_50['best_f1_th_w_pa'],
                # 'results_f1_pa_k_90_wpa': f1_score(labels, results_f1_pa_k_90),
                # 'results_f1_pa_k_90_th_w_pa': results_f1_pa_k_90['best_f1_th_w_pa'],
            }
    else:
        # pred_scores = np.asarray(res_log_socres, np.float64)
        # print("pred_scores.shape = ", pred_scores.shape)
        # results_f1_pa_k_10 = evaluate.evaluate(pred_scores, labels, k=10)
        # # results_f1_pa_k_30 = evaluate.evaluate(pred, labels, k=30)
        # results_f1_pa_k_50 = evaluate.evaluate(pred_scores, labels, k=50)
        # results_f1_pa_k_90 = evaluate.evaluate(pred_scores, labels, k=90)

        eval_res = eval_ad_result(res_log, labels_log, timestamps_log, delay, pred_scores=res_log_socres)



    eval_res['infer_time'] = t
    return res_log, eval_res


def eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay):
    t = time.time()
    
    all_data = {}
    all_repr = {}
    all_repr_wom = {}
    for k in all_train_data:
        all_data[k] = np.concatenate([all_train_data[k], all_test_data[k]])
        all_repr[k] = model.encode(
            all_data[k].reshape(1, -1, 1),
            mask='mask_last',
            casual=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256
        ).squeeze()
        all_repr_wom[k] = model.encode(
            all_data[k].reshape(1, -1, 1),
            casual=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256
        ).squeeze()
        
    res_log = []
    labels_log = []
    timestamps_log = []
    for k in all_data:
        data = all_data[k]
        labels = np.concatenate([all_train_labels[k], all_test_labels[k]])
        timestamps = np.concatenate([all_train_timestamps[k], all_test_timestamps[k]])
        
        err = np.abs(all_repr_wom[k] - all_repr[k]).sum(axis=1)
        ma = np_shift(bn.move_mean(err, 21), 1)
        err_adj = (err - ma) / ma
        
        MIN_WINDOW = len(data) // 10
        thr = bn.move_mean(err_adj, len(err_adj), MIN_WINDOW) + 4 * bn.move_std(err_adj, len(err_adj), MIN_WINDOW)
        res = (err_adj > thr) * 1
        
        for i in range(len(res)):
            if i >= delay and res[i-delay:i].sum() >= 1:
                res[i] = 0

        res_log.append(res[MIN_WINDOW:])
        labels_log.append(labels[MIN_WINDOW:])
        timestamps_log.append(timestamps[MIN_WINDOW:])
    t = time.time() - t
    
    eval_res = eval_ad_result(res_log, labels_log, timestamps_log, delay)
    eval_res['infer_time'] = t
    return res_log, eval_res

