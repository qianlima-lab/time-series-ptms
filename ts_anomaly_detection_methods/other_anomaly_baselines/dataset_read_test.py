import datautils
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


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


def eval_ad_result(test_pred_list, test_labels_list, test_timestamps_list, delay):
    labels = []
    pred = []
    for test_pred, test_labels, test_timestamps in zip(test_pred_list, test_labels_list, test_timestamps_list):
        assert test_pred.shape == test_labels.shape == test_timestamps.shape
        test_labels = reconstruct_label(test_timestamps, test_labels)
        test_pred = reconstruct_label(test_timestamps, test_pred)
        test_pred = get_range_proba(test_pred, test_labels, delay)
        labels.append(test_labels)
        pred.append(test_pred)
    labels = np.concatenate(labels)
    pred = np.concatenate(pred)
    return {
        'f1': f1_score(labels, pred),
        'precision': precision_score(labels, pred),
        'recall': recall_score(labels, pred)
    }


dataset = 'kpi' # yahoo, kpi
print('Loading kpi data... ', end='')
all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(dataset)

print("type = ", type(all_train_data), type(all_train_labels), type(all_train_timestamps), type(all_test_data))
print("delay = ", delay)
i = 1
for k in all_test_data:
    print("i = ", i, ", k = ", k)
    print("all_train_data.shape = ", all_train_data[k].shape)
    print("all_train_labels.shape = ", all_train_labels[k].shape)
    print("all_train_timestamps.shape = ", all_train_timestamps[k].shape)
    print("all_test_data.shape = ", all_test_data[k].shape)
    print("all_test_labels.shape = ", all_test_labels[k].shape)
    print("all_test_timestamps.shape = ", all_test_timestamps[k].shape)
    print("all_train_labels[k][:10] = ", all_train_labels[k][:10])
    print("all_test_timestamps[k][:10] = ", all_test_timestamps[k][:10])
    i = i + 1
    break


# dataset = 'yahoo' # yahoo, kpi
# print('Loading yahoo data... ', end='')
# all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(dataset)
#
# print("type = ", type(all_train_data), type(all_train_labels), type(all_train_timestamps), type(all_test_data))
# print("delay = ", delay)
# i = 1
# for k in all_test_data:
#     print("i = ", i, ", k = ", k)
#     print("all_train_data.shape = ", all_train_data[k].shape)
#     print("all_train_labels.shape = ", all_train_labels[k].shape)
#     print("all_train_timestamps.shape = ", all_train_timestamps[k].shape)
#     print("all_test_data.shape = ", all_test_data[k].shape)
#     print("all_test_labels.shape = ", all_test_labels[k].shape)
#     print("all_test_timestamps.shape = ", all_test_timestamps[k].shape)
#     i = i + 1