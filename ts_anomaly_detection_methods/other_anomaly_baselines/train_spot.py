import numpy as np
import matplotlib.pyplot as plt
from spot import SPOT
import numpy as np
import time
import datetime
import datautils
from sklearn.metrics import f1_score, precision_score, recall_score

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


dataset = 'kpi' # yahoo, kpi
print('Loading data... ', end='')
all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(dataset)

labels = []
pred = []
for k in all_test_data:
	train_data = all_train_data[k] # initial batch
	train_labels = all_train_labels[k] 
	train_timestamps = all_train_timestamps[k]

	test_data = all_test_data[k] # stream
	test_labels = all_test_labels[k]
	test_timestamps = all_test_timestamps[k]

	q = 1e-3  			# risk parameter
	s = SPOT(q)  		# SPOT object
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

labels = np.concatenate(labels)
pred = np.concatenate(pred)

print('\nf1:', f1_score(labels, pred))
print('precision:', precision_score(labels, pred))
print('recall:', recall_score(labels, pred))

print("Finished.")