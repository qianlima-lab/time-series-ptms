import numpy as np
import time
from . import _eval_protocols as eval_protocols

def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])

def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean()
    }
    
def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols):
    padding = 200
    
    t = time.time()
    all_repr = model.encode(
        data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    ts2vec_infer_time = time.time() - t

    print("all_repr.shape = ", all_repr.shape)
    
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    print("train_repr.shape = ", train_repr.shape, ", valid_repr.shape = ", valid_repr.shape, ", test_repr.shape = ", test_repr.shape)

    print("n_covariate_cols = ", n_covariate_cols)
    
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]

    print("train_data.shape = ", train_data.shape, ", valid_data.shape = ", valid_data.shape, ", test_data.shape = ",
          test_data.shape)

    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)
        
        t = time.time()
        lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t
        
        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)
        
        # if test_data.shape[0] > 1:
        #     test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
        #     test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
        # else:
        #     print("test_pred.shape = ", test_pred.shape, ", test_labels.shape = ", test_labels.shape)
        #     print("test_pred.swapaxes(0, 3).shape = ", test_pred.swapaxes(0, 3).shape)
        #     test_pred_inv = scaler.inverse_transform(test_pred)
        #     test_labels_inv = scaler.inverse_transform(test_labels)
            
        out_log[pred_len] = {
            'norm': test_pred,
            # 'raw': test_pred_inv,
            'norm_gt': test_labels,
            # 'raw_gt': test_labels_inv
        }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
            # 'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }
        
    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res


def eval_forecasting_new(model, train_data, valid_data, test_data, pred_lens):
    padding = 200

    t = time.time()
    train_repr = model.encode(
        train_data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    valid_repr = model.encode(
        valid_data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    test_repr = model.encode(
        test_data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    ts2vec_infer_time = time.time() - t

    print("train_data.shape = ", train_data.shape)

    # train_repr = all_repr[:, train_slice]
    # valid_repr = all_repr[:, valid_slice]
    # test_repr = all_repr[:, test_slice]
    # print("train_repr.shape = ", train_repr.shape, ", valid_repr.shape = ", valid_repr.shape, ", test_repr.shape = ",
    #       test_repr.shape)
    #
    # print("n_covariate_cols = ", n_covariate_cols)
    #
    # train_data = data[:, train_slice, n_covariate_cols:]
    # valid_data = data[:, valid_slice, n_covariate_cols:]
    # test_data = data[:, test_slice, n_covariate_cols:]

    print("train_data.shape = ", train_data.shape, ", valid_data.shape = ", valid_data.shape, ", test_data.shape = ",
          test_data.shape)

    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)

        t = time.time()
        lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t

        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)

        # if test_data.shape[0] > 1:
        #     test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
        #     test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
        # else:
        #     print("test_pred.shape = ", test_pred.shape, ", test_labels.shape = ", test_labels.shape)
        #     print("test_pred.swapaxes(0, 3).shape = ", test_pred.swapaxes(0, 3).shape)
        #     test_pred_inv = scaler.inverse_transform(test_pred)
        #     test_labels_inv = scaler.inverse_transform(test_labels)

        out_log[pred_len] = {
            'norm': test_pred,
            # 'raw': test_pred_inv,
            'norm_gt': test_labels,
            # 'raw_gt': test_labels_inv
        }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
            # 'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }

    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res
