import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from utils import data_slice
import datautils
import pdb
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score
import tasks
from ATmodelbatch import AnomalyTransformer
import time
import bottleneck as bn
import argparse
import os
import pickle

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type('torch.DoubleTensor')

logger = logging.getLogger(__name__)


class Config:
	
    window_size=100
    shuffle=True
    epochs=500
    warmup_ratio= 0.1
    lr= 10e-4
    adam_epsilon= 1e-6
    batch_size = 512
    
    in_channel=1
    dataset_name = "kpi"
    d_model=512
    layers=3
    lambda_=3
    
    save_dir = './save_models'
    save_every_epoch = 2
    
    is_train=True
    is_eval=True


def train(config,model,all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay):
    
    train_data = datautils.gen_ano_train_data(all_train_data)
    config.in_channel = train_data.shape[-1]
    train_data = data_slice(train_data,config.window_size)
    train_data = torch.from_numpy(train_data)
    
    if torch.cuda.is_available():
        train_data = train_data.cuda()
        
    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=min(config.batch_size, len(train_dataset)), shuffle=config.shuffle, drop_last=True,generator=torch.Generator(device='cuda:0'))

    total_steps = int(len(train_dataloader) * config.epochs)
    warmup_steps = max(int(total_steps * config.warmup_ratio), 200)
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        eps=config.adam_epsilon,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print("Total steps: {}".format(total_steps))
    print("Warmup steps: {}".format(warmup_steps))


    for epoch in range(int(config.epochs)):
        print(epoch)
        if (epoch+1) % config.save_every_epoch == 0:
            path = config.save_dir+'/'+model.to_string()+'_epoch:%d' % (epoch+1)
            os.makedirs(path,exist_ok=True)
            torch.save(model,path+'/model.pt')
            pdb.set_trace()
            f1,pre,recall = evaluate(config,epoch+1,model,all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
            print('epoch:%d\tf1:%f\tp:%f\tr:%f' % (epoch+1,f1,pre,recall))
            
        model.zero_grad()
        for step, batch in enumerate(train_dataloader):
            batch=batch[0]
            model(batch)
            min_loss = model.min_loss(batch)
            max_loss = model.max_loss(batch)
            #print('minloss:%f\tmaxloss:%f' % (min_loss.detach().cpu(),max_loss.detach().cpu()))
            optimizer.zero_grad()
            min_loss.backward(retain_graph=True)
            max_loss.backward()
            optimizer.step()
            scheduler.step()
            

def evaluate(config,cur_epoch,model,all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay):
    res_log = []
    labels_log = []
    timestamps_log = []
    t = time.time()
    for k in all_train_data:
        train_data = all_train_data[k]
        train_labels = all_train_labels[k]
        train_timestamps = all_train_timestamps[k]
        train_length = train_labels.shape[0]

        test_data = all_test_data[k]
        test_labels = all_test_labels[k]
        test_timestamps = all_test_timestamps[k]
        test_length = test_labels.shape[0]
        
        train_err = model.anomaly_score_whole(train_data).detach().cpu().numpy()
        test_err = model.anomaly_score_whole(test_data).detach().cpu().numpy()
        
        train_err = train_err[:train_length]
        test_err = test_err[:test_length]
        
        ma = tasks.np_shift(bn.move_mean(np.concatenate([train_err, test_err]), 21), 1)
        train_err_adj = (train_err - ma[:len(train_err)]) / ma[:len(train_err)]
        test_err_adj = (test_err - ma[len(train_err):]) / ma[len(train_err):]
        train_err_adj = train_err_adj[22:]

        thr = np.mean(train_err_adj) + 4 * np.std(train_err_adj)
        test_res = (test_err_adj > thr) * 1

        for i in range(len(test_res)):
            if i >= delay and test_res[i-delay:i].sum() >= 1:
                test_res[i] = 0
        res_log.append(test_res)
        labels_log.append(test_labels)
        timestamps_log.append(test_timestamps)
        
    t = time.time() - t
    eval_res = tasks.eval_ad_result(res_log, labels_log, timestamps_log, delay)
    eval_res['infer_time'] = t
    '''
    eval_res:{'f1':,'p':,'r':,}
    '''
    '''save_results'''
    path = config.save_dir+'/'+model.to_string()+'_epoch:%d' % (cur_epoch)
    os.makedirs(path,exist_ok=True)
    with open(path+'/res_log.pkl','wb') as f:
        pickle.dump(res_log,f)
    with open(path+'/eval_res.pkl','wb') as f:
        pickle.dump(eval_res,f)
    with open(path+'/results.txt','w') as f:
        f.write('f1:%f\tp:%f\tr:%f\n' % (eval_res['f1'],eval_res['precision'],eval_res['recall']))
            
    return eval_res['f1'],eval_res['precision'],eval_res['recall']

def main(config):
    
    all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(config.dataset_name)

    
    print('data loaded!')
    model = AnomalyTransformer(config.batch_size,config.window_size,config.in_channel,config.d_model,config.layers,config.lambda_)
    print('model builded!')
    print('train start!')
    if config.is_train:
        model.train()
        train(config,model,all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        '''save_trained_model'''
        path = config.save_dir+'/'+model.to_string()+'_epoch:%d' % (config.epochs)
        os.makedirs(path,exist_ok=True)
        torch.save(model,path+'/model.pt')
    
    print('train finished! evaluating...')
    if config.is_eval:
        model.eval()
        res_log,eval_res = evaluate(config,config.epochs,model,all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)

        
    print('evaluate finished!')
    
if __name__ == "__main__":
    config = Config()
    main(config)
