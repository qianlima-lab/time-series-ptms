# From Transfer to Transformer

This is the training code for our paper *"From Transfer to Transformer: A Survey on
Time-series Pre-Training Models"*


## Usage (Transfer Learning)
1. To pretrain (options: classification or reconstruction) a model on your own dataset, run

```bash
python train.py --dataroot [your UCR datasets directory] --dataset [name of the dataset you want to pretrain on] --backbone [fcn or dilated] --mode pretrain ...
```

2. To finetune the model on a dataset, run

```bash
python train.py --dataroot [your UCR datasets directory] --dataset [name of the dataset you want to finetune on] --source_dataset [the dataset you pretrained on] --save_dir [the directory to save the pretrained weights] --mode finetune ...

```

run 
```bash 
python train.py -h
```

For detailed options and examples, please refer to ```scripts/transfer_pretrain_finetunse.sh```

## Usage (Transformer and Contrastive Learning)
1. Pre-training and classification using TS2Vec model on a UCR dataset, run
```bash 
python train_tsm.py --dataroot [your UCR datasets directory] --normalize_way single ...
```

For detailed options and examples, please refer to ```ts2vec_cls/scripts/ts2vec_tsm_single_norm.sh```


```
@article{yue2022ts2vec,
  title={TS2Vec: Towards Universal Representation of Time Series},
  author={Yue, Zhihan and Wang, Yujing and Duan, Juanyong and Yang, Tianmeng and Huang, Congrui and Tong, Yunhai and Xu, Bixiong},
  journal={Proceedings of AAAI Conference on Artificial Intelligence},
  year={2022}
}
```

2. Pre-training and classification using TSTCC model on a UCR dataset, run
```bash 
python main_ucr.py --dataset [name of the ucr dataset] --device cuda:0 --save_csv_name tstcc_ucr_ --seed 42;
```

For detailed options and examples, please refer to ```tstcc_cls/scripts/fivefold_tstcc_ucr.sh```


```
@inproceedings{ijcai2021-324,
  title     = {Time-Series Representation Learning via Temporal and Contextual Contrasting},
  author    = {Eldele, Emadeldeen and Ragab, Mohamed and Chen, Zhenghua and Wu, Min and Kwoh, Chee Keong and Li, Xiaoli and Guan, Cuntai},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, {IJCAI-21}},
  pages     = {2352--2359},
  year      = {2021},
}
```


## Results
### Transfer learning in UCR datasets
![Encoder_cls](png/encoder_cls_results.png "Classification test accuracy using FCN and Dilated CNN on 128 UCR datasets")

![Transfer](png/transfer_learning_results.png "Comparison of pre-training methods based on transfer learning")