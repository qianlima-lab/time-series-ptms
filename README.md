# From Transfer to Transformer

This is the training code for our paper *"From Transfer to Transformer: A Survey on
Pre-Training Time-series Models"*


## Usage
1. To pretrain a model on your own dataset, run

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
for detailed options

## Results
### Transfer learning in UCR datasets

![FCN](png/fcn_finetuning.png "FCN Fine-tuning accuracy")![FCN-RNN](/png/fcn_rnn_finetuning.png "FCN-RNN fine-tuning accuracy")

<p align="center">*FCN and FCN-RNN fine-tuning accuracy*</p>

![Dilated3CNN](/png/dilated3cnn_finetuning.png "Dilated-3 CNN Fine-tuning accuracy")

![Dilated3CNN](/png/dilated3cnn_rnn_finetuning.png "Dilated-3 CNN-RNN fine-tuning accuracy")
<center>Dilated-3 CNN and  Dilated-3 CNN-RNN fine-tuning accuracy</center>