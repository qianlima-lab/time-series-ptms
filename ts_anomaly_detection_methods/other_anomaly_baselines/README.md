## README_Anomaly_Detection

### Usage

|  ID  |                            Method                            | Year |   Press   |                         Source Code                          |
| :--: | :----------------------------------------------------------: | :--: | :-------: | :----------------------------------------------------------: |
|  1   |  [SPOT](https://dl.acm.org/doi/abs/10.1145/3097983.3098144)  | 2017 |    KDD    |     [github_link](https://github.com/Amossys-team/SPOT)      |
|  2   | [DSPOT](https://dl.acm.org/doi/abs/10.1145/3097983.3098144)  | 2017 |    KDD    |     [github_link](https://github.com/Amossys-team/SPOT)      |
|  3   | [LSTM-VAE](https://ieeexplore.ieee.org/abstract/document/8279425) | 2018 | IEEE RA.L | [github_link](https://github.com/SchindlerLiang/VAE-for-Anomaly-Detection) |
|  4   | [DONUT](https://dl.acm.org/doi/abs/10.1145/3178876.3185996)  | 2018 |    WWW    |     [github_link](https://github.com/NetManAIOps/donut)      |
|  5   |  [SR*](https://dl.acm.org/doi/abs/10.1145/3292500.3330680)   | 2019 |    KDD    |                              -                               |
|  6   |            [AT](https://arxiv.org/abs/2110.02642)            | 2022 |   ICLR    | [github_link](https://github.com/spencerbraun/anomaly_transformer_pytorch) |
|  7   | [TS2Vec](https://www.aaai.org/AAAI22Papers/AAAI-8809.YueZ.pdf) | 2022 |   AAAI    |      [github_link](https://github.com/yuezhihan/ts2vec)      |
1. To train and evaluate SPOT/DSPOT on a dataset, set the dataset_name `dataset='yahoo' or 'kpi'`, and then run the following command:

   ```python
   python train_spot.py
   python train_dspot.py
   ```

2. To train and evaluate LSTM-VAE on a dataset, run the following command:

   ```python
   python train_lstm_vae.py <dataset_name> <run_name> --loader <loader> --gpu <gpu_device_id> --seed 42 --eval
   ```

    `dataset_name`: The dataset name.

    `run_name`: The folder name used to save model, output and evaluation metrics. This can be set to any word.

    `loader`: The data loader used to load the experimental data.

    `gpu_device_id`: The GPU device's ID. This can be  `0,1,2...`

3. To train and evaluate DONUT on a dataset, run the following command:

   ```python
   python train_donut.py <dataset_name> <run_name> --loader <loader> --gpu <gpu_device_id> --seed 42 --eval
   ```

4. The anomaly detection results of the SR are collected from the original [SR](https://dl.acm.org/doi/abs/10.1145/3292500.3330680) article.

5. To train and evaluate AT on a dataset,  set hyper_parameters in the file  `trainATbatch.py` , and then run the following command:

   ```python
   python trainATbatch.py
   ```

6. To train and evaluate TS2Vec on a dataset, run the following command:

   ```python
   python train_ts2vec.py <dataset_name> <run_name> --loader <loader> --repr-dims 320 --gpu <gpu_device_id> --seed 42 --eval
   ```

For detailed options and examples, please refer to `ts_anomaly_detection_methods/other_anomaly_baselines/scripts/ucr.sh`