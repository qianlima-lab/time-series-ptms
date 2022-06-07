python train.py kpi anomaly_0 --loader anomaly --repr-dims 320 --gpu 0 --seed 42 --eval

python train_donut.py kpi anomaly_0 --loader anomaly --gpu 0 --seed 42 --eval

python train_lstm_vae.py kpi anomaly_0 --loader anomaly --gpu 0 --seed 42 --eval
