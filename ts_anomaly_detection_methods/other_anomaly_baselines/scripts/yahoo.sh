python train.py yahoo anomaly_0 --loader anomaly --repr-dims 320 --gpu 0 --seed 42 --eval

python train_donut.py yahoo anomaly_0 --loader anomaly --gpu 0 --seed 42 --eval

python train_lstm_vae.py yahoo anomaly_0 --loader anomaly --gpu 0 --seed 42 --eval
