for seed in $(seq 0 4); do
  # multivar
  python -u train.py ETTh1 forecast_multivar --loader forecast_csv --repr-dims 320 --max-threads 8 --seed ${seed} --eval
  python -u train.py ETTh2 forecast_multivar --loader forecast_csv --repr-dims 320 --max-threads 8 --seed ${seed} --eval
  python -u train.py ETTm1 forecast_multivar --loader forecast_csv --repr-dims 320 --max-threads 8 --seed ${seed} --eval
done