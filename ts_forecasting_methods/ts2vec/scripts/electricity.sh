# multivar
for seed in $(seq 0 4 11 22 43); do
  python -u train.py electricity forecast_multivar --loader forecast_csv --repr-dims 320 --max-threads 8 --seed ${seed} --eval
done

