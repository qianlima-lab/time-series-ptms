python train_gpt4ts_uni.py --anomaly_ratio 1 --data kpi --save_csv_name train_gpt4ts_uni_hm_0720.csv --gpu 1;
python train_gpt4ts_uni.py --anomaly_ratio 1 --data yahoo --save_csv_name train_gpt4ts_uni_hm_0720.csv --gpu 0;
python train_timesnet_uni.py --anomaly_ratio 1 --data kpi --save_csv_name train_timesnet_uni_hm_0720.csv --gpu 0;
python train_timesnet_uni.py --anomaly_ratio 1 --data yahoo --save_csv_name train_timesnet_uni_hm_0720.csv --gpu 0;
python train_dcdetector_nui.py --anormly_ratio 1 --dataset kpi --save_csv_name train_dcdetector_nui_hm_0720.csv --gpu 0;
python train_dcdetector_nui.py --anormly_ratio 1 --dataset yahoo --save_csv_name train_dcdetector_nui_hm_0720.csv --gpu 1;
