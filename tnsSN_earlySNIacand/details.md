```bash
python3 tan_unsupervised.py --niters 200 --lr 0.001 --rec-hidden 64 --latent-dim 1 --enc mtan_rnn --dec mtan_rnn --gen-hidden 50 --learn-emb --kl --seed 42 --num-ref-points 160 --dataset ftransfer_ztf_2024-04-26_572037_copy --sample-tp 1.0 --save 1 --k-iwae 5 --std 0.01 --norm --topic ftransfer_ztf_2024-04-26_572037_copy --dim 2 --embed-time 128
```

Trained for 200 epochs. See above for training hyperparams.

Best model at 102th epoch, based on val set performance. 1126 lcs for training, and 282 for val. Looking at the performance on the train set itself after training (i.e., evaluation on the train set). MSE is used for validation.

Alerts: from Dec 1, 2019 to Dec 1, 2021 and using only the four criteria from the suggested filters of ZTF Avro. See https://zwickytransientfacility.github.io/ztf-avro-alert/filtering.html. Only alerts with lcs >= 10pts and >=4 pts in each band are selected.

