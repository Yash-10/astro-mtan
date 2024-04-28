Things common for all notebooks:
- Alerts: from Dec 1, 2019 to Dec 1, 2021 and using only the four criteria from the suggested filters of ZTF Avro. See https://zwickytransientfacility.github.io/ztf-avro-alert/filtering.html. Only alerts with lcs >= 10pts and >=4 pts in each band are selected.
- NOTE: Even though the notebook shows finkclasses apart from (TNS) SN and early SN Ia candidate, that is fine since TNS SN classifications are only present in the `tnsclass` column of the alerts, not the `finkclass`. So even though the finkclass can be different from these two, they were included because they had a TNS SN identification. The TNS class histogram in the notebook makes it clear.
- Best model selected based on val set performance. 1126 lcs for training, and 282 for val. All notebooks look at the performance on the train set itself after training (i.e., evaluation on the train set). MSE is used for validation on the val set. Trained for 200 epochs.

`visualization_outputs.ipynb` and `visualization_outputs_without_periodic_time_embedding.ipynb` use the below train code:
```bash
python3 tan_unsupervised.py --niters 200 --lr 0.001 --rec-hidden 64 --latent-dim 1 --enc mtan_rnn --dec mtan_rnn --gen-hidden 50 --learn-emb --kl --seed 42 --num-ref-points 160 --dataset ftransfer_ztf_2024-04-26_572037_copy --sample-tp 1.0 --save 1 --k-iwae 5 --std 0.01 --norm --topic ftransfer_ztf_2024-04-26_572037_copy --dim 2 --embed-time 128
```
The only difference in the latter is that periodic time embedding is turned off and replaced with the normal time embedding.

`visualization_outputs_w_periodic_time_embedding_ldim2_timeHrs.ipynb` uses the below training code:

```bash
python3 tan_unsupervised.py --niters 200 --lr 0.001 --rec-hidden 64 --latent-dim 2 --enc mtan_rnn --dec mtan_rnn --gen-hidden 50 --learn-emb --kl --seed 42 --num-ref-points 160 --dataset ftransfer_ztf_2024-04-26_572037_copy --sample-tp 1.0 --save 1 --k-iwae 5 --std 0.01 --norm --topic ftransfer_ztf_2024-04-26_572037_copy --dim 2 --embed-time 128
```
Compared to `visualization_outputs.ipynb`, changes are: latent_dim = 2, lc times are in hours instead of days (in an attempt to reduce the steepness in the rise and fall around the peak and hence improve interpolation of region around the peak). Uses 160 ref points as all above two experiments.
