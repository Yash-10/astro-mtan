Things common for all notebooks:
- Alerts: from Dec 1, 2020, to Dec 1, 2021, and using only the five criteria from the suggested filters of ZTF Avro. See https://zwickytransientfacility.github.io/ztf-avro-alert/filtering.html. Only alerts with lcs >= 10pts and >=4 pts in each band are selected. Parallel-lcs-same-band examples are removed; currently, there are 10 such examples (see constants.py).
- The same SN+EarlySNIacand examples as in the only-SN experiment are used, so refer to `tnsSN_earlySNIacand` for details about that. AGN data includes all entries from `custom_agn` in constants.py and were selected on the data transfer service while polling the alerts.
- Best model selected based on val set performance. 6684 lcs for training, and 1672 for val and 2089 for test. All notebooks look at the performance on the train set after training (i.e., evaluation on the train set). MSE is used for validation on the val set. Trained for 200 epochs, but the best model was at the 10th epoch only.

`visualization_outputs_w_periodic_time_embedding_ldim2_timeHrs.ipynb` uses the below training code:

```bash
python3 tan_unsupervised.py --niters 200 --lr 0.001 --rec-hidden 64 --latent-dim 2 --enc mtan_rnn --dec mtan_rnn --gen-hidden 50 --learn-emb --kl --seed 42 --num-ref-points 160 --dataset ftransfer_ztf_2024-05-20_608105_copy --sample-tp 1.0 --save 1 --k-iwae 5 --std 0.01 --norm --topic ftransfer_ztf_2024-05-20_608105_copy --dim 2 --embed-time 128
```

`visualization_outputs_w_periodic_time_embedding_ldim2_timeHrs_testset.ipynb` is the same as above, except I use the test to evaluate the performance. This is necessary because, ultimately, we want to evaluate performance on data unseen during training. Evaluating on train set was only as a validation check to ensure the model can learn interpolations on the train set (i.e., data on which it was trained).

`visualization_outputs_w_periodic_time_embedding_ldim2_timeHrs_testset_onlySNvisualize.ipynb` is same as `visualization_outputs_w_periodic_time_embedding_ldim2_timeHrs_testset.ipynb`, but only custom_sn examples are shown purposefully. This was done because AGN dominated the SN+AGN data, and the interpolations were good on AGN. We wanted to see if interpolations are also good for SN when training on SN+AGN combined. The previous notebooks described above didn't have much (or even any) visualization for custom_sn decoded lcs.
