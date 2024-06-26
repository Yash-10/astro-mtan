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

`visualization_outputs_w_periodic_time_embedding_ldim2_timeHrs_testset_timeNormModification.ipynb` is the same as `visualization_outputs_w_periodic_time_embedding_ldim2_timeHrs_testset.ipynb`. But the modification is that the results shown are after modifying the time normalization: we still use global time normalization, but instead of normalizing based on min/max time across the entire dataset (train+val+test), we normalize each dataset based on its own statistics (i.e., normalize train lcs times based on train times' min/max, normalize val lcs times based on val times' min/max, etc). The results, however, don't change compared to before, but this change is probably better since previously, information from val and test about the min/max times was leaked into normalizing the train lcs.

For this data:
```
Max and Min time values (in hrs) across the train dataset: 8616.086112007499, 0.0
Max and Min time values (in hrs) across the val dataset: 8544.609719999135, 0.0
Max and Min time values (in hrs) across the test dataset: 8615.560000799596, 0.0
```

so for all three types of data, the max time is very similar, and the min time is the same. This may be the reason why the performance didn't change much, even if time normalization is done for each separately. For different datasets, this observation of performance not changing as time normalization is changed like this may not hold.

---

- `visualization_outputs_w_periodic_time_embedding_ldim2_timeHrs_testset_latest.ipynb` is the latest notebook. The following changes are made compared to the latest notebook before this:
    - The data is more reliable: ZTF18... object IDs with one of the TNS SN classes are removed. A relatively more recent data range (June 1, 2021, to June 1, 2022) is selected to minimize the chances of removing many such transient light curves. Overall, 5604 lcs for train, 1402 for val, and 1752 for test are present.
    - The notebook itself is updated considerably, with more extensive analysis. Some additions include refinement of the definition of the class lists for SN and AGN (reflecting the changes made in the scripts), `faiss` to perform query by example (similarity search) over the encoded representations, more investigation of the extreme data points in the two-dimensional projection plots of the encoded representations, visualization of attention weights to interpret the model, etc.

For this case, AGN interpolation examples can be found [here](https://github.com/Yash-10/fast_transients/blob/dbc76117880c9634e068c171e1381f025845f8f6/tnsSN_earlySNIacand_%26_custom_agn/visualization_outputs_w_periodic_time_embedding_ldim2_timeHrs_testset_latest.ipynb), whereas SN interpolations [here](https://github.com/Yash-10/fast_transients/blob/main/tnsSN_earlySNIacand_%26_custom_agn/visualization_outputs_w_periodic_time_embedding_ldim2_timeHrs_testset_latest.ipynb) [**NOTE** TODO: SN interpolations are temporarily not there since I updated the latest commit of notebook -- written on June 26]. The former notebook is slightly different than the latter (the latter is the latest), but both use the same data and the same trained model, just that the former shows AGN decoded light curves and the latter shows SN decoded light curves + small improvements or bug fixing in the code, which shouldn't drastically affect the interpretation of the results.

For this, the details are below:
```
ftransfer_ztf_2024-05-27_433174_copy

Max and Min time values (in hrs) across the train dataset: 8763.185277599841, 0.0

Max and Min time values (in hrs) across the val dataset: 8762.649167999625, 0.0
Max and Min time values (in hrs) across the test dataset: 8763.17888879031, 0.0
```
