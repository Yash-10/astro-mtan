Run the script `main_preprocessing.py`. This will preprocess the alert folders, prepare the data, split it into train-val-test sets, and save the train and test dataloader as files.
Then run the `tan_unsupervised.py` scripts which runs the actual training. (This needs to be run as a job using train.sh).
Then run `evaluate_unsupervised.py` which will evaluate the model on the test set and store the outputs. 

ftransfer_ztf_2024-04-26_572037 is only SN
	By SN, I mean the following: those alerts are selected from the data transfer service that are either Early SN Ia finkclass or one of TNS SN. After polling, alerts will be grouped based on finkclass in diff directories. I choose only dir == 'SN' or dir == 'SN%20candidate' alerts. After all this yields "custom_sn". Now I combine all such alerts with those having a finkclass "Early SN Ia candidate" (which is in a spearate, third directory). 

ftransfer_ztf_2024-05-27_569011 is SN+AGN

ftransfer_ztf_2024-04-17_832975 is all combined (no preselections on classes)

See notes.txt for data range.
