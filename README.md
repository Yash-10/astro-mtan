# mTAN for astronomical light curves

Code implementing the Multi-Time Attention Network (mTAN) for application on astronomical alert light curves. This repository is a modified clone of the original mTAN code: [`reml-lab/mTAN`](https://github.com/reml-lab/mTAN). Here we focus on Supernovae (SNe) and AGN light curves for our main analysis, but also showcase generalization to unseen classes (see [Evaluation on classes unseen during training](https://github.com/Yash-10/astro-mtan#evaluation-on-classes-unseen-during-training)). This code accompanies our paper: TODO

![mTAND-schema](https://github.com/Yash-10/astro-mtan/blob/main/mTAND_schema.png)


## Installation

1. Clone the repository:

```bash
git clone git@github.com:Yash-10/astro-mtan.git
cd astro-mtan
```

2. Create and activate a Python environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install packages:

```bash
pip install torch numpy pandas scikit-learn matplotlib dask george astropy scipy requests
```
and optionally
```bash
pip install wandb
```
(although the use of wandb is not tested)

## Usage

### 1. Preprocess raw alert folders

Run the dataset preparation flow:

```bash
python main_preprocessing.py
```

- This will preprocess the alert folders, prepare the data, split it into train-val-test sets, and save the train and test dataloader as files, along with some other things required for training the model (step 2 below).
- Set `TOPIC_PATH` to your alert folder path; the alert folder is obtained from [Fink's Data Transfer Service for ZTF](https://ztf.fink-portal.org/download). Details of all parameters used for polling our alerts are mentioned in our paper.


### 2. Train the unsupervised model

Use the SLURM script:

```bash
sbatch train.sh
```

Or run directly. An example usage is below:

```bash
python3 tan_unsupervised.py --niters 300 --lr 0.0001 --rec-hidden 64 --latent-dim 2 --enc mtan_rnn --dec mtan_rnn --gen-hidden 50 --learn-emb --kl --seed 42 --dataset ftransfer_ztf_2025-05-31_430518 --sample-tp 1.0 --save 1 --k-iwae 5 --std 0.01 --norm --topic ftransfer_ztf_2025-05-31_430518 --dim 2 --embed-time 128 --ref-resolution-days 2 --train_val_test_min_max_times_filename train_val_test_min_max_times.npy
```

### 3. Evaluate model outputs

```bash
sbatch evaluate.sh
```

Or directly:

```bash
python3 evaluate_unsupervised.py
```
- Note: there is no argparsing implemented for this script. Instead, the hyperparameter values used in our work are hardcoded inside the script like below:
```python
ref_resolution_days = 2
latent_dim = 2
learn_emb = True
rec_hidden = 64
gen_hidden = 50
...
```
- Set `SETTING` to an appropriate string value; available options are commented inside the python script itself. `SETTING = "test"` leads to the test results without any light curve correction applied before evaluation (used for SNe light curves) and `SETTING = "test_CORRECTED"` leads to the test results with light curve correction applied before evaluation (used for AGN light curves).

**Details on light curve correction**
- Light curve correction is not applied before training in our code, but only before evaluation, because the decision to correct light curves was made after training was completed. We needed to perform a separate polling of alerts from the Fink Data Transfer service (but with the same parameters as the original poll) to get additional fields required for correcting the light curves that were not present in our original poll. This created a new "corrected" version of the dataset. Although all light curves are corrected in this new data, for our analysis in `results.ipynb` we use the original (uncorrected) alerts for SN but use the new (corrected) alerts for AGN. Ideally, this correction can be applied before training too, which can be more consistent, but we didn't find the results to degrade even if correction was only applied during testing compared to the case of no correction for any light curve.

We performed the correction of the newly polled alerts using:

```bash
sbatch correct_mags.sh
```
or directly running

```python
python3 correct_mags.py
```

but given the above discussion, the correction can ideally be incorporated within code in step 1 itself had our original polling of alerts included all the alert fields.

- The light curve correction itself is performed using the [`lc_correction`](https://github.com/alercebroker/lc_correction) library.

### 4. Analysis of results

Go to `results.ipynb`. This notebook contains different aspects of the post-evaluation analysis, such as latent space visualization, interpolation, attention map visualization, and other plots, and are separated in different sections in the notebook.

### 4.1. Other tasks

### Measure inference execution time

Run the mTAN execution-time script:

```bash
sbatch mtan_gp.sh
```

This executes `mtan_exec_time.py` or `gp_exec_time.py`, depending on what line is commented out in the bash script.


### Evaluation on classes unseen during training

- First run:

```bash
python3 OOD_evaluate_unsupervised_prepare_data.py
```
to get and save data for three new classes: Tidal Disruption Events (TDEs), Long-period variables, and RR Lyrae (the latter two types of light curves are corrected, whereas TDEs are kept uncorrected). We use Fink's API service for this, which gives the full light curves, and predefine the ZTF object IDs to retrieve.

Then change the `SETTING` to `"OOD_test"` and run exactly as step 3 above ("Evaluate model outputs").

## Citation

If you use this code in your work, please cite our paper:

```
TODO
```

## License

[MIT](https://github.com/Yash-10/astro-mtan/blob/main/LICENSE)
