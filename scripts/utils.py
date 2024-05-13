import shutil
from distutils.dir_util import copy_tree

import os
import glob
import numpy as np
import torch
import pandas as pd

import matplotlib.pyplot as plt

def get_dirs(topic_path):
    """Prints the no. of alerts for each fink class."""
    DIRS = f'{topic_path}/*'
    for DIR in glob.glob(DIRS):
        num_files = 0
        for f in glob.glob(os.path.join(DIR, '*.parquet')):
            df = pd.read_parquet(f)
            num_files += len(df)
        print(f'{DIR.split("/")[-1]}: {num_files} alerts')

def read_alert(folder):
    pdf = pd.read_parquet(folder)
    return pdf

def get_lc(
        df_alerts, name, fid_column='fid', magpsf_column='magpsf', jd_column='jd',
        objectId_column='objectId', sigmapsf_column='sigmapsf', finkclass_column='finkclass',
        #extract_subset=False, start_index=None, end_index=None
        make_first_time_zero=True, convert_to_tensor=False, normalize_times=False,
        local_time_normalization=False, max_time=None, min_time=None, time_in_hrs=True
    ):
    """Get the light curve given an alerts dataframe (df_alerts) and the objectId (name).
    
    Returns a tuple (object_Id, tt, vals, mask, labels)
    """
    # Accumulate all alerts for the provided objectId
    pdf = df_alerts[df_alerts[objectId_column] == name].sort_values(by=jd_column)

    if pdf.empty:
        raise ValueError(f'No alerts exist for objectId = {name}, so cannot make a light curve!')

    # Labels of ZTF filters
    filtdic = {1: 'g', 2: 'r'}

    observation_data, observation_mask = [], []
    # for filt in np.unique(pdf['fid']):
    # Don't loop over pdf['fid'] since in pdf, we might not get all filters. For creating the dataset, we need fixed-sized arrays, so we should select all filters instead of filters seen in this pdf.
    for filt in np.unique(df_alerts[fid_column]):
        #if extract_subset:
        #    maskFilt = pdf[fid_column] == filt
        #    observation_data.append(
        #        (pdf[magpsf_column] * maskFilt).iloc[start_index:end_index]
        #    )
        #    observation_mask.append(
        #        maskFilt.astype(int)
        #    )
        #else:
        maskFilt = pdf[fid_column] == filt
        observation_data.append(
            pdf[magpsf_column] * maskFilt  # because when observation for this filter is not present, we want to replace the observed value with zero, as done in the mTAN code.
        )
        observation_mask.append(
            maskFilt.astype(int)
        )

    observation_data = np.array(observation_data).T  # after transpose: seqlen x num_channels
    observation_mask = np.array(observation_mask).T  # after transpose: seqlen x num_channels

    times = np.expand_dims(pdf[jd_column], 1)  # Add dimension at the 1st index to prepare for concatenation.
    #data = np.concatenate((observation_data, observation_mask, times), axis=1)

    if make_first_time_zero:
        # As per the Physionet dataset (at least) from the mTAN paper, the times always start at zero. Their time units are also in hours. So this replicates that.
        # Make the first time to zero. The below two lines are only for machine learning purposes since the time must always start at zero for all light curves.
        times = times - times[0]
        if time_in_hrs:
            times = times * 24  # to convert times into hours.

    if normalize_times:
        # NOTE: If you use get_lc for different length light curves, note the possible caveat that a normalized time value of 1 means the same for two very different length light curves. It's possible that despite this, the relative difference in the times already encodes the information about different duration/length light curves. Not sure definitively.
        times = normalize_time_values(times, local_time_normalization=local_time_normalization, max_time=max_time, min_time=min_time)

    if finkclass_column is not None:  # finkclass_column will be None when getting the light curve from the API service instead of polling the alerts.
        common_finkclasses = df_alerts[df_alerts[objectId_column] == name][finkclass_column].mode().tolist()
    else:
        common_finkclasses = [None]
    
    if convert_to_tensor:
        times = torch.from_numpy(times)
        observation_data = torch.from_numpy(observation_data)
        observation_mask = torch.from_numpy(observation_mask)

    data = (name, times, observation_data, observation_mask, common_finkclasses)

    return data

def normalize_time_values(times, local_time_normalization=False, max_time=None, min_time=None):
    """`times` must start with zero and be in units of hours. This function assumes that.
    times are multipled by 48 after normalization which means the normalized time valus lie in [0, 48] hours.
    """
    if local_time_normalization:
        normalized_times = (times - np.min(times)) / (np.max(times) - np.min(times))
    else:  # Means global normalization must be used. In this case, max_time argument will be used.
        if max_time is None or min_time is None:
            raise ValueError("max_time and min_time both must be provided if using global time normalization.")
        assert max_time > np.min(times)  # Otherwise time will become negative.
        normalized_times = (times - min_time) / (max_time - min_time)
    #normalized_times *= 48  # Doing this is not needed since anyways variable_time_collate_fn will normalize to the [0, 1] range.
    return normalized_times

def get_lc_old(df_alerts, name, fid_column='fid', magpsf_column='magpsf', jd_column='jd', objectId_column='objectId', sigmapsf_column='sigmapsf', finkclass_column='finkclass'):
    """Get the light curve given an alerts dataframe (df_alerts) and the objectId (name)."""
    # Accumulate all alerts for the provided objectId
    pdf = df_alerts[df_alerts[objectId_column] == name].sort_values(by=jd_column)

    if pdf.empty:
        raise ValueError(f'No alerts exist for objectId = {name}, so cannot make a light curve!')

    # Labels of ZTF filters
    filtdic = {1: 'g', 2: 'r'}

    observation_data, observation_mask = [], []
    # for filt in np.unique(pdf['fid']):
    # Don't loop over pdf['fid'] since in pdf, we might not get all filters. For creating the dataset, we need fixed-sized arrays, so we should select all filters instead of filters seen in this pdf.
    for filt in np.unique(df_alerts[fid_column]):
        maskFilt = pdf[fid_column] == filt
        observation_data.append(
            pdf[magpsf_column] * maskFilt
        )
        observation_mask.append(
            maskFilt.astype(int)
        )

    observation_data = np.array(observation_data).T  # after transpose: seqlen x num_channels
    observation_mask = np.array(observation_mask).T  # after transpose: seqlen x num_channels

    times = np.expand_dims(pdf[jd_column], 1)  # Add dimension at the 1st index to prepare for concatenation.
    data = np.concatenate((observation_data, observation_mask, times), axis=1)

    return data

def plot_lc(
        df_alerts, name, title_suffix='', fid_column='fid', magpsf_column='magpsf', jd_column='jd',
        sigmapsf_column='sigmapsf', finkclass_column='finkclass', objectId_column='objectId'
):
    """Plots photometry for the given name (objectId) from the alerts dataframe.

    Parameters
    ----------
    name: str
        objectID
    """
    # Accumulate all alerts for the provided objectId
    pdf = df_alerts[df_alerts[objectId_column] == name].sort_values(by=jd_column)

    if pdf.empty:
        raise ValueError(f'No alerts exist for objectId = {name}!')

    fig = plt.figure(figsize=(15, 6))

    # Colors to plot
    colordic = {1: 'C0', 2: 'C1'}

    # Labels of ZTF filters
    filtdic = {1: 'g', 2: 'r'}

    for filt in np.unique(pdf[fid_column]):
        # select data from one filter at a time
        maskFilt = pdf[fid_column] == filt

        plt.errorbar(
            pdf[maskFilt][jd_column],
            pdf[maskFilt][magpsf_column],
            pdf[maskFilt][sigmapsf_column],
            ls = '', marker='o', color=colordic[filt], label=filtdic[filt]
        )

        plt.errorbar(
            pdf[maskFilt][jd_column],
            pdf[maskFilt][magpsf_column],
            pdf[maskFilt][sigmapsf_column],
            ls='', marker='^', color=colordic[filt]
        )

        if finkclass_column is not None:
            _offset_x = (pdf[maskFilt][jd_column].max() - pdf[maskFilt][jd_column].min()) / 200
            _offset_y = (pdf[maskFilt][magpsf_column].max() - pdf[maskFilt][magpsf_column].min()) / 100
            for x, y, string in zip(pdf[maskFilt][jd_column], pdf[maskFilt][magpsf_column], pdf[maskFilt][finkclass_column]):
                plt.text(
                    x+_offset_x, y+_offset_y, string,
                    color=colordic[filt]
                )

    plt.gca().invert_yaxis()
    plt.legend()
    plt.title(f'{pdf[objectId_column].unique()[0]}'+': '+title_suffix)
    plt.xlabel('Modified Julian Date')
    plt.ylabel('Magnitude')
    plt.show()
    # msg = """
    # - Circles (●) with error bars show valid alerts that pass the Fink quality cuts.
    # - Upper triangles with errors (▲), represent alert measurements that do not satisfy Fink quality cuts, but are nevetheless contained in the history of valid alerts and used by classifiers.
    # - Lower triangles (▽), represent 5-sigma mag limit in difference image based on PSF-fit photometry contained in the history of valid alerts.
    # """
    # print(msg)


def plot_lc_normalized_data(observed_data, observed_mask, observed_tp, title=None):
    """Just like plot_lc() but plots normalized data (i.e., after all preprocessing on magnitudes and times), just before inputting them to the model.

    combined_data must be a tuple of (observed_data, observed_mask, time) coming from a dataloader (since it must be normalized).
    observed_data shape: [1, lc_length, dim]
    observed_mask shape: [1, lc_length, dim]
    observed_tp shape: [1, lc_length, 1]

    Below is an example code to do that:
    ```
    dim = 2
    test_loader = torch.load('test_dataloader.pth')
    batch = next(iter(test_loader))
    data = batch[0]

    # `index` below controls which image in this batch should be shown.
    index = 0
    assert index <= batch_len - 1

    batch_len = data.shape[0]
    observed_data = data[index, :, :dim].cpu().detach().numpy()
    observed_mask = data[index, :, dim:2 * dim].cpu().detach().numpy()
    observed_tp = data[index, :, -1].cpu().detach().numpy()
    ```
    """
    dim = 2
    # Replace observed values that are zero with nan since they are unobserved and must not show as zero in the plot.
    observed_data[observed_data == 0] = np.nan

    # Colors to plot
    colordic = {1: 'C0', 2: 'C1'}

    # Labels of ZTF filters
    filtdic = {1: 'g', 2: 'r'}

    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    for i in range(dim):
        ax.plot(
            observed_tp.squeeze(), observed_data[:, i],
            ls = '', marker='o', color=colordic[i+1], label=filtdic[i+1]
        )

    plt.gca().invert_yaxis()
    ax.legend()
    ax.set_title(f'{title}')
    ax.set_xlabel('Normalized time')
    ax.set_ylabel('Normalized magnitude')
    plt.show()

def pad_rows_to_match_columns(array, target_columns):
    """
    Pad each row of a 2D NumPy array with zeros to match a specified number of columns.

    Parameters:
    - array: 2D NumPy array
    - target_columns: Number of columns to match

    Returns:
    - Padded 2D NumPy array
    """
    # Get the number of columns in the original array
    original_columns = array.shape[1]

    # Calculate the number of columns to pad for each row
    pad_width = target_columns - original_columns

    # Pad each row with zeros
    padded_array = np.pad(array, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

    return padded_array
