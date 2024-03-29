import shutil
from distutils.dir_util import copy_tree

import os
import glob
import numpy as np
import pandas as pd

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

def get_lc(df_alerts, name, fid_column='fid', magpsf_column='magpsf', jd_column='jd', objectId_column='objectId', sigmapsf_column='sigmapsf', finkclass_column='finkclass'):
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
        df_alerts, name, fid_column='fid', magpsf_column='magpsf', jd_column='jd',
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
    plt.title(f'{pdf[objectId_column].unique()[0]}')
    plt.xlabel('Modified Julian Date')
    plt.ylabel('Magnitude')
    plt.show()
    # msg = """
    # - Circles (●) with error bars show valid alerts that pass the Fink quality cuts.
    # - Upper triangles with errors (▲), represent alert measurements that do not satisfy Fink quality cuts, but are nevetheless contained in the history of valid alerts and used by classifiers.
    # - Lower triangles (▽), represent 5-sigma mag limit in difference image based on PSF-fit photometry contained in the history of valid alerts.
    # """
    # print(msg)


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
