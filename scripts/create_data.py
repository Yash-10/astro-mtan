import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from torch.utils.data import TensorDataset, DataLoader


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

def get_lc(df_alerts, name):
    # Accumulate all alerts for the provided objectId
    pdf = df_alerts[df_alerts['objectId'] == name].sort_values(by='jd')

    if pdf.empty:
        raise ValueError(f'No alerts exist for objectId = {name}, so cannot make a light curve!')

    # Labels of ZTF filters
    # filtdic = {1: 'g', 2: 'r'}

    observation_data, observation_mask = [], []
    # for filt in np.unique(pdf['fid']):
    # Don't loop over pdf['fid'] since in pdf, we might not get all filters. For creating the dataset, we need fixed-sized arrays, so we should select all filters instead of filters seen in this pdf.
    for filt in np.unique(df_alerts['fid']):
        maskFilt = pdf['fid'] == filt
        observation_data.append(
            pdf['magpsf'] * maskFilt
        )
        observation_mask.append(
            maskFilt.astype(int)
        )

    observation_data = np.array(observation_data).T  # after transpose: seqlen x num_channels
    observation_mask = np.array(observation_mask).T  # after transpose: seqlen x num_channels

    times = np.expand_dims(pdf['jd'], 1)  # Add dimension at the 1st index to prepare for concatenation.
    data = np.concatenate((observation_data, observation_mask, times), axis=1)

    return data

def read_alert(folder):
    pdf = pd.read_parquet(folder)
    return pdf

def get_df_alerts(topic):
    df_alerts = read_alert(f'/content/{topic}')
    print(df_alerts.head(2))
    df_alerts_shape = df_alerts.shape
    print(f'No. of alerts = {len(df_alerts)}')
    print(f'No. of transients = {len(df_alerts["objectId"].unique())}')

    # Select the objectIds (transients) that have more than or equal to three alerts in atleast one passband/filter.
    # Note that we mean more than three alerts in the time period in which the alerts are captured and not from the start of the survey.
    # See notes above.
    # For requiring three rather than two alerts, it's because if there are one or two alerts, you can fit anything to them with good accuracy.
    # Only when you have three points or more, can we fit something meaningful.
    df_alerts = df_alerts.groupby('objectId').filter(
        lambda group: (len(group[group['fid'] == 1]) >= 3) or (len(group[group['fid'] == 2]) >= 3)
    )

    print(f'{len(df_alerts)} alerts selected out of {df_alerts_shape[0]}')

    for objectId in df_alerts['objectId'].unique():
        pdf = df_alerts[df_alerts['objectId'] == objectId]
        assert (len(pdf[pdf['fid'] == 1]) >= 3) or (len(pdf[pdf['fid'] == 2]) >= 3)
    
    return df_alerts

def get_data(topic, train_batch_size=32, train_size=0.8, dim=2):
    df_alerts = get_df_alerts(topic)

    seq_len_all = []  # stores the sequence length of all light curves.
    for objId in df_alerts['objectId'].unique():
        # lc_data = get_lc(df_alerts, objId)  # of shape (n, 5), n is the total no. of alerts (including all bands) for that objectId
        pdf = df_alerts[df_alerts['objectId'] == objId]
        seq_len_all.append(len(pdf))

    max_seq_len = max(seq_len_all)
    print(max_seq_len)

    # dim = 2  # no. of channels/passbands in the light curve.

    final_data, final_objIds = [], []

    for objId in df_alerts['objectId'].unique():
        lc_data = get_lc(df_alerts, objId)  # of shape (n, 5), n is the total no. of alerts (including all bands) for that objectId
        obs_time = lc_data[:, -1]
        obs_mask = lc_data[:, dim:2*dim]
        obs_data = lc_data[:, :dim]
        # Make the first time to zero.
        obs_time = obs_time - obs_time[0]
        obs_time = obs_time * 24  # to convert times into hours.
        observed_tp_final = np.expand_dims(
            np.pad(obs_time, (0, max_seq_len-len(lc_data)), 'constant'),  # pad the time array with zero elements before the first time, and `max_seq_len-len(lc_data)` zeros after the last datapoint.
            1
        )
        observed_mask_final = pad_rows_to_match_columns(obs_mask.T, max_seq_len).T
        observed_data_final = pad_rows_to_match_columns(obs_data.T, max_seq_len).T
        data = np.concatenate((observed_data_final, observed_mask_final, observed_tp_final), axis=1)

        final_data.append(data)
        final_objIds.append(objId)

    final_data = np.array(final_data).astype(np.float32)
    print(final_data.shape, len(final_objIds))

    # TODO: should we use stratified split? stratifying based on the most common finkclass across all alerts of a given objId.
    # TODO: Ensure that using random_state=42 and shuffle=True gives the same output.
    train_data, test_data = train_test_split(final_data, train_size=train_size, random_state=42, shuffle=True)

    le = preprocessing.LabelEncoder()
    final_objIds = le.fit_transform(final_objIds)  # use le.inverse_transform to get the string from the encoded value.

    train_data_objId, test_data_objId = train_test_split(final_objIds, train_size=train_size, random_state=42, shuffle=True)
    train_data_objId = torch.as_tensor(train_data_objId)
    test_data_objId = torch.as_tensor(test_data_objId)

    train_data = torch.as_tensor(train_data)
    test_data = torch.as_tensor(test_data)

    print(train_data.shape, test_data.shape, train_data_objId.shape, test_data_objId.shape)

    train_dataset = TensorDataset(train_data, train_data_objId)
    test_dataset = TensorDataset(test_data, test_data_objId)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=True)

    data_obj = {
        "train_dataloader": train_loader,
        "test_dataloader": test_loader,
        "input_dim": dim
    }

    return data_obj
