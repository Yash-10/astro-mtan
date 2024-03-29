import io
import numpy as np
import requests
import pandas as pd
from utils import get_lc, pad_rows_to_match_columns

import torch
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader

def prepare_data(df_alerts, dim=2, train_size=0.8, train_batch_size=32):
    """

    dim: no. of channels/passbands in the light curve.
    """
    # Mine specific cases and add them to the data
    # simbad_CataclyV_objId_list = ['ZTF18abucxou']
    # tns_cv_objId_list = []
    tns_tde_objId_list = [
        'ZTF24aaecooj', 'ZTF18aahqkbt', 'ZTF19accskvu', 'ZTF22aafujzv', 'ZTF18aabdajx', 'ZTF24aaahxwr',
        'ZTF23abvzeqp', 'ZTF18aabtxvd', 'ZTF20aahmtso', 'ZTF22aadesap', 'ZTF20abwtifz', 'ZTF21aanxhjv',
        'ZTF18achzddr', 'ZTF23abohtqf', 'ZTF19abzrhgq', 'ZTF23abgnxfv', 'ZTF22abegjtx', 'ZTF17aaazdba'
    ]  # These 18 objects are obtained from the fink portal (https://fink-portal.org/; class="(TNS) TDE"). TDEs are not present in SIMBAD.
    tns_novae_objId_list = []
    simbad_novae_objId_list = []

    # get data for many objects
    r = requests.post(
      'https://fink-portal.org/api/v1/objects',
      json={
        'objectId': ','.join(tns_tde_objId_list),
        'output-format': 'json'
      }
    )

    # Format output in a DataFrame
    pdf = pd.read_json(io.BytesIO(r.content))
    # pdf[['i:jd', 'i:magpsf', 'i:sigmapsf']]

    # These below conditions are the same as used for polling the alerts, except the ndethist condition (see the commented line)
    # The ndethist condition may not be appropriate here. Since these are full light curves, ndethist will be > 80 in non-trivial no. of cases. Instead, we manually look at the light curves and select subregions ourselves, majorly around the event of interest.
    conditions = (
        (pdf['i:nbad'] == 0)  # 1142
        & (pdf['i:fwhm'] <= 5)  # 1136
        & (pdf['i:elong'] <= 1.2)  # 1030
        & (pdf['i:magdiff'].abs() <= 0.1)  # 586
        & ((pdf['i:drb'] > 0.8) & (pdf['i:rb'] > 0.8))  # 551
        #& ((pdf['i:ndethist'] > 2) & (pdf['i:ndethist'] < 80))  # 148
        & (pdf['i:isdiffpos'] == "t")  # 146
        & ((pdf['i:ssdistnr'] >= 10) | (pdf['i:ssdistnr'] <= 0))  # 146
        & (~((pdf['i:sgscore1'] > 0.5) & (pdf['i:distpsnr1'] < 1.5) & (pdf['i:distpsnr1'] >= 0)))  # 146
        & (pdf['i:jd'] - pdf['i:jdstarthist'] > 30/60/24)  # 144
    )
    print(f'No. of rows of additional alerts (before): {len(pdf)}')
    pdf_filtered = pdf[conditions]
    print(f'No. of rows of additional alerts (after): {len(pdf_filtered)}')

    pdf_filtered_filtered = pdf_filtered.groupby('i:objectId').filter(
        lambda group: (len(group[group['i:fid'] == 1]) >= 3) or (len(group[group['i:fid'] == 2]) >= 3)
    )

    seq_len_all = []  # stores the sequence length of all light curves.
    for objId in df_alerts['objectId'].unique():
        # lc_data = get_lc(df_alerts, objId)  # of shape (n, 5), n is the total no. of alerts (including all bands) for that objectId. 5 because 2 dims for observation in the two bands, 2 dims for observation_mask in the two bands, and the last dimension for time values.
        pdf = df_alerts[df_alerts['objectId'] == objId]
        seq_len_all.append(len(pdf))

    max_seq_len = max(seq_len_all)
    # max_seq_len does not denote the maximum time duration, i.e., if max_seq_len = 60, it doesn't mean 60 hours/minutes, for example.
    print(f'Max. sequence length (or the max no. of datapoints of lightcurves) in the dataset: {max_seq_len}')

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

    # Now add separately mined transients (TDEs).
    # TODO: Remove this and instead manually seelct regions of each light curve and add them separately.
    for objId in pdf_filtered_filtered['i:objectId'].unique():
        lc_data = get_lc(
            pdf_filtered_filtered, objId,
            fid_column='i:fid', magpsf_column='i:magpsf', jd_column='i:jd',
            sigmapsf_column='i:sigmapsf', finkclass_column=None, objectId_column='i:objectId'
        )  # of shape (n, 5), n is the total no. of alerts (including all bands) for that objectId
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

    final_data = np.array(final_data)
    print(final_data.shape, len(final_objIds))
 
    # TODO: should we use stratified split? stratifying based on the most common finkclass across all alerts of a given objId --> can do for classification, not required for unsupervised learning.
    # TODO: Ensure that using random_state=42 and shuffle=True gives the same output since I am using train_test_independently for splitting the data and the objIds.
    # TODO: ensure multiple runs of label encoding on the same number gives the same encoded value.
    le = preprocessing.LabelEncoder()
    final_objIds_encoded = le.fit_transform(final_objIds)  # use le.inverse_transform to get the string from the encoded value.

    train_data, test_data, train_data_objId, test_data_objId = train_test_split(final_data, final_objIds_encoded, train_size=train_size, random_state=42, shuffle=True)
    train_data, val_data, train_data_objId, val_data_objId = train_test_split(train_data, train_data_objId, train_size=train_size, random_state=42, shuffle=True)

    train_data_objId = torch.as_tensor(train_data_objId)
    val_data_objId = torch.as_tensor(val_data_objId)
    test_data_objId = torch.as_tensor(test_data_objId)
    train_data = torch.as_tensor(train_data)
    val_data = torch.as_tensor(val_data)
    test_data = torch.as_tensor(test_data)

    print(f'train_data.shape, val_data.shape, test_data.shape, train_data_objId.shape, val_data_objId.shape, test_data_objId.shape: {train_data.shape, val_data.shape, test_data.shape, train_data_objId.shape, val_data_objId.shape, test_data_objId.shape}')

    train_dataset = TensorDataset(train_data, train_data_objId)
    val_dataset = TensorDataset(val_data, val_data_objId)
    test_dataset = TensorDataset(test_data, test_data_objId)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=False)

    data_obj = {
        "final_data": final_data,
        "final_objIds": final_objIds,
        "final_objIds_encoded": final_objIds_encoded,
        "train_dataloader": train_loader,
        "test_dataloader": test_loader,
        "val_dataloader": val_loader,
        "input_dim": dim
    }
    return data_obj
 
