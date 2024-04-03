import io
import numpy as np
import requests
import pandas as pd
from utils import get_lc, pad_rows_to_match_columns
from mtan_utils import variable_time_collate_fn, get_data_min_max

import torch
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader

def add_tns_tde():  # TODO: Generalize this function to allow any object, not just TDEs. This will help in curating a confident, labelled dataset.
    tns_tde_objIds = ['ZTF24aaahxwr', 'ZTF24aaecooj', 'ZTF20aahmtso', 'ZTF22aafujzv', 'ZTF22aadesap', 'ZTF18aabdajx', 'ZTF21aanxhjv', 'ZTF22abegjtx']
    flags = [1, 1, 0, 0, 0, 0, 0, 0]  # 1 means use entire light curve since it contains few points already. 0 means need to manually select a subset.
    subset_indices = [(None, None), (None, None), (1,35), (0,25), (0,20), (0,9), (0,15), (0,21)]  # indices are 0:len(pdf), 0:len(pdf), 1:35, 0:25, etc.

    total_data, total_objId, total_common_finkclasses = [], [], []
    for counter, tobjId in enumerate(tns_tde_objIds):
        X, Y = subset_indices[counter]

        # get data for many objects
        r = requests.post(
          'https://fink-portal.org/api/v1/objects',
          json={
            'objectId': ','.join(tns_tde_objIds),
            'output-format': 'json'
          }
        )

        fid_column, jd_column, magpsf_column, sigmapsf_column, objectId_column = 'i:fid', 'i:jd', 'i:magpsf', 'i:sigmapsf', 'i:objectId'

        # Format output in a DataFrame
        pdf = pd.read_json(io.BytesIO(r.content))
        pdf = pdf[pdf[objectId_column] == tobjId].sort_values(by=jd_column)

        if X is not None and Y is not None:
            jds, magpsfs, sigmapsfs, filters = [],[],[],[]
            for filt in np.unique(pdf[fid_column]):
                # select data from one filter at a time
                maskFilt = pdf[fid_column] == filt
                jds.append(pdf[maskFilt][jd_column][X:Y])
                magpsfs.append(pdf[maskFilt][magpsf_column][X:Y])
                sigmapsfs.append(pdf[maskFilt][sigmapsf_column][X:Y])
                for _ in range(Y-X):
                    filters.append(filt)

            jds = np.expand_dims(np.array(jds).flatten(), 1)
            magpsfs = np.expand_dims(np.array(magpsfs).flatten(), 1)
            sigmapsfs = np.expand_dims(np.array(sigmapsfs).flatten(), 1)
            df = pd.DataFrame(np.hstack((jds, magpsfs, sigmapsfs)))
            df.columns = ['i:jd', 'i:magpsf', 'i:sigmapsf']
            df['i:fid'] = filters
            df['i:objectId'] = tobjId
            df['i:finkclass'] = 'TDE'
            lc_data = get_lc(df, tobjId, fid_column=fid_column, jd_column=jd_column, magpsf_column=magpsf_column, sigmapsf_column=sigmapsf_column, objectId_column=objectId_column, finkclass_column='i:finkclass', convert_to_tensor=True)
        else:
            lc_data = get_lc(pdf, tobjId, fid_column=fid_column, jd_column=jd_column, magpsf_column=magpsf_column, sigmapsf_column=sigmapsf_column, objectId_column=objectId_column, finkclass_column=None, convert_to_tensor=True)

        total_data.append(lc_data)
        assert lc_data[0] == tobjId
        total_objId.append(tobjId)
        total_common_finkclasses.append(lc_data[-1] if lc_data[-1] != [None] else 'TDE')

    return total_data, total_objId, total_common_finkclasses


def prepare_data(df_alerts, dim=2, train_size=0.7, train_batch_size=32, classify=False, activity=False, convert_to_tensor=False):
    """
    """
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'  # We don't require GPU fr preparing the data but only for training.

    total_data, total_objId, total_common_finkclasses = [], [], []
    for objId in df_alerts['objectId'].unique():
        lc_data = get_lc(df_alerts, objId, convert_to_tensor=True)  # returns a tuple (object_Id, tt, vals, mask, labels). objectId will be a string, no. of entries/rows in tt, vals, and mask will be `n` = the total no. of alerts (including all bands) for that objectId
        total_data.append(lc_data)
        assert lc_data[0] == objId
        total_objId.append(objId)
        total_common_finkclasses.append(lc_data[-1])

    ################################################
    # Add cases manually. Currently we only add TDEs
    tns_tde_total_data, tns_tde_total_objId, tns_tde_total_common_finkclasses = add_tns_tde()
    total_data.extend(tns_tde_total_data)
    total_objId.extend(tns_tde_total_objId)
    total_common_finkclasses.extend(tns_tde_total_common_finkclasses)
    ################################################

    data_min, data_max = get_data_min_max(total_data)
    print(f'data_min, data_max: {data_min, data_max}')
    data_min, data_max = data_min.to(device), data_max.to(device)

    # TODO: should we use stratified split? stratifying based on the most common finkclass across all alerts of a given objId --> can do for classification, not required for unsupervised learning.
    # TODO: Ensure that using random_state=42 and shuffle=True gives the same output since I am using train_test_independently for splitting the data and the objIds.
    # TODO: ensure multiple runs of label encoding on the same number gives the same encoded value.
    # We are encoding the objectIds just for efficiency because string types may not be efficient with PyTorch.
    le = preprocessing.LabelEncoder()
    total_objId_encoded = le.fit_transform(total_objId)  # use le.inverse_transform to get the string from the encoded value.

    train_data, test_data, train_data_objId, test_data_objId = train_test_split(total_data, total_objId_encoded, train_size=train_size, random_state=42, shuffle=True)
    train_data, val_data, train_data_objId, val_data_objId = train_test_split(train_data, train_data_objId, train_size=train_size, random_state=42, shuffle=True)

    # Note: As per the mTAN code, we are using the same data_min and data_max across train, val, and test sets: these min/max vals are calculated using all three combined above.
    train_data_combined = variable_time_collate_fn(train_data, device, classify=classify, activity=activity,
                                                      data_min=data_min, data_max=data_max)
    val_data_combined = variable_time_collate_fn(val_data, device, classify=classify, activity=activity,
                                                      data_min=data_min, data_max=data_max)
    test_data_combined = variable_time_collate_fn(test_data, device, classify=classify, activity=activity,
                                                      data_min=data_min, data_max=data_max)

    print(f'train_data_combined.shape, val_data_combined.shape, test_data_combined.shape: {train_data_combined.shape, val_data_combined.shape, test_data_combined.shape}')

    ##### A quick check #####
    seq_len_all = []  # stores the sequence length of all light curves.
    for objId in df_alerts['objectId'].unique():
        pdf = df_alerts[df_alerts['objectId'] == objId]
        seq_len_all.append(len(pdf))

    max_seq_len = max(seq_len_all)  # max_seq_len does not denote the maximum time duration, i.e., if max_seq_len = 60, it doesn't mean 60 hours/minutes.
    print(f'Max. sequence length (or the max no. of datapoints of lightcurves) in the dataset (max_seq_len): {max_seq_len}')
    print('TODO: max_seq_len AND THE MAX(1ST DIMENSION OF TRAIN_DATA_COMBINED, VAL_DATA_COMBINED, TEST_DATA_COMBINED) MUST BE SAME -- CHECK THAT')
    #########################

    train_loader = DataLoader(train_data_combined, batch_size=train_batch_size, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_data_combined, batch_size=1, num_workers=2, shuffle=False)
    test_loader = DataLoader(test_data_combined, batch_size=1, num_workers=2, shuffle=False)

    # Since total_common_finkclass may be heterogenous, we pad them just for convenience before saving as an numpy array.
    # Credit for below code: https://stackoverflow.com/a/43146373
    pad = len(max(total_common_finkclasses, key=len))
    total_common_finkclasses = np.array([i + [0]*(pad-len(i)) for i in total_common_finkclasses])

    train_data_objId_raw = le.inverse_transform(train_data_objId)
    val_data_objId_raw = le.inverse_transform(val_data_objId)
    test_data_objId_raw = le.inverse_transform(test_data_objId)
    
    data_obj = {
        #"final_data": np.array(total_data),  # This may give error since total_data is a list containing variable length entries.
        "total_objIds": np.array(total_objId),
        "train_objIds": train_data_objId_raw,
        "val_objIds": val_data_objId_raw,
        "test_objIds": test_data_objId_raw,
        "total_objIds_encoded": np.array(total_objId_encoded),
        "total_common_finkclasses": total_common_finkclasses,
        "train_dataloader": train_loader,
        "test_dataloader": test_loader,
        "val_dataloader": val_loader,
        "input_dim": dim
    }

    return data_obj




def prepare_data_old(df_alerts, dim=2, train_size=0.8, train_batch_size=32):
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
    # Out of these, I see these to show a good peak in both bands and can be selected and others can be discarded: tns_tde_objIds = ['ZTF24aaahxwr', 'ZTF24aaecooj', 'ZTF20aahmtso', 'ZTF22aafujzv', 'ZTF22aadesap', 'ZTF18aabdajx', 'ZTF21aanxhjv', 'ZTF22abegjtx']
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
        lc_data = get_lc(df_alerts, objId)
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
 
