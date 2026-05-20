import requests
import numpy as np
import pandas as pd
import io
from utils import get_lc, pad_rows_to_match_columns
from mtan_utils import variable_time_collate_fn, get_data_min_max, get_data_min_max_single_record, subsample_timepoints, subsample_timepoints_continuous_window

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

from lc_correction.compute import *
from lc_correction.helpers import *


def get_data_for_objectId(objectId):
	r = requests.post(
		'https://api.ztf.fink-portal.org/api/v1/objects',
		json={
			'objectId': objectId,
			'output-format': 'json'
		}
	)

	# Format output in a DataFrame
	pdf = pd.read_json(io.BytesIO(r.content))

	return pdf

# Extract a part of the light curve #
def _extract_subsection_lc(pdf, objectId, cutoff_days=550):
    alerts_id = pdf[pdf['objectId'] == objectId]
    alerts_id_sorted = alerts_id.sort_values(by='jd', inplace=False)
    alerts_id_sorted.reset_index(drop=True, inplace=True)  # important
    return alerts_id_sorted.iloc[:(alerts_id_sorted['jd'] - alerts_id_sorted['jd'].min()).loc[::-1,].le(cutoff_days).idxmax()]

object_ids = {
    'rrlyrae': ['ZTF17aaaebgd', 'ZTF18abadkhl', 'ZTF18adalmbo', 'ZTF18abcosdx', 'ZTF18abklama'],
    'lpv':     ['ZTF18abryodl', 'ZTF18aabjjhq', 'ZTF18aaaaogd'],
    'tde':     ['ZTF20aahmtso', 'ZTF25aclckzl', 'ZTF20accxwrk', 'ZTF22aadesap'],  # ZTF25aclckzl: https://www.wis-tns.org/astronotes/astronote/2026-7
}

# Fetch and concatenate within each group
pdfs = {
    label: pd.concat([get_data_for_objectId(oid) for oid in ids], ignore_index=True, sort=False)
    for label, ids in object_ids.items()
}
#print(pdfs['rrlyrae']['i:objectId'].unique())

RENAME_COL_MAP = {
    'i:candid':    'candid',
    'i:isdiffpos': 'isdiffpos',
    'i:distnr':    'distnr',
    'i:magnr':     'magnr',
    'i:magpsf':    'magpsf',
    'i:sigmagnr':  'sigmagnr',
    'i:sigmapsf':  'sigmapsf',
    'i:objectId':  'objectId',
    'i:jdendref':  'jdendref',
    'i:fid':       'fid',
    'i:jd':        'jd',
}

def _rename_columns(pdf):
    return pdf.rename(columns=RENAME_COL_MAP)

# Correct only variable star classes (rrlyrae, lpv), NOT tde because it's a transient
CORRECT_LABELS = {'rrlyrae', 'lpv'}

corrected_pdfs = {}
for label, pdf in pdfs.items():
    pdf = _rename_columns(pdf)
    if label in CORRECT_LABELS:
        corrected = pdf.groupby(["objectId", "fid"]).apply(apply_correction_df)
        corrected.reset_index(inplace=True)
        corrected_pdfs[label] = corrected
    else:
        corrected_pdfs[label] = pdf  # TDEs: keep raw difference photometry

print(len(corrected_pdfs))
# Extract subsection and merge
pdf_merged = pd.concat(
    [_extract_subsection_lc(corrected_pdfs[label], oid)
     for label, ids in object_ids.items()
     for oid in ids],
    ignore_index=True, sort=False
)
assert set(pdf_merged['objectId'].unique()) == set(oid for ids in object_ids.values() for oid in ids), \
    "Some objectIds missing from pdf_merged"
print(pdf_merged['objectId'].unique())

"""
# Extract subsection using the first object ID of each group, then merge
pdf_merged = pd.concat(
    [_extract_subsection_lc(pdfs[label], ids[0]) for label, ids in object_ids.items()],
    ignore_index=True, sort=False
)
"""
"""
pdf_rrlyrae1 = get_data_for_objectId('ZTF17aaaebgd')
pdf_rrlyrae2 = get_data_for_objectId('ZTF18abadkhl')
pdf_rrlyrae3 = get_data_for_objectId('ZTF18adalmbo')
pdf_rrlyrae4 = get_data_for_objectId('ZTF18abcosdx')
pdf_rrlyrae5 = get_data_for_objectId('ZTF18abklama')
pdf_lpv1 = get_data_for_objectId('ZTF18abryodl')
pdf_lpv2 = get_data_for_objectId('ZTF18aabjjhq')
pdf_lpv3 = get_data_for_objectId('ZTF18aaaaogd')
pdf_tde1 = get_data_for_objectId('ZTF20aahmtso')
pdf_tde2 = get_data_for_objectId('ZTF25aclckzl')  # see https://www.wis-tns.org/astronotes/astronote/2026-7
pdf_tde3 = get_data_for_objectId('ZTF20accxwrk')
pdf_tde4 = get_data_for_objectId('ZTF22aadesap')
"""

"""
pdf_rrlyrae = _extract_subsection_lc(pdf_rrlyrae, 'ZTF17aaaebgd')
pdf_lpv = _extract_subsection_lc(pdf_lpv, 'ZTF18abryodl')
pdf_tde = _extract_subsection_lc(pdf_tde, 'ZTF20aahmtso')

#####################################

# Source - https://stackoverflow.com/a
# Posted by Joran Beasley, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-25, License - CC BY-SA 4.0
pdf_merged = pd.concat([pdf_rrlyrae, pdf_lpv, pdf_tde], ignore_index=True, sort=False)
"""

rrlyrae_lpv_ids = set(object_ids['rrlyrae'] + object_ids['lpv'])

device = 'cpu'  # We don't require GPU fr preparing the data but only for training.
time_in_hrs = True
convert_to_tensor = True
min_time, max_time = None, None  # This doesn't matter because it's not used as normalize_times=False in get_lc. We do normalization inside variable_time_collate_fn instead.

total_data, total_objId, total_common_finkclasses = [], [], []
min_max_mags = []
for objId in pdf_merged['objectId'].unique():
    if objId in rrlyrae_lpv_ids:
        magpsf_col = 'magpsf_corr'
        sigmapsf_col = 'sigmapsf_corr'
    else:
        magpsf_col = 'magpsf'
        sigmapsf_col = 'sigmapsf'

    # NOTE: It's important to note that here we use normalize_times=False since we want min_time and max_time calculated on the specific dataset (train, val, OR test), just as done in mTAN phyionet data preprocessing. This normalization is done in variable_time_collate_fn.
    lc_data = get_lc(pdf_merged, objId, make_first_time_zero=True, convert_to_tensor=convert_to_tensor, normalize_times=False, local_time_normalization=False, max_time=max_time, min_time=min_time, time_in_hrs=time_in_hrs, fid_column='fid', magpsf_column=magpsf_col, jd_column='jd', objectId_column='objectId', sigmapsf_column=sigmapsf_col, finkclass_column=None)  # returns a tuple (object_Id, tt, vals, mask, labels). objectId will be a string, no. of entries/rows in tt, vals, and mask will be `n` = the total no. of alerts (including all bands) for that objectId
    total_data.append(lc_data)
    assert lc_data[0] == objId

    _lc_data_obs = lc_data[2]
    min_max_mags.append((objId, _lc_data_obs[_lc_data_obs != 0.0].min().item(), _lc_data_obs.max().item()))

    total_objId.append(objId)
    total_common_finkclasses.append(lc_data[-1])

data_min, data_max = None, None  # Since we don't use min/max calculated across the entire train/val/test dataset.
train_val_test_min_max_times = np.load('train_val_test_min_max_times.npy')
train_min_time, train_max_time = train_val_test_min_max_times[0], train_val_test_min_max_times[1]
print(train_min_time, train_max_time)

classify = False
activity = False
data_combined, data_Ids = variable_time_collate_fn(total_data, device, classify=classify, activity=activity, data_min=data_min, data_max=data_max, train_min_time=train_min_time, train_max_time=train_max_time)

from prepare_data import MyDataSet

test_dataset = MyDataSet(data_combined, data_Ids, transform=None)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=False)
torch.save(data_combined, f'OOD_test_data_combined.pth')
torch.save(test_loader, f'OOD_test_dataloader.pth')
np.save(f'OOD_test_objIds.npy', data_Ids)
np.save('OOD_min_max_mags.npy', np.array(min_max_mags))
pdf_merged.to_json('OOD_df_alerts.json')
