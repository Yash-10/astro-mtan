import pandas as pd

#df = pd.read_parquet('ftransfer_ztf_2026-02-22_444163/')
df = pd.read_parquet('ftransfer_ztf_2026-02-27_723419/')

from lc_correction.compute import *
from lc_correction.helpers import *

#from compute import apply_correction_df

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
corrected = df.groupby(["objectId", "fid"]).parallel_apply(apply_correction_df)
corrected.reset_index(inplace=True)

print('num_corrected = ', corrected['corrected'].sum(), 'out of', len(corrected))

#corrected = corrected.drop_duplicates(subset=['objectId', 'candid', 'fid', 'jd'], keep='first')

#print('num_corrected after removing duplicates', corrected['corrected'].sum())

corrected_small = corrected.reset_index(drop=True)[['objectId', 'candid', 'fid', 'jd', 'magpsf_corr', 'sigmapsf_corr', 'corrected']]

original_df_alerts = pd.read_parquet('alerts_processed_ftransfer_ztf_2025_05_31_430518.parquet')

keys = ['objectId', 'candid', 'fid', 'jd']
# dtype alignment
#original_df_alerts['objectId'] = original_df_alerts['objectId'].astype('category')
#corrected_small['objectId'] = corrected_small['objectId'].astype('category')

original_df_alerts['fid'] = original_df_alerts['fid'].astype('int8')
corrected_small['fid'] = corrected_small['fid'].astype('int8')

# index join
original_df_alerts = original_df_alerts.set_index(keys)
corrected_small = corrected_small.set_index(keys)

original_df_alerts_corrected = (
    original_df_alerts
    .join(corrected_small, how='left')
    .reset_index()
)

"""
matched = (original_df_alerts_corrected['corrected'] == True).sum()

# Drop rows for whom correction was not possible. NOTE: correction is needed for AGNs/variable stars but not transients in general.
# But here rows for transient alerts are also dropped. As long as we have the original dataframe containing uncorrected mags for all
# alerts (which we do), this shouldn't be a problem as we can extract the uncorrected mags from there. Here we simply remove uncorrected rows
# for simplicity.
original_df_alerts_corrected = original_df_alerts_corrected[original_df_alerts_corrected['corrected'] == True]

#fill_values = {col: -99999 for col in corrected_small.columns}
#original_df_alerts_corrected = original_df_alerts_corrected.fillna(value=fill_values)

#original_df_alerts_corrected = original_df_alerts.merge(
#    corrected_small,
#    on=['objectId', 'candid', 'fid', 'jd'],
#    how='left',
#)
print(len(original_df_alerts_corrected), len(original_df_alerts))
#assert len(original_df_alerts_corrected) == len(original_df_alerts)

print(f"Matched rows: {matched} / {len(original_df_alerts_corrected)}")

print("====")
assert original_df_alerts_corrected['corrected'].all()
assert not(original_df_alerts_corrected['magpsf_corr'].isna().any())
assert not original_df_alerts_corrected.isna().to_numpy().any(), "DataFrame contains NaN values!"
"""

# ---------------------------------------------------------
# HANDLE objectIds WITH AT LEAST ONE uncorrected ROW:
# For consistency within a light curve, overwrite ALL alerts
# of such objectIds with uncorrected magpsf/sigmapsf.
# ---------------------------------------------------------

# Find objectIds where at least one alert has corrected != True or corrected=True
# but magpsf/sigmapsf set to 100 by the lc-correction code.
uncorrected_objids = original_df_alerts_corrected.loc[
    (original_df_alerts_corrected['corrected'] != True) |
    (original_df_alerts_corrected['magpsf_corr'] == 100) |
    (original_df_alerts_corrected['sigmapsf_corr'] == 100),
    'objectId'
].unique()

print(f"objectIds with >= 1 uncorrected alert: {len(uncorrected_objids)}")

# Add flag column: True = used corrected photometry, False = fell back to magpsf
original_df_alerts_corrected['used_magpsf_corr'] = True

# For ALL alerts of those objectIds, overwrite with uncorrected magnitudes
uncorrected_mask = original_df_alerts_corrected['objectId'].isin(uncorrected_objids)
original_df_alerts_corrected.loc[uncorrected_mask, 'magpsf_corr']   = original_df_alerts_corrected.loc[uncorrected_mask, 'magpsf']
original_df_alerts_corrected.loc[uncorrected_mask, 'sigmapsf_corr'] = original_df_alerts_corrected.loc[uncorrected_mask, 'sigmapsf']
original_df_alerts_corrected.loc[uncorrected_mask, 'used_magpsf_corr'] = False

# ---------------------------------------------------------
# INTERNAL TESTS
# ---------------------------------------------------------
# 1. No NaNs in magpsf_corr / sigmapsf_corr
assert not original_df_alerts_corrected['magpsf_corr'].isna().any(),   "NaN found in magpsf_corr"
assert not original_df_alerts_corrected['sigmapsf_corr'].isna().any(), "NaN found in sigmapsf_corr"

# 2. Within each objectId, used_magpsf_corr must be constant (all True or all False)
inconsistent = (
    original_df_alerts_corrected
    .groupby('objectId')['used_magpsf_corr']
    .nunique()
)
assert (inconsistent == 1).all(), "Inconsistent used_magpsf_corr within an objectId"

# 3. For overwritten objectIds, magpsf_corr must equal magpsf exactly
overwritten = original_df_alerts_corrected[~original_df_alerts_corrected['used_magpsf_corr']]
pd.testing.assert_series_equal(
    overwritten['magpsf_corr'].reset_index(drop=True),
    overwritten['magpsf'].reset_index(drop=True),
    check_names=False
)
pd.testing.assert_series_equal(
    overwritten['sigmapsf_corr'].reset_index(drop=True),
    overwritten['sigmapsf'].reset_index(drop=True),
    check_names=False
)

# 4. For non-overwritten objectIds, used_magpsf_corr must all be True
assert original_df_alerts_corrected[original_df_alerts_corrected['used_magpsf_corr']]['corrected'].all(), \
    "Non-overwritten rows contain uncorrected alerts"

# 5. No magpsf/sigmapsf=100 values remain in used rows
corrected_rows = original_df_alerts_corrected[original_df_alerts_corrected['used_magpsf_corr']]
assert (corrected_rows['magpsf_corr'] != 100).all(), "magpsf_corr value 100 found in corrected rows"
assert (corrected_rows['sigmapsf_corr'] != 100).all(), "sigmapsf_corr value 100 found in corrected rows"

print("All internal tests passed.")
print(f"Total alerts: {len(original_df_alerts_corrected)}")
print(f"Using corrected photometry: {original_df_alerts_corrected['used_magpsf_corr'].sum()}")
print(f"Fallen back to magpsf:      {(~original_df_alerts_corrected['used_magpsf_corr']).sum()}")

# ---------------------------------------------------------

original_df_alerts_corrected.to_parquet(
    "alerts_processed_ftransfer_ztf_2025_05_31_430518_CORRECTED.parquet",
    engine="pyarrow",
    index=False
)

import pandas as pd
#df_alerts_corrected = pd.read_parquet('alerts_processed_ftransfer_ztf_2025_05_31_430518_CORRECTED.parquet')
from prepare_data import prepare_data
data_obj = prepare_data(original_df_alerts_corrected, dim=2, train_size=0.8, train_batch_size=8, convert_to_tensor=True, custom_train_min_time=None, custom_train_max_time=None, magpsf_column='magpsf_corr', sigmapsf_column='sigmapsf_corr')#, truncate_agn_months=9)#, remove_less_than_3months_agns=False, truncate_agn_months=None)

import torch
import numpy as np
TOPIC_PATH = '/home/ygondhal/ftransfer_ztf_2025-05-31_430518'
torch.save(data_obj["test_dataloader"], f'test_dataloader_{TOPIC_PATH.split("/")[-1].replace("-", "_")}_CORRECTED.pth')
torch.save(data_obj["test_data_combined"], f'test_data_combined_{TOPIC_PATH.split("/")[-1].replace("-", "_")}_CORRECTED.pth')
np.save(f'test_objIds_{TOPIC_PATH.split("/")[-1].replace("-", "_")}_CORRECTED.npy', data_obj["test_objIds"])
np.save(f'min_max_mags_{TOPIC_PATH.split("/")[-1].replace("-", "_")}_CORRECTED.npy', data_obj['min_max_mags'])
np.save(f'min_max_magdiffs_{TOPIC_PATH.split("/")[-1].replace("-", "_")}_CORRECTED.npy', data_obj['min_max_magdiffs'])

