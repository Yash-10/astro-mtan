import os
import glob
import shutil
import numpy as np
import pandas as pd
import torch
from utils import read_alert
from prepare_data import prepare_data, get_tns_tde_alerts
from constants import agn_list, stars_list, sn_list

TOPIC_PATH = '/home/ygondhal/ftransfer_ztf_2024-04-26_572037_copy'


def preprocessing_alert_folders(topic_path):
    """Preprocessing for arranging subfolders (denoting the fink class) after the
    raw alert stream has been polled. This function mainly groups different subgroups together.

    Caution: If using this function, remember that reading the alerts into a dataframe will still
    show the ungrouped finkclasses and not the revised ones as implemented in this function. Probably
    the best way is to not modify the alerts dataframe to show the grouped finkclass, but instead take
    care of it downstream."""

    df_alerts = read_alert(topic_path)
    print(f'Alerts dataframe columns: {df_alerts.columns}')
    print(f'No. of alerts (before preprocessing) = {len(df_alerts)}')
    print(f'No. of transients (before preprocessing) = {len(df_alerts["objectId"].unique())}')


    DIRS = f'{topic_path}/*'

    AGN_DIR = os.path.join(f'{topic_path}', 'custom_agn')
    STARS_DIR = os.path.join(f'{topic_path}', 'custom_stars')
    SN_DIR = os.path.join(f'{topic_path}', 'custom_sn')
    SIMBAD_GALAXIES_DIR = os.path.join(f'{topic_path}', 'custom_simbad_galaxies')
    os.makedirs(AGN_DIR, exist_ok=True)
    os.makedirs(STARS_DIR, exist_ok=True)
    os.makedirs(SIMBAD_GALAXIES_DIR, exist_ok=True)

    print('Starting arranding subfolders...')
    for raw_dir in glob.glob(DIRS):
        dir = raw_dir.split('/')[-1].split('finkclass=')[-1]
        if dir in agn_list:
            shutil.move(raw_dir, AGN_DIR)
            df_alerts.finkclass.replace(dir, 'custom_agn', inplace=True)
            # shutil.rmtree(raw_dir)
        elif dir in stars_list:
            shutil.move(raw_dir, STARS_DIR)
            df_alerts.finkclass.replace(dir, 'custom_stars', inplace=True)
            # shutil.rmtree(raw_dir)
        elif dir == 'SN' or dir == 'SN%20candidate':  # NOTE: For SN, the finer TNS classes wouldn't be present at the folder level, all those will instead be combined inside these two folders. To get the actual TNS class, one can read the parquets inside these two folders and look at the `tnsclass` column.
            # Also NOTE: "(TNS) SN ..." may also be present in other folders like AGN, but those will not be given the `custom_sn` label. This is irrelevant for the unsupervised learning, but may become important for supervised classifications.
        #elif dir in sn_list:
            #if df_alerts.tnsclass.isin(sn_list):  # TODO: Not sure if this condition is needed. Sometimes the tnsclass in these cases may contain "Unknown" as well, so this condition removes those cases. But if it's needed or not is not entirely clear.
            shutil.move(raw_dir, SN_DIR)
            df_alerts.finkclass.replace(dir, 'custom_sn', inplace=True)
            # shutil.rmtree(raw_dir)
        #elif dir in simbad_galaxies_list:
        #    shutil.move(raw_dir, SIMBAD_GALAXIES_DIR)
        #    df_alerts.finkclass.replace(dir, 'custom_simbad_galaxies', inplace=True)
            # shutil.rmtree(raw_dir)
        else:
            print(f'Folder {dir} not in the alerts, skipping...')
    print('Done!')


    df_alerts_shape = df_alerts.shape

    ######################################## THESE ARE THE OLD CONDITIONS ########################################
    # After renaming the columns, preprocess the alerts based on some criteria
    # Select the objectIds (transients) that have more than or equal to three alerts in atleast one passband/filter.
    # Note that we mean more than three alerts in the time period in which the alerts are captured and not from the start of the survey.
    # See notes above.
    # For requiring three rather than two alerts, it's because if there are one or two alerts, you can fit anything to them with good accuracy.
    # Only when you have three points or more, can we fit something meaningful.
    #df_alerts = df_alerts.groupby('objectId').filter(
    #    lambda group: (len(group[group['fid'] == 1]) >= 3) or (len(group[group['fid'] == 2]) >= 3)
    #)
    #############################################################################################################

    # NOTE: Below line temporarily added. Remove after experiment done.
    #df_alerts = df_alerts[(df_alerts['finkclass'] == 'custom_sn') | (df_alerts['finkclass'] == 'Early SN Ia candidate')]

    # Select those having >=10 points in the light curve and at least 4 points in each band.
    df_alerts = df_alerts.groupby('objectId').filter(
            lambda group: (len(group) >= 10) and (len(group[group['fid'] == 1]) >= 4) and (len(group[group['fid'] == 2]) >= 4)  # and (len(group) <= 30)
    )

    print(f'{len(df_alerts)} alerts selected out of {df_alerts_shape[0]}')

    for objectId in df_alerts['objectId'].unique():
        pdf = df_alerts[df_alerts['objectId'] == objectId]
        assert (len(pdf) >= 10) and (len(pdf[pdf['fid'] == 1]) >= 4) and (len(pdf[pdf['fid'] == 2]) >= 4)
        #assert (len(pdf[pdf['fid'] == 1]) >= 3) or (len(pdf[pdf['fid'] == 2]) >= 3)

    print(f'No. of alerts (after preprocessing) = {len(df_alerts)}')
    print(f'No. of transients (after preprocessing) = {len(df_alerts["objectId"].unique())}')

    ################### Adding alerts manually #######################################
    #tns_processed_alerts = get_tns_tde_alerts()
    #df_alerts = pd.concat([df_alerts, tns_processed_alerts], ignore_index=True)
    ##################################################################################

    return df_alerts

df_alerts = preprocessing_alert_folders(TOPIC_PATH)
df_alerts.to_parquet(f'alerts_processed_{TOPIC_PATH.split("/")[-1].replace("-", "_")}'+'.parquet')

data_obj = prepare_data(df_alerts, dim=2, train_size=0.8, train_batch_size=8, convert_to_tensor=True)

torch.save(data_obj["train_dataloader"], 'train_dataloader.pth')
torch.save(data_obj["test_dataloader"], 'test_dataloader.pth')
torch.save(data_obj["val_dataloader"], 'val_dataloader.pth')
torch.save(data_obj["train_data_combined"], 'train_data_combined.pth')
torch.save(data_obj["val_data_combined"], 'val_data_combined.pth')
torch.save(data_obj["test_data_combined"], 'test_data_combined.pth')
np.save('total_objIds.npy', data_obj["total_objIds"])
np.save('train_objIds.npy', data_obj["train_objIds"])
np.save('val_objIds.npy', data_obj["val_objIds"])
#np.save('test_objIds.npy', data_obj["test_objIds"])
#np.save('total_objIds_encoded.npy', data_obj["total_objIds_encoded"])
np.save('total_common_finkclasses.npy', data_obj["total_common_finkclasses"])
np.save('duration_lcs.npy', data_obj["duration_lcs"])
np.save('min_max_magdiffs.npy', data_obj['min_max_magdiffs'])

"""
# Now save the finkclass for each objectId. The most common finkclass of all alerts of that object is taken.
# [0] because we assume only one finkclass will have the maximum occurence.
# If more than one finkclass have maximum occurence, all such finkclasses will be included.
#objIds_finkclass = []
#for objectId in df_alerts['objectId'].unique():
#    common_finkclasses = df_alerts[df_alerts['objectId'] == objectId]['finkclass'].mode()
#    for common_finkclass in common_finkclasses:  # generally len(common_finkclasses) is expected to be one only, but sometimes there may be multiple modes.
#        objIds_finkclass.append((objectId, common_finkclass))
#
#objIds_finkclass = np.array(objIds_finkclass)
#np.save('objIds_finkclass.npy', objIds_finkclass)
"""
