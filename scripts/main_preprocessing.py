import os
import glob
import shutil
import numpy as np
import pandas as pd
import torch
from urllib.parse import unquote
from utils import read_alert
from prepare_data import prepare_data, get_tns_tde_alerts
from constants import agn_list, stars_list, sn_list, to_remove_objIds

TOPIC_PATH = '/home/ygondhal/ftransfer_ztf_2024-05-27_433174_copy'


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

    #AGN_DIR = os.path.join(f'{topic_path}', 'custom_agn')
    #STARS_DIR = os.path.join(f'{topic_path}', 'custom_stars')
    #SN_DIR = os.path.join(f'{topic_path}', 'custom_sn')
    #SIMBAD_GALAXIES_DIR = os.path.join(f'{topic_path}', 'custom_simbad_galaxies')
    #os.makedirs(AGN_DIR, exist_ok=True)
    #os.makedirs(STARS_DIR, exist_ok=True)
    #os.makedirs(SIMBAD_GALAXIES_DIR, exist_ok=True)

    print('Starting arranding subfolders...')
    # NOTE: The finkclass column in the alerts at the end of the function come from the folder itself. If you read alerts of a specific folder, you will see there is no finkclass column.
    for raw_dir in glob.glob(DIRS):
        dir_ = raw_dir.split('/')[-1].split('finkclass=')[-1]
        if dir_ in agn_list:
            #shutil.move(raw_dir, AGN_DIR)  # NOTE: Now alerts are not transferred to a separate directory since that is less flexible when we want to assign custom_.. class not based on finkclass, e.g., tnsclass. Moving folders directly in this case is not possible.
            df_alerts.finkclass.replace(unquote(dir_), 'custom_agn', inplace=True)
            # shutil.rmtree(raw_dir)
        elif dir_ in stars_list:
            #shutil.move(raw_dir, STARS_DIR)
            df_alerts.finkclass.replace(unquote(dir_), 'custom_stars', inplace=True)
            # shutil.rmtree(raw_dir)
        elif dir_ == 'SN' or dir_ == 'SN%20candidate':
            # NOTE: For SN, the finer TNS classes wouldn't be present at the folder level, all those will instead be combined inside these two folders. To get the actual TNS class, one can read the parquets inside these two folders and look at the `tnsclass` column.
            # Also NOTE: "(TNS) SN ..." may also be present in other folders like AGN, but those will not be given the `custom_sn` label. This is irrelevant for the unsupervised learning, but may become important for supervised classifications.
            #if df_alerts.tnsclass.isin(sn_list):  # TODO: Not sure if this condition is needed. Sometimes the tnsclass in these cases may contain "Unknown" as well, so this condition removes those cases. But if it's needed or not is not entirely clear.
            """
            shutil.move(raw_dir, SN_DIR)
            df_alerts.finkclass.replace(unquote(dir_), 'custom_sn', inplace=True)
            """
            pass
            # shutil.rmtree(raw_dir)
        #elif dir in simbad_galaxies_list:
        #    shutil.move(raw_dir, SIMBAD_GALAXIES_DIR)
        #    df_alerts.finkclass.replace(dir, 'custom_simbad_galaxies', inplace=True)
            # shutil.rmtree(raw_dir)
        else:
            print(f'Folder {dir_} not in the alerts, skipping...')

    # NOTE: For SN, since custom_sn really is assigned based on TNS class, we do the below operation so that any alert not with either SN or SN candidate finkclass can still be added to custom_sn if it has one of the TNS SN classes.
    def f(row):
        return 'custom_sn' if row['tnsclass'] in sn_list else row['finkclass']
     
    df_alerts['finkclass'] = df_alerts.apply(lambda row: f(row), axis = 1)
    #df_alerts.loc[df_alerts.tnsclass.isin(sn_list), 'finkclass'] = 'custom_sn'

    print('finkclass value_counts after first processing...')
    print(df_alerts['finkclass'].value_counts())

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

    # NOTE: Below line temporarily adds further selections to only select a subset of data. Remove/comment after experiment done.
    # 1. For SN dataset (assuming polling is done for all TNS SN classes and finkclass Early SN Ia candidate.
    #df_alerts = df_alerts[(df_alerts['finkclass'] == 'custom_sn') | (df_alerts['finkclass'] == 'Early SN Ia candidate')]
    # 2. For AGN dataset (assuming polling is done for all classes in agn_list, whether TNS or SIMBAD, whatever classes are available in the data transfer service online.
    #df_alerts = df_alerts[df_alerts['finkclass'] == 'custom_agn']
    # 3. For SN + AGN dataset
    df_alerts = df_alerts[(df_alerts['finkclass'] == 'custom_sn') | (df_alerts['finkclass'] == 'Early SN Ia candidate') | (df_alerts['finkclass'] == 'custom_agn')]

    # Select those having >=10 points in the light curve and at least 4 points in each band.
    df_alerts = df_alerts.groupby('objectId').filter(
            lambda group: (len(group) >= 10) and (len(group[group['fid'] == 1]) >= 4) and (len(group[group['fid'] == 2]) >= 4)  # and (len(group) <= 30)
    )

    print(f'{len(df_alerts)} alerts selected out of {df_alerts_shape[0]}')

    for objectId in df_alerts['objectId'].unique():
        pdf = df_alerts[df_alerts['objectId'] == objectId]
        assert (len(pdf) >= 10) and (len(pdf[pdf['fid'] == 1]) >= 4) and (len(pdf[pdf['fid'] == 2]) >= 4)
        #assert (len(pdf[pdf['fid'] == 1]) >= 3) or (len(pdf[pdf['fid'] == 2]) >= 3)

    ################### Adding alerts manually #######################################
    #tns_processed_alerts = get_tns_tde_alerts()
    #df_alerts = pd.concat([df_alerts, tns_processed_alerts], ignore_index=True)
    ##################################################################################

    # APPLY FURTHER SELECTION CRITERIA
    # 1. Some examples are manually removed. See constants.py for details. These are parallel-lc-same-band examples.
    df_alerts = df_alerts[~df_alerts['objectId'].isin(to_remove_objIds)]
    # 2. Remove ZTF18.. object IDs with one of the TNS SN classification.
    df_alerts = df_alerts[~((df_alerts['objectId'].str.contains('ZTF18')) & (df_alerts['tnsclass'].isin(sn_list)))]

    print(f'No. of alerts (after preprocessing) = {len(df_alerts)}')
    print(f'No. of transients (after preprocessing) = {len(df_alerts["objectId"].unique())}')

    print('finkclass and tnsclass value_counts [FINAL]...')
    print(df_alerts['finkclass'].value_counts())
    print(df_alerts['tnsclass'].value_counts())

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
np.save('test_objIds.npy', data_obj["test_objIds"])
#np.save('total_objIds_encoded.npy', data_obj["total_objIds_encoded"])
np.save('total_common_finkclasses.npy', data_obj["total_common_finkclasses"])
np.save('duration_lcs.npy', data_obj["duration_lcs"])
np.save("num_datapoints_lcs.npy", data_obj["seq_len_all"])
np.save('min_max_magdiffs.npy', data_obj['min_max_magdiffs'])
np.save('min_max_mags.npy', data_obj['min_max_mags'])

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
