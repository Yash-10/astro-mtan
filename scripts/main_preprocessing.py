import os
import glob
import shutil
import numpy as np
import torch
from utils import read_alert
from prepare_data import prepare_data


TOPIC_PATH = '/home/ygondhal/ftransfer_ztf_2024-04-02_252737'


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

    # We collate come subclasses into a single class based on the below rule.
    agn_list = ['AGN','Blazar','BLLac','LINER','QSO','Seyfert', 'Seyfert_1', 'Seyfert_2']
    stars_list = ['EB*','CataclyV*','LMXB','RRLyr','RotV*','Star','WD*','low-mass*']
    simbad_galaxies_list = [
            "galaxy",
            "Galaxy",
            "EmG",
            "Seyfert",
            "Seyfert_1",
            "Seyfert_2",
            "BlueCompG",
            "StarburstG",
            "LSB_G",
            "HII_G",
            "High_z_G",
            "GinPair",
            "GinGroup",
            "BClG",
            "GinCl",
            "PartofG",
            "Compact_Gr_G",
            "IG",
            "PairG",
            "GroupG",
            "ClG",
            "SuperClG",
            "Void",
        ]

    AGN_DIR = os.path.join(f'{topic_path}', 'custom_agn')
    STARS_DIR = os.path.join(f'{topic_path}', 'custom_stars')
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
        elif dir in simbad_galaxies_list:
            shutil.move(raw_dir, SIMBAD_GALAXIES_DIR)
            df_alerts.finkclass.replace(dir, 'custom_simbad_galaxies', inplace=True)
            # shutil.rmtree(raw_dir)
        else:
            print(f'Folder {dir} not in the alerts, skipping...')
    print('Done!')


    df_alerts_shape = df_alerts.shape

    # After renaming the columns, preprocess the alerts based on some criteria
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

    print(f'No. of alerts (after preprocessing) = {len(df_alerts)}')
    print(f'No. of transients (after preprocessing) = {len(df_alerts["objectId"].unique())}')

    return df_alerts

df_alerts = preprocessing_alert_folders(TOPIC_PATH)
df_alerts.to_parquet(f'alerts_processed_{TOPIC_PATH.split("/")[-1].replace("-", "_")}'+'.parquet')

data_obj = prepare_data(df_alerts, dim=2, train_size=0.7, train_batch_size=32, convert_to_tensor=True)

torch.save(data_obj["train_dataloader"], 'train_dataloader.pth')
torch.save(data_obj["test_dataloader"], 'test_dataloader.pth')
torch.save(data_obj["val_dataloader"], 'val_dataloader.pth')
np.save('total_objIds.npy', data_obj["total_objIds"])
np.save('train_objIds.npy', data_obj["train_objIds"])
np.save('val_objIds.npy', data_obj["val_objIds"])
np.save('test_objIds.npy', data_obj["test_objIds"])
np.save('total_objIds_encoded.npy', data_obj["total_objIds_encoded"])
np.save('total_common_finkclasses.npy', data_obj["total_common_finkclasses"])

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
