def plot_num_datapoints_lc_all():
    import numpy as np
    import pandas as pd
    from utils import get_lc

    df_alerts = pd.read_parquet('alerts_processed_ftransfer_ztf_2024_04_02_252737.parquet')

    num_data_points_all = []

    for objId in df_alerts['objectId'].unique():
        data = get_lc(df_alerts, objId, make_first_time_zero=False, convert_to_tensor=False)
        # data = (name, times, observation_data, observation_mask, common_finkclasses)
        num_data_points_all.append(len(data[1]))

    import matplotlib.pyplot as plt
    plt.hist(num_data_points_all, histtype='step')
    plt.yscale('log')
    plt.xlabel('#datapoints in light curve')
    plt.ylabel('Counts')
    plt.axvline(x=np.mean(num_data_points_all), linestyle='--', c='black', label='mean')
    plt.axvline(x=np.median(num_data_points_all), linestyle='--', c='red', label='median')
    plt.legend()
    plt.savefig('num_datapoints_all.png', bbox_inches='tight')

def plot_alert_locations():
    import requests
    import io
    import astropy.coordinates as coord
    import astropy.units as u
    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    APIURL = 'https://fink-portal.org'

    from astropy.io import fits 
    from astropy.utils.data import download_file
    import healpy as hp
    from astropy.coordinates import SkyCoord

    from constants import agn_list, stars_list, simbad_galaxies_list


    # Downloads the HI data in a fits file format
    dust_mapfile = download_file(
        'https://irsa.ipac.caltech.edu/data/Planck/release_1/all-sky-maps/maps/COM_CompMap_dust-commrul_0256_R1.00.fits',
        cache=True, show_progress=True)

    def clean_dust_map(map):
        """Replaces non-positive pixels with smallest positive pixel value."""
        pos = map[map > 0]
        posmin = pos.min()
        map[map <= 0] = 0.# posmin
        return map
    dust = fits.open(dust_mapfile)
    dustmap = dust[1].data['I']
    dustmap = clean_dust_map(dustmap)

    df_alerts = pd.read_parquet('alerts_processed_ftransfer_ztf_2024_04_02_252737.parquet')

    df_alerts_agn_stars = df_alerts[df_alerts['finkclass'].isin(agn_list + stars_list)]
    df_alerts_simbad_galaxies = df_alerts[df_alerts['finkclass'].isin(simbad_galaxies_list)]
    df_alerts_cv = df_alerts[df_alerts['finkclass'] == 'CataclyV*']
    df_alerts_tns_tde = df_alerts[df_alerts['finkclass'] == 'TNS (TDE)']
    df_alerts_sn = df_alerts[df_alerts['finkclass'] == 'SN']

    alerts_classes = [df_alerts_agn_stars, df_alerts_simbad_galaxies, df_alerts_cv, df_alerts_tns_tde, df_alerts_sn]
    alerts_classes_names = ['agn_stars', 'simbad_galaxies', 'CV', 'TNS (TDE)', 'SN']
    colors = ['#d95f0e', '#1f78b4', '#df65b0', '#33a02c', 'gray']

    from astropy.coordinates import SkyCoord

    fig = plt.figure(figsize=(15, 10))
    ax = plt.subplot(projection='aitoff')

    for i, ac in enumerate(alerts_classes):
        coords_alerts = SkyCoord(ac['ra'], ac['dec'], unit='deg').galactic
        plt.scatter(coords_alerts.l.wrap_at('180d').radian, coords_alerts.b.radian, color=f'{colors[i]}', alpha=1, marker='.',label=f'{alerts_classes_names[i]}', s=72)
        x = np.arange(-180, 180, 0.1)
        plt.plot(x, [20*np.pi/180]*len(x), color='grey', alpha=0.5,linestyle='dashed')
        plt.plot(x, [-20*np.pi/180]*len(x), color='grey', alpha=0.5,linestyle='dashed')
    plt.grid();
    plt.legend()
    plt.savefig('aitoff_1.png', bbox_inches='tight', dpi=200)

    fig = plt.figure(figsize=(15, 10))
    # ax = plt.subplot(projection='aitoff')
    hp.projview(dustmap, nest=True, norm='hist', unit='MJy/sr', projection_type="aitoff")

    for i, ac in enumerate(alerts_classes):
        coords_alerts = SkyCoord(ac['ra'], ac['dec'], unit='deg').galactic
        plt.scatter(coords_alerts.l.wrap_at('180d').radian, coords_alerts.b.radian, color='grey', alpha=.1, marker='o',s=10,label=f'{alerts_classes_names[i]}')
    plt.savefig('aitoff_2.png', bbox_inches='tight', dpi=200)


