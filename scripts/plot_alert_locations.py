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

from astropy.coordinates import SkyCoord
coords_alerts = SkyCoord(df_alerts['ra'], df_alerts['dec'], unit='deg').galactic

fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(projection='aitoff')
plt.scatter(coords_alerts.l.wrap_at('180d').radian, coords_alerts.b.radian, color='grey', alpha=1, marker='.',label='alerts')
x = np.arange(-180, 180, 0.1)
plt.plot(x, [20*np.pi/180]*len(x), color='grey', alpha=0.5,linestyle='dashed')
plt.plot(x, [-20*np.pi/180]*len(x), color='grey', alpha=0.5,linestyle='dashed')
plt.grid();
plt.legend()
plt.savefig('aitoff_1.png', bbox_inches='tight', dpi=200)

fig = plt.figure(figsize=(15, 10))
# ax = plt.subplot(projection='aitoff')
hp.projview(dustmap, nest=True, norm='hist', unit='MJy/sr', projection_type="aitoff")
plt.scatter(coords_alerts.l.wrap_at('180d').radian, coords_alerts.b.radian, color='white', alpha=.1, marker='o',s=10,label='alerts')
plt.savefig('aitoff_2.png', bbox_inches='tight', dpi=200)

