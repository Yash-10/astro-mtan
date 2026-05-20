import numpy as np
import pandas as pd
import logging

DISTANCE_THRESHOLD = np.inf  #: max threshold for distnr
SCORE_THRESHOLD = 0.4  #: max threshold for sgscore
CHINR_THRESHOLD = 2  #: max threshold for chinr
SHARPNR_MAX = 0.1  #: max value for sharpnr
SHARPNR_MIN = -0.13  #: min value for sharpnr
ZERO_MAG = 100.  #: default value for zero magnitude (a big value!)
TRIPLE_NAN = (np.nan, np.nan, np.nan)
MAGNITUDE_THRESHOLD = 13.2

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',)


def correction(magnr, magpsf, sigmagnr, sigmapsf, isdiffpos, oid=None):
    """
    Correction function. Implement of correction formula.

    :param magnr: Magnitude of nearest source in reference image PSF-catalog within 30 arcsec [mag]
    :type magnr: float

    :param magpsf: Magnitude from PSF-fit photometry [mag]
    :type magpsf: float

    :param sigmagnr: 1-sigma uncertainty in magnr within 30 arcsec [mag]
    :type sigmagnr: float

    :param sigmapsf: 1-sigma uncertainty in magpsf [mag]
    :type sigmapsf: float

    :param isdiffpos: 1 => candidate is from positive (sci minus ref) subtraction; 0 => candidate is from negative (ref minus sci) subtraction
    :type isdiffpos: int

    :return: Correction for magnitude, sigma and sigma_ext
    :rtype: tuple

    Example::

        (m_corr, s_corr, s_corr_ext) = correction(a, b, c, d, e)
    """
    if magnr < 0 or magpsf < 0:
        return TRIPLE_NAN

    try:
        aux1 = 10**(-0.4 * magnr)
        aux2 = 10**(-0.4 * magpsf)
        aux3 = aux1 + isdiffpos * aux2
        if aux3 > 0:
            magpsf_corr = -2.5 * np.log10(aux3)
            aux4 = aux2**2 * sigmapsf**2 - aux1**2 * sigmagnr**2

            if aux4 >= 0:
                sigmapsf_corr = np.sqrt(aux4) / aux3
            else:
                sigmapsf_corr = ZERO_MAG

            sigmapsf_corr_ext = aux2 * sigmapsf / aux3
        else:
            magpsf_corr = ZERO_MAG
            sigmapsf_corr = ZERO_MAG
            sigmapsf_corr_ext = ZERO_MAG

        return magpsf_corr, sigmapsf_corr, sigmapsf_corr_ext

    except Exception as e:
        logging.error('Object {}: {}'.format(oid, e))
        return TRIPLE_NAN


def apply_correction(candidate):
    """
    Correction function for a set of detections

    :param candidate: A dataframe with detections of a candidate.
    :type candidate: :py:class:`pd.DataFrame`

    :return: Wrapper for correction for magnitude, sigma and sigma_ext
    :rtype: tuple

    Example::

        (m_corr, s_corr, s_corr_ext) = correction(a, b, c, d, e)
    """
    isdiffpos = 1 if (candidate["isdiffpos"] in ["t", "1"]) else -1
    magnr = candidate["magnr"]
    magpsf = candidate['magpsf']
    sigmagnr = candidate['sigmagnr']
    sigmapsf = candidate['sigmapsf']

    magpsf_corr, sigmapsf_corr, sigmapsf_corr_ext = correction(magnr, magpsf, sigmagnr, sigmapsf, isdiffpos)

    return magpsf_corr, sigmapsf_corr, sigmapsf_corr_ext


def near_stellar(first_distnr, first_distpsnr1, first_sgscore1, first_chinr, first_sharpnr):
    """
    Get if object is near stellar

    :param first_distnr: Distance to nearest source in reference image PSF-catalog within 30 arcsec [pixels]
    :type first_distnr: :py:class:`float`

    :param first_distpsnr1: Distance of closest source from PS1 catalog; if exists within 30 arcsec [arcsec]
    :type first_distpsnr1: :py:class:`float`

    :param first_sgscore1: Star/Galaxy score of closest source from PS1 catalog 0 <= sgscore <= 1 where closer to 1 implies higher likelihood of being a star
    :type first_sgscore1: :py:class:`float`

    :param first_chinr: DAOPhot chi parameter of nearest source in reference image PSF-catalog within 30 arcsec
    :type first_chinr: :py:class:`float`

    :param first_sharpnr: DAOPhot sharp parameter of nearest source in reference image PSF-catalog within 30 arcsec
    :type first_sharpnr: :py:class:`float`


    :return: if the object is near stellar
    :rtype: tuple
    """

    nearZTF = 0 <= first_distnr < DISTANCE_THRESHOLD
    nearPS1 = 0 <= first_distpsnr1 < DISTANCE_THRESHOLD
    stellarPS1 = first_sgscore1 > SCORE_THRESHOLD
    stellarZTF = first_chinr < CHINR_THRESHOLD and SHARPNR_MIN < first_sharpnr < SHARPNR_MAX
    return nearZTF, nearPS1, stellarPS1, stellarZTF


def is_stellar(nearZTF, nearPS1, stellarPS1, stellarZTF):
    """
    Get if object is stellar

    :param nearZTF:
    :type nearZTF: bool

    :param nearPS1:
    :type nearPS1: bool

    :param stellarPS1:
    :type stellarPS1: bool

    :param stellarZTF:
    :type stellarZTF: bool

    :return: if the object is stellar
    :rtype: bool

    """
    return (nearZTF & nearPS1 & stellarPS1) | (nearZTF & ~nearPS1 & stellarZTF)


def is_dubious(corrected, isdiffpos, corr_magstats):
    """Get if object is dubious

    :param corrected:
    :type corrected: bool

    :param isdiffpos:
    :type isdiffpos: bool

    :param corr_magstats:
    :type corr_magstats: bool

    :return: if the object is dubious
    :rtype: bool
    """
    return (~corrected & (isdiffpos == -1)) | (corr_magstats & ~corrected) | (~corr_magstats & corrected)


def dmdt(magpsf_first, sigmapsf_first, nd_diffmaglim, mjd_first, nd_mjd):
    """
    Calculate dm/dt

    :param magpsf_first:
    :type magpsf_first: float

    :param sigmapsf_first:
    :type sigmapsf_first: float

    :param nd_diffmaglim:
    :type nd_diffmaglim: float

    :param mjd_first:
    :type mjd_first: float

    :param nd_mjd:
    :type nd_mjd: float

    :return: dm_sigma, dt, dmsigdt
    :rtype: tuple

    Example::

        dm_sigma, dt, dmsigdt = dmdt(magpsf_first,
                                     sigmapsf_first,
                                     nd.diffmaglim,
                                     mjd_first,
                                     nd.mjd)
    """
    dm_sigma = magpsf_first + sigmapsf_first - nd_diffmaglim
    dt = mjd_first - nd_mjd
    dmsigdt = (dm_sigma / dt)
    frame = {
        "dm_sigma": dm_sigma,
        "dt": dt,
        "dmsigdt": dmsigdt
    }
    df = pd.DataFrame(frame)
    return df


def apply_correction_df(df, calculate_dubious = False):
    """
    Correction function for a set of detections with the same object id and filter id. Use with pd.DataFrame.apply(this)

    :param df: A dataframe with detections of a candidate.
    :type df: :py:class:`pd.DataFrame`

    :return: A pandas dataframe with detections corrected
    :rtype: :py:class:`pd.DataFrame`

    Example::

        corrected = detections.groupby(["objectId", "fid"]).apply(apply_correction_df)
    """

    df.set_index("candid", inplace=True)

    df['isdiffpos'] = df['isdiffpos'].map({'t': 1., 'f': -1., '1': 1., '0': -1.})
    df["corrected"] = df["distnr"] < DISTANCE_THRESHOLD
    correction_results = df.apply(
        lambda x: correction(x.magnr, x.magpsf, x.sigmagnr, x.sigmapsf, x.isdiffpos, x.objectId)
        if x["corrected"]
        else (np.nan, np.nan, np.nan), axis=1, result_type="expand")

    df["magpsf_corr"], df["sigmapsf_corr"], df["sigmapsf_corr_ext"] = correction_results[0], correction_results[1], correction_results[2]
    df["mjdendref"] = df["jdendref"] - 2400000.5

    if calculate_dubious:
        corr_magstats = df.loc[df.index.min()]["corrected"]
        df["dubious"] = is_dubious(df["corrected"], df['isdiffpos'], corr_magstats)

    return df.drop(["objectId", "fid"], axis=1)


def get_flag_saturation(detections):
    detections = detections[detections.corrected]
    total = detections['magpsf_corr'].count()
    if total == 0:
        return np.nan
    satured = (detections["magpsf_corr"] < MAGNITUDE_THRESHOLD).sum()
    return satured/total


def get_flag_reference(detections, first_detection):
    if len(detections) == 0:
        return np.nan
    last_reference = detections["mjdendref"].max()
    return last_reference > first_detection


def apply_mag_stats(df, distnr=None, distpsnr1=None, sgscore1=None, chinr=None, sharpnr=None, flags=False):
    """
    :param df: A dataframe with corrected detections of a candidate.
    :type df: :py:class:`pd.DataFrame`

    :param distnr:
    :type distnr: float

    :param distpsnr1:
    :type distpsnr1: float

    :param sgscore1:
    :type sgscore1: float

    :param chinr:
    :type chinr: float

    :param sharpnr:
    :type sharpnr: float

    :param flags: If you want compute flags, set it like True
    :type flags: boolean

    :return: A pandas dataframe with magnitude statistics
    :rtype: :py:class:`pd.DataFrame`
    """
    response = {}
    # minimum and maximum candid
    idxmin = df.mjd.values.argmin()
    idxmax = df.mjd.values.argmax()

    df_min = df.iloc[idxmin]
    df_max = df.iloc[idxmax]

    # corrected at the first detection?
    response['corrected'] = df_min["corrected"]

    distnr = df_min.distnr if distnr is None else distnr
    distpsnr1 = df_min.distpsnr1 if distpsnr1 is None else distpsnr1
    sgscore1 = df_min.sgscore1 if sgscore1 is None else sgscore1
    chinr = df_min.chinr if chinr is None else chinr
    sharpnr = df_min.sharpnr if sharpnr is None else sharpnr

    response["nearZTF"], response["nearPS1"], response["stellarZTF"], response["stellarPS1"] = near_stellar(distnr,
                                                                                                            distpsnr1,
                                                                                                            sgscore1,
                                                                                                            chinr,
                                                                                                            sharpnr)
    response["stellar"] = is_stellar(response["nearZTF"], response["nearPS1"], response["stellarZTF"], response["stellarPS1"])
    # number of detections and dubious detections
    response["ndet"] = df.shape[0]
    response["ndubious"] = df.dubious.sum()

    # reference id
    rfids = df.rfid.unique().astype(np.float)
    rfids = rfids[~np.isnan(rfids)]
    response["nrfid"] = len(rfids)

    # psf magnitude statatistics
    response["magpsf_mean"] = df.magpsf.mean()
    response["magpsf_median"] = df.magpsf.median()
    response["magpsf_max"] = df.magpsf.max()
    response["magpsf_min"] = df.magpsf.min()
    response["sigmapsf"] = df.magpsf.std()
    response["magpsf_first"] = df_min.magpsf
    response["sigmapsf_first"] = df_min.sigmapsf
    response["magpsf_last"] = df_max.magpsf

    # psf corrected magnitude statatistics
    response["magpsf_corr_mean"] = df.magpsf_corr.mean()
    response["magpsf_corr_median"] = df.magpsf_corr.median()
    response["magpsf_corr_max"] = df.magpsf_corr.max()
    response["magpsf_corr_min"] = df.magpsf_corr.min()
    response["sigmapsf_corr"] = df.magpsf_corr.std()
    response["magpsf_corr_first"] = df_min.magpsf_corr
    response["magpsf_corr_last"] = df_max.magpsf_corr

    # corrected psf magnitude statistics
    response["magap_mean"] = df.magap.mean()
    response["magap_median"] = df.magap.median()
    response["magap_max"] = df.magap.max()
    response["magap_min"] = df.magap.min()
    response["sigmap"] = df.magap.std()
    response["magap_first"] = df_min.magap
    response["magap_last"] = df_max.magap

    # time statistics
    response["first_mjd"] = df_min.mjd
    response["last_mjd"] = df_max.mjd

    # flags
    if flags:
        response["saturation_rate"] = get_flag_saturation(df)
    return pd.Series(response)


def apply_objstats_from_correction(df, flags=False):
    """
    :param df: A dataframe with corrected detections of a candidate.
    :type df: :py:class:`pd.DataFrame`

    :param flags: If you want compute flags, set it like True
    :type flags: boolean

    :return: A pandas series with statistics of an object
    :rtype: :py:class:`pd.Series`
    """
    response = {}
    df_mjd = df.mjd
    idxmax = df_mjd.values.argmax()
    df_max = df.iloc[idxmax]
    df_ra = df.ra
    df_dec = df.dec
    response["ndethist"] = df_max.ndethist
    response["ncovhist"] = df_max.ncovhist
    response["mjdstarthist"] = df_max.jdstarthist - 2400000.5
    response["mjdendhist"] = df_max.jdendhist - 2400000.5
    response["meanra"] = df_ra.mean()
    response["meandec"] = df_dec.mean()
    response["sigmara"] = df_ra.std()
    response["sigmadec"] = df_dec.std()
    response["firstmjd"] = df_mjd.min()
    response["lastmjd"] = df_max.mjd
    response["deltamjd"] = response["lastmjd"] - response["firstmjd"]

    # flags
    if flags:
        response["diffpos"] = df.isdiffpos.min() > 0
        response["reference_change"] = get_flag_reference(df, response["firstmjd"])
    return pd.Series(response)

def apply_objstats_from_magstats(df):
    """
    :param df: A dataframe with magnitude statistics.
    :type df: :py:class:`pd.DataFrame`

    :return: A pandas series with statistics of an object
    :rtype: :py:class:`pd.Series`
    """
    response = {}
    response["nearZTF"] = df.nearZTF.all()
    response["nearPS1"] = df.nearPS1.all()
    response["stellar"] = df.stellar.all()
    response["corrected"] = df.corrected.all()
    response["ndet"] = df.ndet.sum()  # sum of detections in all bands
    response["ndubious"] = df.ndubious.sum()  # sum of dubious corrections in all bands
    fids = df.fid.unique()
    if 1 in fids and 2 in fids:
        df.set_index("fid", inplace=True)
        g = df.loc[1]
        r = df.loc[2]
        response["g-r_max"] = g["magpsf_min"] - r["magpsf_min"]  # 1=g ; 2=r
        response["g-r_max_corr"] = g["magpsf_corr_min"] - r["magpsf_corr_min"]
        response["g-r_mean"] = g["magpsf_mean"] - r["magpsf_mean"]
        response["g-r_mean_corr"] = g["magpsf_corr_mean"] - r["magpsf_corr_mean"]
    else:
        response["g-r_max"] = np.nan
        response["g-r_max_corr"] = np.nan
        response["g-r_mean"] = np.nan
        response["g-r_mean_corr"] = np.nan
    return pd.Series(response)


def apply_object_stats_df(corrected, magstats, step_name=None, flags=False):
    """
    :param corrected: A dataframe with corrected detections.
    :type corrected: :py:class:`pd.DataFrame`

    :param magstats: A dataframe with magnitude statistics.
    :type magstats: :py:class:`pd.DataFrame`

    :param step_name:
    :type step_name: string


    :return: Object statistics in a dataframe
    :rtype: :py:class:`pd.DataFrame`
    """
    basic_stats = corrected.groupby("objectId").apply(apply_objstats_from_correction, flags=flags)
    obj_magstats = magstats.groupby("objectId").apply(apply_objstats_from_magstats)
    basic_stats['step_id_corr'] = 'corr_bulk_0.0.1' if step_name is None else step_name
    return basic_stats.join(obj_magstats)


def do_dmdt(df, dt_min=0.5):
    """
    :param nd:  A dataframe with non detections.
    :type nd: :py:class:`pd.DataFrame`

    :param magstats:  A dataframe with magnitude statistics.
    :type magstats: :py:class:`pd.DataFrame`

    :param dt_min:
    :type dt_min: float

    :return: Compute of dmdt of an object
    :rtype: :py:class:`pd.Series`
    """
    response = {}
    df.reset_index(inplace=True)
    magstat_data = df.iloc[0]
    mjd_first = magstat_data.first_mjd
    mask = df.mjd < (mjd_first - dt_min)
    df_masked = df.loc[mask]

    response["close_nondet"] = df_masked.mjd.max() < df.loc[df.mjd < mjd_first, "mjd"].max()
    # is there some non-detection before the first detection
    if mask.sum() > 0:
        magpsf_first = magstat_data.magpsf_first
        sigmapsf_first = magstat_data.sigmapsf_first
        dmdts = dmdt(magpsf_first,
                     sigmapsf_first,
                     df_masked.diffmaglim,
                     mjd_first,
                     df_masked.mjd)
        idxmin = dmdts.dmsigdt.idxmin()
        min_dmdts = dmdts.loc[idxmin]
        min_dmdts = min_dmdts if isinstance(min_dmdts, pd.Series) else min_dmdts.iloc[0]
        min_nd = df.loc[idxmin]
        min_nd = min_nd if isinstance(min_nd, pd.Series) else min_nd.iloc[0]

        response["dmdt_first"] = min_dmdts.dmsigdt
        response["dm_first"] = magpsf_first - min_nd.diffmaglim
        response["sigmadm_first"] = sigmapsf_first - min_nd.diffmaglim
        response["dt_first"] = min_dmdts["dt"]
    else:
        response["dmdt_first"] = np.nan
        response["dm_first"] = np.nan
        response["sigmadm_first"] = np.nan
        response["dt_first"] = np.nan
    return pd.Series(response)


def do_dmdt_df(magstats, non_dets, dt_min=0.5):
    """
    :param magstats:  A dataframe with magnitude statistics.
    :type magstats: :py:class:`pd.DataFrame`

    :param non_dets:  A dataframe with non detections.
    :type non_dets: :py:class:`pd.DataFrame`


    :return: Compute of dmdt of an object in a dataframe
    :rtype: :py:class:`pd.DataFrame`
    """
    magstats.set_index(["objectId", "fid"],inplace=True)
    non_dets_magstats = non_dets.join(magstats, on=["objectId", "fid"], how="inner", rsuffix="_stats")

    apply_dmdt_df = lambda x: do_dmdt(x, dt_min=dt_min)
    result = non_dets_magstats.groupby(["objectId", "fid"]).apply(apply_dmdt_df)
    magstats.reset_index(inplace=True)
    return result
