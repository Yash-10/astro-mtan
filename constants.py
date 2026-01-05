# We collate come subclasses into a single class based on the below rule.
# RECALL: The "*" ASCII character is replaced by "%2A" in the URL-encoding format in all below class names wherever applicable. This is only to match the corresponding names of directories in the alert folder.
agn_list = ['AGN','Blazar','BLLac','LINER','QSO','Seyfert', 'Seyfert_1', 'Seyfert_2']
stars_list = ['EB%2A','LMXB','RRLyr','RotV%2A','Star','WD%2A','low-mass%2A']
# We decided not to use simbad_galaxies_list.
simbad_galaxies_list = [
        "galaxy",
        "Galaxy",
        "EmG",
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

# While the agn and stars list are used in the preprcessing of the alert folders, the sn_list below is not used since the alert folders don't contain these granular classes, but only "SN" and "SN candidate". See main_preprocessing.py for details. Recall that simbad_galaxies_list is no longer used, although kept for storage purposes.
sn_list = ["(TNS) SLSN-I","(TNS) SLSN-II","(TNS) SN","(TNS) SN I","(TNS) SN Ia","(TNS) SN Ia-91bg-like","(TNS) SN Ia-91T-like","(TNS) SN Ia-CSM","(TNS) SN Ia-pec","(TNS) SN Iax[02cx-like]","(TNS) SN Ib","(TNS) SN Ib-Ca-rich","(TNS) SN Ib-pec","(TNS) SN Ib/c","(TNS) SN Ibn","(TNS) SN Ic","(TNS) SN Ic-BL","(TNS) SN Ic-pec","(TNS) SN Icn","(TNS) SN II","(TNS) SN II-pec","(TNS) SN IIb","(TNS) SN IIL","(TNS) SN IIn","(TNS) SN IIn-pec","(TNS) SN IIP"]

# ObjectIds from one of custom_sn or Early SN Ia candidate.
# These need to be removed because their visual inspection shows more than one parallel light curve for the same band. This arises because of using different template images.
to_remove_objIds = ['ZTF18abtsuqv', 'ZTF18aauselm', 'ZTF18abtzzka', 'ZTF18abvkswd', 'ZTF18aagtdce', 'ZTF18aaiscil', 'ZTF18aaapivw', 'ZTF18aaibcxu', 'ZTF18abtgtpn', 'ZTF18abtrvkm']  # TODO: This list needs to be updated as and when we find more such examples.
# TODO: Note that apart from SN, we also need to find TDE examples having "ZTF18" on object ID.
