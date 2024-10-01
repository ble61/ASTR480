"""Lots of queries timed out or overwhelmed Horizons, so asking again slower

Has to be run on Trevor in OzStar
"""

from astroquery.jplhorizons import Horizons
import pandas as pd
from astropy.io import fits
from astropy import wcs
from astropy.time import Time
import numpy as np

def _get_wcs_time(sector):
    basePath = f"/fred/oz335/TESSdata/Sector{sector}/Cam1/Ccd2/cut3of16/"
    wcsFname = f"{basePath}wcs.fits"
    targetWSC = fits.open(wcsFname)[0]
    # w = wcs.WCS(targetWSC.header)
    targetTime_i = Time(targetWSC.header["DATE-OBS"])
    return targetTime_i


def requery(sector):
    allPropDF = pd.read_csv(f"/fred/oz335/bleicester/Data/{sector}_All_Properties.csv")

    qTime = _get_wcs_time(sector)

    for j in range(len(allPropDF.index)):
        if str(allPropDF.at[j,"a"] == str(np.nan)):
            try:
                horizQ = Horizons(id=allPropDF.at[j,"Name"], epochs=qTime.jd,
                                location="500@10")
                elements = horizQ.elements()
                # subscriting to get number instead of list of 1 value????
                a = elements["a"].value[0]
                e = elements["e"].value[0]
                i = elements["incl"].value[0]
                H = elements["H"].value[0]

                allPropDF.at[j,"a"] = a
                allPropDF.at[j,"e"] = e
                allPropDF.at[j,"i"] = i
                allPropDF.at[j,"H"] = H

            except Exception as ex:
                print(f"Query failed because {ex}")
        else:
            continue

    allPropDF.to_csv(f"/fred/oz335/bleicester/Data/{sector}_All_Properties.csv")


requery(29)

    
