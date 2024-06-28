import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import lightkurve as lk

def detect_period(times, fluxes, fluxErrs):
    """
    Calculates the periodogram and returns the period at the max power.
    """
    lc =lk.LightCurve(data=None, time=times, flux=fluxes, flux_err=fluxErrs)
    lc = lc.normalize()
    pg = lc.to_periodogram()
    period = pg.period_at_max_power

    return period.value

def load_matches(sector, cam, ccd, cut):
    matchesDF = pd.read_csv(f"./{sector}_{cam}_{ccd}_{cut}_InterpAndDetect_Matches.csv")
    matchesDF.drop(columns="Unnamed: 0", inplace=True)
    return matchesDF

def name_cut(df, name:str, colName:str="NameMatch"):
    "take a df and gives  only the values where the colName == name"
    return df.loc[df.index[df[colName]==name]]


matchesDF = load_matches(29,1,3,7)

unqNames = pd.unique(matchesDF["NameMatch"])

periodDict = {}


for name in unqNames:
    nameDf = name_cut(matchesDF, name)
    
    if nameDf.shape[0]<2:
        continue
    time = nameDf["mjd"].values
    flux = nameDf["flux"].values
    fluxErr = np.ones_like(flux)*1e-4
    period = detect_period(time,flux,fluxErr)
    periodDict[name] = period


print(periodDict)

plt.scatter(np.arange(len(periodDict)),periodDict.values())
