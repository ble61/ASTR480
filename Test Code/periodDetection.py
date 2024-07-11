import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import lightkurve as lk

plt.rcParams.update({
    "font.size": 18,
    "font.family": "serif",
    "figure.autolayout": True,
    "axes.grid": False,
    # "xtick.minor.visible": True,
    # "ytick.minor.visible": True,
})


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


matchesDF = load_matches(22,1,3,8)

unqNames = pd.unique(matchesDF["NameMatch"])

periodDict = {}

badCount = 0

numObser = 2

for name in unqNames:
    nameDf = name_cut(matchesDF, name)
    
    
    if nameDf.shape[0]<numObser:
        badCount+=1
        print(f"{badCount}. not enough points (<{numObser}) for {name}")
        continue
    time = nameDf["mjd"].values
    flux = nameDf["flux"].values
    fluxErr = np.ones_like(flux)*1e-4
    period = detect_period(time,flux,fluxErr)
    periodDict[name] = period


# print(periodDict)
print(len(unqNames))

fig, ax = plt.subplots()

ax.scatter(np.arange(len(periodDict)),periodDict.values())
ax.set_ylabel("Period [days]")
ax.set_xlabel("Index")


maxPName = max(periodDict, key=periodDict.get)
maxP = periodDict[maxPName]
print(maxPName, maxP)