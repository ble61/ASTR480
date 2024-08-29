import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import lightkurve as lk

import astropy.units as u
from astropy.timeseries import LombScargle as lsp

%matplotlib widget 

plt.rcParams.update({
    "font.size": 18,
    "font.family": "serif",
    "figure.autolayout": True,
    "axes.grid": False,
    # "xtick.minor.visible": True,
    # "ytick.minor.visible": True,
})

#TODO Nyquist criterion
#TODO LC database

def detect_period_lk(times, fluxes, fluxErrs):
    """
    Calculates the periodogram using LightKurve and returns the period at the max power.
    """
    lc =lk.LightCurve(data=None, time=times, flux=fluxes, flux_err=fluxErrs)
    # lc = lc.normalize()
    pg = lc.to_periodogram()
    period = pg.period_at_max_power

    return period.value

def detect_period_ap(times, fluxes, plotting = False):
    """
    Calculates the periodogram with astropy and returns the period at the max power.
    """
    #* Seems to be sec 27 and 56 when it changes
    #12hr queries constant
    if sector <27: 
        nyquistP = 1*60*60*u.s #1/2hr ffi, so 1hr nyqist
    elif sector>=27 and sector<56:
        nyquistP = 20*60*u.s #1/6hr (10 min) ffi
    else: 
        nyquistP = 400*u.s   # 200 s ffi
    

    minFreq = 1/(0.1*(times.max()-times.min())*u.day).to(u.s)
    maxFreq = 1/nyquistP

    lsper = lsp((u.Quantity(times, u.day)).to(u.s), fluxes)
    freqs, powers = lsper.autopower(samples_per_peak=5, minimum_frequency=minFreq, maximum_frequency=maxFreq)
    bestPow = np.max(powers)
    bestFreq = freqs[np.argmax(powers)]

    t_fit=(np.linspace(times.min(),times.max(),1000)*u.day).to(u.s)
    y_fit = lsper.model(t_fit, bestFreq)
    modelParams = lsper.model_parameters(bestFreq)
    f_a_prob = lsper.false_alarm_probability(np.max(powers),samples_per_peak=50, minimum_frequency=minFreq, maximum_frequency=maxFreq)

    if plotting:
        fig, ax = plt.subplots(2, figsize = (8,10))

        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set(xlabel="Log Frequency", ylabel="Power")
        ax[0].plot(freqs, powers, c="k", label="Periodogram")
        ax[0].scatter(bestFreq,bestPow, c="gold",marker="*", s=100, label=f"Best Frequency = {bestFreq.round(7)} \nFalse Alarm Probability = {f_a_prob.round(3)}")
        ax[0].legend()

        ax[1].set(xlabel="Time [MJD]", ylabel="Flux")
        ax[1].scatter(times, fluxes, label = "Light curve", c="tab:orange", marker = "d", s=10)
        ax[1].plot(t_fit.to(u.day),y_fit, ls="--", c="k", label="Model")


    return (1/bestFreq)/u.s, f_a_prob, modelParams


def load_interps(sector, cam, ccd, cut):
    interpDf = pd.read_csv(f"./InterpolatedQuerryResult_{sector}_{cam}_{ccd}_{cut}.csv")
    interpDf.drop(columns="Unnamed: 0", inplace=True)
    return interpDf

def load_interps(sector, cam, ccd, cut):
    lcDF = pd.read_csv(f"./Interps_with_lc_{sector}_{cam}_{ccd}_{cut}.csv")
    lcDF.drop(columns="Unnamed: 0", inplace=True)
    return lcDF

def load_matches(sector, cam, ccd, cut):
    matchesDF = pd.read_csv(f"./{sector}_{cam}_{ccd}_{cut}_InterpAndDetect_Matches.csv")
    matchesDF.drop(columns="Unnamed: 0", inplace=True)
    return matchesDF

def name_cut(df, name:str, colName:str="NameMatch"):
    "take a df and gives  only the values where the colName == name"
    toReturn = df.loc[df.index[df[colName]==name]]
    toReturn.reset_index(drop=True, inplace=True)
    return toReturn

numObser = 3 #*any less and NaNs come up, or it completely breaks
#TODO, what should it be for a reliable P

def compute_periods_lk(posDF):

    unqNames = pd.unique(posDF["Name"]) #!name col better
    periodList_lk = []
    badCount = 0

    for name in unqNames:
        nameDf = name_cut(posDF, name, colName="Name")
        if nameDf.shape[0]<=numObser:
            badCount+=1
            # print(f"{badCount}. not enough points (<={numObser}) for {name}")
            continue
        time = nameDf["epoch"].values
        flux = nameDf["Flux"].values
        fluxErr = np.ones_like(flux)*1e-4 #!Have to make up err for lk
        period_lk = detect_period_lk(time,flux,fluxErr)
        periodList_lk.append([name, period_lk,np.nan, np.nan]) #NaNs as LK doesn't spit those values out

    lkPeriods = pd.DataFrame(periodList_lk, columns=["Name", "Best Period","False Alarm Probability", "Model Parameters"])

    return lkPeriods, badCount

def compute_periods_ap(posDF):
    unqNames = pd.unique(interpLcDF["Name"])
    periodList_ap = []
    badCount = 0

    for name in unqNames:
        nameDf = name_cut(interpLcDF, name, colName="Name")
        
        
        if nameDf.shape[0]<=numObser:
            badCount+=1
            # print(f"{badCount}. not enough points (<={numObser}) for {name}")
            continue
        time = nameDf["epoch"].values
        flux = nameDf["Flux"].values

        period_ap, f_a_Prob, theta = detect_period_ap(time, flux)
        periodList_ap.append([name, period_ap, period_ap/(60*60), period_ap/(60*60*24), f_a_Prob, theta])

    apPeriods = pd.DataFrame(periodList_ap, columns=["Name", "Best Period [Seconds]", "Best Period [Hours]","Best Period [Days]",  "False Alarm Probability", "Model Parameters"])

    return apPeriods, badCount

def singleNameLSP(posDf, name):
    cut = name_cut(posDf,name, colName="Name")
    detect_period_ap(cut["epoch"],cut["Flux"],plotting=True)


sector = 22
cam = 1
ccd = 3
cut = 7


interpLcDF = load_interps(22,1,3,7)


singleNameLSP(interpLcDF," Ruff ")


# lkPer, badCountlk = compute_periods_lk(interpLcDF)

apPer, badCountap = compute_periods_ap(interpLcDF)


# numPerlk = len(lkPer.index)
numPerap = len(apPer.index)
# print(badCountlk, numPerlk)
print(badCountap, numPerap)

fig, ax = plt.subplots()
# ax.errorbar(lkPer.index, lkPer["Best Period"], fmt=".", c="tab:blue", capsize = 2, label="Lightkurve")

#!Not error in period, but level of uncertanty due to f_a_Prob
ax.errorbar(apPer.index, apPer["Best Period [Days]"], apPer["False Alarm Probability"]*10, fmt=".",c="tab:orange", capsize=2, label="Astropy")


ax.set_ylabel("Period [days]")
ax.set_xlabel("Index")
ax.legend()


# maxPlk = lkPer.loc[lkPer["Best Period"].idxmax()]
maxPap = apPer.loc[apPer["Best Period [Seconds]"].idxmax()]
# print(maxPlk)
print("")
print(maxPap)


# singleNameLSP(interpLcDF,maxPap["Name"])



# minPlk = lkPer.loc[lkPer["Best Period"].idxmin()]
minPap = apPer.loc[apPer["Best Period [Seconds]"].idxmin()]
print("")
# print(minPlk)
print("")
print(minPap)


# singleNameLSP(interpLcDF,minPap["Name"])


singleNameLSP(interpLcDF," Ruff ")



