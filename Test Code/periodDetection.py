import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import lightkurve as lk

import astropy.units as u
from astropy.timeseries import LombScargle as lsp

# %matplotlib widget 

plt.rcParams.update({
    "font.size": 18,
    "font.family": "serif",
    "figure.autolayout": True,
    "axes.grid": False,
    # "xtick.minor.visible": True,
    # "ytick.minor.visible": True,
})


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
    lsper = lsp(times*u.day, fluxes)#.autopower(samples_per_peak=50)
    freqs, powers = lsper.autopower(samples_per_peak=5)
    bestPow = np.max(powers)
    bestFreq = freqs[np.argmax(powers)]
    t_fit=np.linspace(times.min(),times.max(),1000)*u.day
    y_fit = lsper.model(t_fit, bestFreq)
    modelParams = lsper.model_parameters(bestFreq)
    f_a_prob = lsper.false_alarm_probability(np.max(powers),samples_per_peak=50)

    if plotting:
        fig, ax = plt.subplots(2, figsize = (8,10))

        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set(xlabel="Log Frequency [1/d]", ylabel="Power")
        ax[0].plot(freqs, powers, c="k", label="Periodogram")
        ax[0].scatter(bestFreq,bestPow, c="gold",marker="*", s=100, label=f"Best Frequency = {bestFreq.round(2)} /d \nFalse Alarm Probability = {f_a_prob.round(3)}")
        ax[0].legend()

        ax[1].set(xlabel="Time [MJD]", ylabel="Flux")
        ax[1].scatter(times, fluxes, label = "Light curve", c="tab:orange", marker = "d", s=10)
        ax[1].plot(t_fit,y_fit, ls="--", c="k", label="Model")


    return (1/bestFreq)/u.day, f_a_prob, modelParams

def load_matches(sector, cam, ccd, cut):
    matchesDF = pd.read_csv(f"./{sector}_{cam}_{ccd}_{cut}_InterpAndDetect_Matches.csv")
    matchesDF.drop(columns="Unnamed: 0", inplace=True)
    return matchesDF

def name_cut(df, name:str, colName:str="NameMatch"):
    "take a df and gives  only the values where the colName == name"
    return df.loc[df.index[df[colName]==name]]

numObser = 3 #*any less and NaNs come up, or it completely breaks

def compute_periods_lk(posDF):

    unqNames = pd.unique(posDF["NameMatch"]) #!name col better
    periodList_lk = []
    badCount = 0

    for name in unqNames:
        nameDf = name_cut(posDF, name)
        if nameDf.shape[0]<=numObser:
            badCount+=1
            # print(f"{badCount}. not enough points (<={numObser}) for {name}")
            continue
        time = nameDf["mjd"].values
        flux = nameDf["flux"].values
        fluxErr = np.ones_like(flux)*1e-4 #!Have to make up err for lk
        period_lk = detect_period_lk(time,flux,fluxErr)
        periodList_lk.append([name, period_lk,np.nan, np.nan]) #NaNs as LK doesn't spit those values out

    lkPeriods = pd.DataFrame(periodList_lk, columns=["Name", "Best Period","False Alarm Probability", "Model Parameters"])

    return lkPeriods, badCount

def compute_periods_ap(posDF):
    unqNames = pd.unique(matchesDF["NameMatch"])
    periodList_ap = []
    badCount = 0

    for name in unqNames:
        nameDf = name_cut(matchesDF, name)
        
        
        if nameDf.shape[0]<=numObser:
            badCount+=1
            # print(f"{badCount}. not enough points (<={numObser}) for {name}")
            continue
        time = nameDf["mjd"].values
        flux = nameDf["flux"].values

        period_ap, f_a_Prob, theta = detect_period_ap(time, flux)
        periodList_ap.append([name, period_ap, f_a_Prob, theta])

    apPeriods = pd.DataFrame(periodList_ap, columns=["Name", "Best Period", "False Alarm Probability", "Model Parameters"])

    return apPeriods, badCount

def singleNameLSP(posDf, name):
    cut = name_cut(posDf,name)
    detect_period_ap(cut["mjd"],cut["flux"],plotting=True)


matchesDF = load_matches(29,1,3,7)

lkPer, badCountlk = compute_periods_lk(matchesDF)

apPer, badCountap = compute_periods_ap(matchesDF)

numPerlk = len(lkPer.index)
numPerap = len(apPer.index)
print(badCountlk, numPerlk)
print(badCountap, numPerap)

fig, ax = plt.subplots()
ax.errorbar(lkPer.index, lkPer["Best Period"], fmt=".", c="tab:blue", capsize = 2, label="Lightkurve")

#!Not error in period, but level of uncertanty due to f_a_Prob
ax.errorbar(apPer.index, apPer["Best Period"], apPer["False Alarm Probability"]*10, fmt=".",c="tab:orange", capsize=2, label="Astropy")

ax.set_ylabel("Period [days]")
ax.set_xlabel("Index")
ax.legend()


maxPlk = lkPer.loc[lkPer["Best Period"].idxmax()]
maxPap = apPer.loc[apPer["Best Period"].idxmax()]
print(maxPlk)
print("")
print(maxPap)


singleNameLSP(matchesDF,maxPap["Name"])